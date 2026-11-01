import os
import lmdb
import torch
import pickle
import warnings
from pathlib import Path
from yacs.config import CfgNode
from torch.utils.data import Dataset
from typing import List, Union, Callable


class BrainSignalDataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            split: str = 'train',
            datasets: Union[str, List, None] = None,
            transform: Callable = None,
            target_transform: Callable = None,
            modality_cfg: CfgNode =None,
            pretraining: bool = False,
    ):
        """
        Brain signal dataset

        Args:
            root (str or pathlib.Path): Root directory of the brain signal.
            split (str, optional): The dataset split, supports 'train', 'val' or 'test'.
            datasets (str or list, optional): The datasets to be included
            transform (callable, optional): A function/transform that takes in the signal
                and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            modality_cfg (CfgNode, optional): dataset modality. If not specify it,
                the dataset is defaulted as unknown modality
        """
        self.root = root
        self.split = split
        self.env = {}
        self.pretraining = pretraining
        assert self.split in ['train', 'val', 'test']

        if datasets is None:
            datasets = os.listdir(root)
        elif isinstance(datasets, str):
            datasets = [datasets]
        self.datasets = datasets

        self.keys = []
        for dataset in datasets:
            if os.path.isdir(os.path.join(root, dataset)):
                path = os.path.join(root, dataset, split, 'BrainSignal.lmdb')
                self.env[dataset] = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
                with self.env[dataset].begin(write=False) as txn:
                    self.keys.extend(pickle.loads(txn.get('__keys__'.encode("ascii"))))

        self.transform = transform
        self.target_transform = target_transform
        self.modality = modality_cfg

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        dataset = self.keys[idx]['dataset']
        key = self.keys[idx]['key']

        with self.env[dataset].begin(write=False) as txn:
            sample = pickle.loads(txn.get(key.encode("ascii")))
            signal = sample['signal']
            target = sample['y']
            if not self.pretraining:
                if 'CamCAN' in dataset:
                    target = [target['age'][0]]
                elif 'LEMON' in dataset:
                    target = target[0]

        if dataset in self.modality.fMRI:
            modality = [0]   # fMRI
        elif dataset in self.modality.EEG:
            modality = [1]   # EEG
        elif dataset in self.modality.MEG:
            modality = [2]   # MEG
        else:
            modality = [3]  # unknown modality
            warnings.warn(f"The supported modalities are fMRI, EEG and MEG. If {dataset} belongs to one of these modalities, please define it explicitly in the config file. Otherwise, you can ignore the warning!", UserWarning)

        if self.transform:
            signal, mask, meta_data = self.transform(signal)
        if self.target_transform:
            target = self.target_transform(target)

        if self.pretraining:
            return torch.FloatTensor(signal), torch.FloatTensor(mask), torch.IntTensor(modality), meta_data
        else:
            return torch.FloatTensor(signal), torch.FloatTensor(target), torch.FloatTensor(mask)

    def extra_repr(self):
        if len(self.datasets) == 1:
            description = f'{self.datasets[0]}({len(self)})'
        else:
            description = f'{self.split}({len(self)})'
        return description
