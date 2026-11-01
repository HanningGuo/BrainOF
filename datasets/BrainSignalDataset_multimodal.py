import os
import lmdb
import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Union, Callable


class BrainSignalDataset_multimodal(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            split: str = 'train',
            datasets: Union[str, None] = None,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        """
        Brain signal dataset

        Args:
            root (str or pathlib.Path): Root directory of the brain signal.
            split (str, optional): The dataset split, supports 'train', 'val' or 'test'.
            datasets (str, optional): The datasets to be included
            transform (callable, optional): A function/transform that takes in the signal
                and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """
        self.root = root
        self.split = split
        self.datasets = datasets
        assert self.split in ['train', 'val', 'test']

        self.keys = []
        if os.path.isdir(os.path.join(root, self.datasets)):
            path = os.path.join(root, self.datasets, split, 'BrainSignal.lmdb')
            self.env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.keys.extend(pickle.loads(txn.get('__keys__'.encode("ascii"))))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        dataset = self.keys[idx]['dataset']
        key = self.keys[idx]['key']

        with self.env.begin(write=False) as txn:
            sample = pickle.loads(txn.get(key.encode("ascii")))
            signal1, signal2 = sample['signal']
            target = sample['y']
            if 'CamCAN' in dataset:
                target = [target['age'][0]]
            elif 'LEMON' in dataset:
                target = target[0]

        if self.transform:
            signal1, mask1, _ = self.transform(signal1)
            signal2, mask2, _ = self.transform(signal2)
        if self.target_transform:
            target = self.target_transform(target)

        return (torch.FloatTensor(signal1), torch.FloatTensor(signal2)), torch.FloatTensor(target), (torch.FloatTensor(mask1), torch.FloatTensor(mask2))

    def extra_repr(self):
        return f'{self.datasets}({len(self)})'
