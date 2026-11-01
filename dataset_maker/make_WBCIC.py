import os
import h5py
import numpy as np
import pymatreader
from typing import Union
from pathlib import Path


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed eeg data
        eeg_prep = pymatreader.read_mat(fname)['data']
        # get 4 seconds sample
        signals = np.transpose(eeg_prep, (2, 0, 1))

        for idx, signal in enumerate(signals):
            if '2C' in fname:
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_2C_sample{idx:04d}.h5'
            elif '3C' in fname:
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_3C_sample{idx:04d}.h5'
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "WBCIC_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/WBCIC/derivatives"
    save_path = "/path/to/BrainSignalDataset/WBCIC"

    # prepare the train and val dataset
    sub_list = []
    for dataset in os.listdir(root):
        if not dataset.startswith('.'):
            for sub in os.listdir(os.path.join(root, dataset)):
                if sub.startswith('sub'):
                    for sess in os.listdir(os.path.join(root, dataset, sub)):
                        if not sess.startswith('.'):
                            for file in os.listdir(os.path.join(root, dataset, sub, sess, 'eeg')):
                                sub_path = os.path.join(root, dataset, sub, sess, 'eeg', file)
                                if not file.startswith('.'):
                                    sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
