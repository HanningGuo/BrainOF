import os
import h5py
import numpy as np
from typing import Union
from pathlib import Path


ch_names = ['F1', 'F2', 'FC1', 'FC2', 'C1', 'C2', 'CP1', 'CP2', 'P1', 'P2',
            'FP1', 'FP2', 'AF3', 'AF4', 'F3', 'F4', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'P3', 'P4', 'PO3', 'PO4', 'O1', 'O2',
            'F5', 'F6', 'FC5',  'FC6', 'C5',  'C6', 'CP5',  'CP6', 'P5', 'P6', 'PO5', 'PO6',
            'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P8', 'PO7', 'PO8', 'CB1', 'CB2',
            'FPZ', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ']
ch_idx = [8, 10, 17, 19, 26, 28, 35, 37, 44, 46, 0, 2, 3, 4, 7, 11, 16, 20, 25, 29, 34, 38, 43, 47, 52, 54, 58, 60, 6, 12, 15, 21, 24, 30, 33, 39, 42, 48, 51, 55, 5, 13, 14, 22, 23, 31, 32, 40, 41, 49, 50, 56, 57, 61, 1, 9, 18, 27, 36, 45, 53, 59]

origin_ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
                   'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
                   'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                   'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
                   'CPZ', 'CP2',  'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                   'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
                   'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_prep = np.load(fname)
        nth = 0
        for eeg in eeg_prep:
            # reorder those channel with specific order
            eeg = eeg[np.array(ch_idx), :]
            for start_idx in range(0, eeg.shape[-1], 2000):
                nth += 1
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{nth:04d}.h5'
                signal = eeg[:, start_idx:start_idx+2000]
                target = ''
                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "seed_dv_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/SEED-DV/EEG"
    save_path = "/path/to/BrainSignalDataset/SEED_DV"

    # prepare the train dataset
    sub_list = []
    for sub in os.listdir(root):
        if sub.endswith("npy") and not sub.startswith("."):
            sub_path = os.path.join(root, sub)
            if os.path.isfile(sub_path):
                sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
