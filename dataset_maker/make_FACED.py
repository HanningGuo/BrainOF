import os
import h5py
import scipy
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed eeg data
        eeg = pd.read_pickle(fname)
        # resample brain signal to 200 HZ
        eeg = scipy.signal.resample(eeg, 6000, axis=2)
        # segment the whole signals into 10 seconds sample
        signals = np.vstack([eeg[:, :, :2000], eeg[:, :, 2000:4000], eeg[:, :, 4000:]])

        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "FACED_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/FACED/Processed_data"
    save_path = "/path/to/BrainSignalDataset/FACED"

    # prepare the train dataset
    sub_list = []
    for sub in os.listdir(root):
        if os.path.isfile(os.path.join(root, sub)):
            sub_list.append((os.path.join(root, sub), 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
