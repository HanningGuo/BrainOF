import os
import h5py
import pymatreader
from typing import Union
from pathlib import Path


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed eeg data
        eeg_prep = pymatreader.read_mat(fname)
        # segment the signals into 10 seconds sample
        signals = []
        for key in eeg_prep:
            if 'eeg' in key:
                for idx in range(eeg_prep[key].shape[1]//2000):
                    signals.append(eeg_prep[key][:, idx*2000:(idx+1) * 2000])

        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "SEED_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/SEED/SEED_EEG/Preprocessed_EEG"
    save_path = "/path/to/BrainSignalDataset/SEED"

    # prepare the train and val dataset
    sub_list = []
    for sub in os.listdir(root):
        if sub.endswith(".mat") and not sub.startswith('label') and not sub.startswith('.'):
            sub_path = os.path.join(root, sub)
            if os.path.isfile(sub_path):
                sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
