import os
import h5py
import numpy as np
import nibabel as nib
from typing import Union
from pathlib import Path


def preparing(subject: tuple, save_path: Union[str, Path]):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI beta series data
        fMRI_vertex = nib.load(fname)
        signals = fMRI_vertex.get_fdata()

        for idx, signal in enumerate(signals):
            signal = np.expand_dims(signal, axis=0)
            save_fname = fname.split(os.sep)[-3] + fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = np.eye(180, dtype=int)[idx]

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "HAD_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/HAD"
    save_path = "/path/to/BrainSignalDataset/HAD"

    # dataset splits
    train_subs = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07",
                  "sub-08", "sub-09", "sub-10", "sub-11", "sub-12", "sub-13", "sub-14",
                  "sub-15", "sub-16", "sub-17", "sub-18", "sub-19", "sub-20", "sub-21",
                  "sub-22", "sub-23", "sub-24"]
    val_subs = ["sub-25", "sub-26", "sub-27"]
    test_subs = ["sub-28", "sub-29", "sub-30"]

    # prepare the train dataset
    sub_list = []
    root = os.path.join(root, 'derivatives', 'ciftify')
    for sub in os.listdir(root):
        if sub.startswith('sub') and not sub.endswith('.html'):
            for file in os.listdir(os.path.join(root, sub, 'results')):
                if "dscalar" in file:
                    sub_path = os.path.join(root, sub, 'results', file)
                    if os.path.isfile(sub_path):
                        if sub in train_subs:
                            split = 'train'
                        elif sub in val_subs:
                            split = 'val'
                        elif sub in test_subs:
                            split = 'test'
                        else:
                            raise ValueError(f"{sub} is not in the splitting!")
                        sub_list.append((sub_path, split))

    # prepare the fMRI and save them
    for sub in sub_list:
        preparing(sub, save_path)
