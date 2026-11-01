import os
import h5py
import pymatreader
from typing import Union
from pathlib import Path


def preparing(subject: tuple, save_path: Union[str, Path]):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI data
        fMRI_roi = pymatreader.read_mat(fname)['ROISignals']
        # extract time courses from AAL
        fMRI_roi = fMRI_roi[:, 0:116].T

        signals = [fMRI_roi]

        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "REST_meta_MDD_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/REST_meta_MDD/Results/ROISignals_FunImgARCWF"
    save_path = "/path/to/BrainSignalDataset/REST_meta_MDD"

    # prepare the train, val and test dataset
    sub_list= []
    for sub in os.listdir(root):
        if sub.startswith('ROISignals') and sub.endswith('.mat') and 'S4' not in sub:
            sub_path = os.path.join(root, sub)
            sub_list.append((sub_path, 'train'))

    # prepare the fMRI and save them
    for sub in sub_list:
        preparing(sub, save_path)
