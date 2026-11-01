import os
import h5py
import numpy as np
import nibabel as nib
from typing import Union
from pathlib import Path

BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_MNI152NLin2009cAsym.nii.gz").get_fdata()


def preparing(subject: tuple, save_path: Union[str, Path]):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI data
        fMRI_voxel = nib.load(fname).get_fdata()
        # parcel voxel signal to brain region
        fMRI_roi = np.zeros((int(BN_atlas.max()), fMRI_voxel.shape[-1]))
        roi = np.unique(BN_atlas)
        # exclude roi 0
        roi = roi[roi != 0]
        for i, value in enumerate(roi):
            fMRI_roi[i] = fMRI_voxel[BN_atlas == value].mean(axis=0)

        signals = [fMRI_roi]

        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "AOMIC_PIOP2_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/AOMIC_PIOP2"
    save_path = "/path/to/BrainSignalDataset/AOMIC_PIOP2"

    # prepare the train dataset
    sub_list = []
    root = os.path.join(root, 'derivatives', 'fmriprep')
    for sub in os.listdir(root):
        if sub.startswith('sub') and not sub.endswith('.html'):
            for file in os.listdir(os.path.join(root, sub, 'func')):
                if 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' in file:
                    sub_path = os.path.join(root, sub, 'func', file)
                    if os.path.isfile(sub_path):
                        sub_list.append((sub_path, 'train'))

    # prepare the fMRI and save them
    for sub in sub_list:
        preparing(sub, save_path)
