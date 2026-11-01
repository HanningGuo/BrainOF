import os
import h5py
import numpy as np
import nibabel as nib
from typing import Union
from pathlib import Path

# 3T fMRI
BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_2mm.nii.gz").get_fdata()
# 7T fMRI
# BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_1.6mm.nii.gz").get_fdata()


def preparing(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
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

        # 3T rfMRI
        signals = [fMRI_roi[:,:ts_length], fMRI_roi[:,ts_length:2*ts_length], fMRI_roi[:,2*ts_length:]]
        # 7T rfMRI (overlapping window)
        # signals = [fMRI_roi[:,:ts_length], fMRI_roi[:,(ts_length-30):(-ts_length+30)], fMRI_roi[:,-ts_length:]]
        # 3T tfMRI and partly 7T tfMRI
        # signals = [fMRI_roi]
        # 7T Movie Watching
        # signals = [fMRI_roi[:,:512], fMRI_roi[:,512:]]    # Movie 1
        # signals = [fMRI_roi[:,18:530], fMRI_roi[:,530:]]    # Movie 2
        # signals = [fMRI_roi[:,:425], fMRI_roi[:,405:]]    # Movie 3
        # signals = [fMRI_roi[:,:512], fMRI_roi[:,512:]]    # Movie 4

        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[7] + '_' + fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''

            if signal.shape[1] > 512:
                # continue
                raise ValueError(' the length must be lower than 512')

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "HCP_YA_fMRI_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/HCP_YA/"
    save_path = "/path/to/BrainSignalDataset/HCP_YA_fMRI_Restin"
    # save_path = "/path/to/BrainSignalDataset/HCP_YA_fMRI_Task"

    # 3T rfMRI
    prefix = 'rfMRI'
    tasks = ['REST1', 'REST2']
    directions = ['LR', 'RL']
    # 7T rfMRI
    # tasks = ['REST1_7T', 'REST2_7T', 'REST3_7T', 'REST4_7T']
    # directions = ['PA', 'AP']

    # 3T tfMRI
    # prefix = 'tfMRI'
    # tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    # directions = ['LR', 'RL']
    # 7T tfMRI
    # tasks = ['MOVIE1_7T']   # ['MOVIE1_7T', 'MOVIE2_7T', 'MOVIE3_7T', 'MOVIE4_7T']
    # tasks = ['RETBAR1_7T', 'RETBAR2_7T', 'RETCCW_7T', 'RETCON_7T', 'RETCW_7T', 'RETEXP_7T']
    # directions = ['PA', 'AP']

    # prepare the train dataset
    sub_list = []
    for sub in os.listdir(root):
        if '.' not in sub:
            nii_path = os.path.join(root, sub, 'MNINonLinear', 'Results')
            for task in tasks:
                for direction in directions:
                    sub_path = os.path.join(nii_path, f'{prefix}_{task}_{direction}', f'{prefix}_{task}_{direction}.nii.gz')
                    if os.path.isfile(sub_path):
                        sub_list.append((sub_path, 'train'))

    # prepare the fMRI and save them
    for sub in sub_list:
        # 3T rfMRI
        preparing(sub, save_path, 400)
        # tfMRI and 7T rfMRI
        # preparing(sub, save_path)
