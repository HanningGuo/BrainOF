import os
import h5py
import numpy as np
import nibabel as nib
from typing import Union
from pathlib import Path

BN_atlas_normal = nib.load("./Brainnetome Atlas/BN_Atlas_246_2mm.nii.gz").get_fdata()
BN_atlas_pediatric = nib.load("./Brainnetome Atlas/BN_Atlas_246_Pediatric_res-2.nii.gz").get_fdata()


def preparing(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI data
        fMRI_voxel = nib.load(fname).get_fdata()
        # choose the brain atlas
        if 'Pediatric' in fname:
            BN_atlas = BN_atlas_pediatric
        else:
            BN_atlas = BN_atlas_normal
        # parcel voxel signal to brain region
        fMRI_roi = np.zeros((int(BN_atlas.max()), fMRI_voxel.shape[-1]))
        roi = np.unique(BN_atlas)
        # exclude roi 0
        roi = roi[roi != 0]
        for i, value in enumerate(roi):
            fMRI_roi[i] = fMRI_voxel[BN_atlas == value].mean(axis=0)

        if fMRI_roi.shape[1] > 512:
            signals = [fMRI_roi[:, :ts_length], fMRI_roi[:, ts_length:2*ts_length], fMRI_roi[:, 2*ts_length:]]
        elif fMRI_roi.shape[1] > 80:
            signals = [fMRI_roi]
        else:
            signals = None

        if signals is not None:
            for idx, signal in enumerate(signals):
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
                target = ''

                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "ABIDE2_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/ABIDE2/RawData"
    save_path = "/path/to/BrainSignalDataset/ABIDE2"

    # prepare the train dataset
    sub_list = []
    for site in os.listdir(root):
        root_site = os.path.join(root, site, 'derivatives', 'fmriprep')
        for sub in os.listdir(root_site):
            if sub.startswith('sub') and not sub.endswith('.html'):
                for sess in os.listdir(os.path.join(root_site, sub)):
                    if sess.startswith("ses"):
                        if os.path.exists(os.path.join(root_site, sub, sess, 'func')):
                            for file in os.listdir(os.path.join(root_site, sub, sess, 'func')):
                                if 'res-2_desc-preproc_bold.nii.gz' in file:
                                    sub_path = os.path.join(root_site, sub, sess, 'func', file)
                                    if os.path.isfile(sub_path):
                                        sub_list.append((sub_path, 'train'))

    # prepare the fMRI and save them
    for sub in sub_list:
        preparing(sub, save_path)
