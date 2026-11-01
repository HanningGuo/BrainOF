import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union
from pathlib import Path
from sklearn.model_selection import train_test_split

BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_Pediatric_res-2.nii.gz").get_fdata()


def preparing(subject: tuple, save_path: Union[str, Path]):
    fname, label, split = subject
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

        # discard the first five time points due to its unstable
        signals = [fMRI_roi[:, 5:]]

        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            if label == 'CN':  # healthy control
                target = np.array([1, 0])
            elif label == 'ADHD':    # ADHD
                target = np.array([0, 1])

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "ADHD200_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/ADHD200/RawDataBIDS"
    save_path = "/path/to/BrainSignalDataset/ADHD200"

    # prepare the train dataset
    sub_list, sub_dirs, labels = [], [], []
    for site in os.listdir(root):
        if site != 'Brown':
            root_site = os.path.join(root, site, 'derivatives', 'fmriprep')
            if os.path.exists(root_site):
                phenotypic = pd.read_csv(os.path.join(root, site, 'participants.tsv'), sep="\t", encoding="latin1")
                for sub in os.listdir(root_site):
                    if sub.startswith('sub') and not sub.endswith('.html'):
                        for sess in os.listdir(os.path.join(root_site, sub)):
                            if sess.startswith("ses"):
                                if os.path.exists(os.path.join(root_site, sub, sess, 'func')):
                                    for file in os.listdir(os.path.join(root_site, sub, sess, 'func')):
                                        if 'res-2_desc-preproc_bold.nii.gz' in file:
                                            sub_path = os.path.join(root_site, sub, sess, 'func', file)
                                            if os.path.isfile(sub_path):
                                                dx = phenotypic.loc[phenotypic['participant_id'] == int(sub.split("-")[-1]), 'dx'].iloc[0]
                                                if 'Typically Developing Children' in dx:
                                                    label = 'CN'
                                                elif 'ADHD' in dx:
                                                    label = 'ADHD'
                                                sub_dirs.append(sub_path)
                                                labels.append(label)

    train_subs, tmp_subs, train_labels, tmp_labels = train_test_split(sub_dirs, labels, test_size=0.3, stratify=labels)
    val_subs, test_subs, val_labels, test_labels = train_test_split(tmp_subs, tmp_labels, test_size=0.5, stratify=tmp_labels)

    for train_sub, train_label in zip(train_subs, train_labels):
        sub_list.append((train_sub, train_label, 'train'))
    for val_sub, val_label in zip(val_subs, val_labels):
        sub_list.append((val_sub, val_label, 'val'))
    for test_sub, test_label in zip(test_subs, test_labels):
        sub_list.append((test_sub, test_label, 'test'))

    # prepare the fMRI and save them
    for sub in sub_list:
        preparing(sub, save_path)
