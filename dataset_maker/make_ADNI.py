import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import train_test_split


BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_2mm.nii.gz").get_fdata()


def preparing(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
    fname, label, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split), exist_ok=True)
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
        fMRI_roi = fMRI_roi[:, 5:]

        if fMRI_roi.shape[1] > 512:
            signals = [fMRI_roi[:, :ts_length], fMRI_roi[:, ts_length:2*ts_length], fMRI_roi[:, 2*ts_length:]]
        elif fMRI_roi.shape[1] > 80:
            signals = [fMRI_roi]
        else:
            signals = None

        if signals is not None:
            for idx, signal in enumerate(signals):
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
                if label == 'CN':  # healthy control
                    target = np.array([1, 0])
                elif label == 'AD':    # Alzheimer disease
                    target = np.array([0, 1])

                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "ADNI_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/ADNI"
    save_path = "/path/to/BrainSignalDataset/ADNI"
    phenotypic = pd.read_csv("/path/to/ADNI.csv")

    # pick the gender- and age-matched data
    phenotypic = phenotypic[phenotypic['Modality'] == 'fMRI']
    phenotypic_gender = (
        phenotypic[['Subject', 'Sex']]
        .drop_duplicates(subset=['Subject', 'Sex'])
        .dropna(subset=['Sex'])
        .groupby('Subject', as_index=False)
        .first()
    )
    phenotypic_age = (
        phenotypic[['Subject', 'Age']]
        .groupby('Subject', as_index=False)
        .agg({'Age': 'mean'})
    )
    phenotypic_group = (
        phenotypic[['Subject', 'Group']]
        .drop_duplicates(subset=['Subject', 'Group'])
        .groupby('Subject', as_index=False)
        .first()
    )
    phenotypic_all = (
        phenotypic_group
        .merge(phenotypic_gender, on='Subject', how='left')
        .merge(phenotypic_age, on='Subject', how='left')
    ).dropna(subset=['Age'])

    phenotypic_all['Age_bin'] = pd.qcut(phenotypic_all['Age'], q=5, duplicates='drop')
    AD = phenotypic_all[phenotypic_all['Group'] == 'AD']
    CN = phenotypic_all[phenotypic_all['Group'] == 'CN']

    # Desired counts per stratum from index_group
    target_counts = (
        AD.groupby(['Sex', 'Age_bin'])
        .size()
        .rename('Num')
        .reset_index()
    )

    selected_subject_ids = []
    used_subject_ids = set()
    for _, row in target_counts.iterrows():
        g, b, n = row['Sex'], row['Age_bin'], row['Num']
        stratum_CN = CN[(CN['Sex'] == g) & (CN['Age_bin'] == b) & (~CN['Subject'].isin(used_subject_ids))]
        if len(stratum_CN) == 0:
            continue  # or raise if strict matching required
        take = min(n, len(stratum_CN))
        chosen = stratum_CN.sample(n=take, replace=False)
        chosen_ids = chosen['Subject'].tolist()
        selected_subject_ids.extend(chosen_ids)
        used_subject_ids.update(chosen_ids)

    matched_recordings = phenotypic[(phenotypic['Group'] == 'CN') & (phenotypic['Subject'].isin(selected_subject_ids))].copy()
    final_phenotypic = pd.concat([matched_recordings, AD])

    # prepare the train dataset
    sub_list, sub_dirs, labels = [], [], []
    root = os.path.join(root, 'derivatives', 'fmriprep')
    for sub in os.listdir(root):
        if sub.startswith('sub') and not sub.endswith('.html'):
            for sess in os.listdir(os.path.join(root, sub)):
                if sess.startswith("ses"):
                    if os.path.exists(os.path.join(root, sub, sess, 'func')):
                        for file in os.listdir(os.path.join(root, sub, sess, 'func')):
                            if 'res-2_desc-preproc_bold.nii.gz' in file:
                                sub_path = os.path.join(root, sub, sess, 'func', file)
                                if os.path.isfile(sub_path):
                                    label = final_phenotypic.loc[final_phenotypic['Subject'] == sub.split("-")[-1].replace('S', '_S_'), 'Group']
                                    if not label.empty:
                                        sub_dirs.append(sub_path)
                                        labels.append(label.iloc[0])

    train_subs, tmp_subs, train_labels, tmp_labels = train_test_split(sub_dirs, labels, test_size=0.3, stratify=labels)
    val_subs, test_subs, val_labels, test_labels = train_test_split(tmp_subs, tmp_labels, test_size=0.5, stratify=tmp_labels)

    for train_sub, train_label in zip(train_subs, train_labels):
        sub_list.append((train_sub, train_label, 'train'))
    for val_sub, val_label in zip(val_subs, val_labels):
        sub_list.append((val_sub, val_label, 'val'))
    for test_sub, test_label in zip(test_subs, test_labels):
        sub_list.append((test_sub, test_label, 'test'))

    # prepare the fMRI and save them
    with Pool(processes=8) as pool:
        preparing_fn = partial(preparing, save_path=save_path)
        pool.map(preparing_fn, sub_list)
    # for sub in sub_list:
    #     preparing(sub, save_path)
