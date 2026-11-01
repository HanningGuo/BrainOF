import os
import mne
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union
from pathlib import Path
from sklearn.model_selection import train_test_split


ch_names = ['F1', 'F2', 'FC1', 'FC2', 'C1', 'C2', 'CP1', 'CP2', 'P1', 'P2',
            'Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'F4', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'P3', 'P4', 'PO3', 'PO4', 'O1', 'O2',
            'F5', 'F6', 'FC5',  'FC6', 'C5',  'C6', 'CP5',  'CP6', 'P5',  'P6', 'PO7', 'PO8',
            'AF7', 'AF8', 'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P8', 'PO9', 'PO10',
            'AFz', 'Fz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']

BN_atlas = nib.load("/Brainnetome Atlas/BN_Atlas_246_2mm.nii.gz").get_fdata()


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, label, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_eeglab(fname, preload=preload)
        # pick common channels between whole channel and current subject, and reorder those channel with specific order
        eeg_raw.reorder_channels(list(filter(lambda x: x not in (set(ch_names) - set(eeg_raw.ch_names)), ch_names)))

        # find the boundary and the start of brain signal
        boundary = [index for index, value in enumerate(eeg_raw.annotations.description) if value in ['boundary', '32766']]
        boundary.append(eeg_raw.annotations.description.size - 1)
        sample_event = []
        for i in range(len(boundary) - 1):
            if eeg_raw.annotations.onset[boundary[i]+1] + 10 <= eeg_raw.annotations.onset[boundary[i+1]]:
                sample_event.append(mne.make_fixed_length_events(eeg_raw, start=eeg_raw.annotations.onset[boundary[i]+1], stop=eeg_raw.annotations.onset[boundary[i+1]], id=10, duration=10, overlap=0))
        # set the 10-second length event
        eeg_raw.sample_event = np.concatenate(sample_event)
        eeg_raw.sample_event_id = {'10 seconds': 10}
        # segment the whole signals into 10 seconds sample
        eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=10,
                                preload=True, reject=None, proj=False)

        # resample brain signal to 200 HZ
        eeg_epochs.resample(200, n_jobs=n_jobs)
        # set average reference
        eeg_epochs.set_eeg_reference(ref_channels='average', ch_type='eeg')

        signals = eeg_epochs.get_data(units='uV')
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            age, gender = label
            if age in ['20-25', '25-30', '30-35', '35-40']:
                target_age = np.array([1, 0])
            elif age in ['55-60', '60-65', '65-70', '70-75', '75-80']:
                target_age = np.array([0, 1])

            # Gender: 1=female 2=male
            if gender == 1:
                target_gender = np.array([1, 0])
            elif gender == 2:
                target_gender = np.array([0, 1])

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=(target_age, target_gender))
    except:
        with open(os.path.join(save_path, "lemon_error.log"), "a") as f:
            f.write(fname + "\n")


def preparing(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
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

        signals = np.stack((fMRI_roi[:, :ts_length], fMRI_roi[:, ts_length:2*ts_length]))
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            age, gender = label
            if age in ['20-25', '25-30', '30-35', '35-40']:
                target_age = np.array([1, 0])
            elif age in ['55-60', '60-65', '65-70', '70-75', '75-80']:
                target_age = np.array([0, 1])

            # Gender: 1=female 2=male
            if gender == 1:
                target_gender = np.array([1, 0])
            elif gender == 2:
                target_gender = np.array([0, 1])

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=(target_age, target_gender))
    except:
        with open(os.path.join(save_path, "lemon_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root_eeg = "/path/to/LEMON/EEG"
    root_fMRI = "/path/to/LEMON/fMRI"
    save_path_eeg = "/path/to/BrainSignalDataset/LEMON_EEG"
    save_path_fMRI = "/path/to/BrainSignalDataset/LEMON_fMRI"
    demographics = pd.read_csv("/path/to/LEMON/Participants_LEMON.csv")

    subs_eeg_only = ['sub-032339', 'sub-032341', 'sub-032435', 'sub-032438', 'sub-032456', 'sub-032457',
                     'sub-032459', 'sub-032466', 'sub-032509']
    subs_fMRI_only = ['sub-032309', 'sub-032335', 'sub-032365', 'sub-032366', 'sub-032374', 'sub-032382',
                      'sub-032384', 'sub-032404', 'sub-032410', 'sub-032419', 'sub-032426', 'sub-032433',
                      'sub-032443', 'sub-032445', 'sub-032461', 'sub-032485', 'sub-032486', 'sub-032487',
                      'sub-032488', 'sub-032489', 'sub-032492', 'sub-032500', 'sub-032519', 'sub-032520',
                      'sub-032527']

    # prepare the train, val and test dataset
    sub_list, subjects_all, labels_all = [], [], []
    for sub in os.listdir(root_eeg):
        if os.path.isfile(os.path.join(root_eeg, sub, f'{sub}_EC.set')):
            subjects_all.append(sub)
            gender = demographics[demographics.iloc[:, 0] == sub].iloc[0, 1]
            age = demographics[demographics.iloc[:, 0] == sub].iloc[0, 2]
            labels_all.append((age,gender))

    subjects_eeg, labels_eeg, subjects_common, labels_common = [], [], [], []
    for sub, label in zip(subjects_all, labels_all):
        if sub in subs_eeg_only:
            subjects_eeg.append(sub)
            labels_eeg.append(label)
        else:
            subjects_common.append(sub)
            labels_common.append(label)

    _, genders = zip(*labels_common)
    train_subs, tmp_subs, train_labels, tmp_labels = train_test_split(subjects_common, labels_common, test_size=0.3, stratify=genders)
    _, tmp_genders = zip(*tmp_labels)
    val_subs, test_subs, val_labels, test_labels = train_test_split(tmp_subs, tmp_labels, test_size=0.5, stratify=tmp_genders)

    for sub_eeg, label_eeg in zip(subjects_eeg, labels_eeg):
        eeg_EC = os.path.join(root_eeg, sub_eeg, f'{sub_eeg}_EC.set')
        eeg_EO = os.path.join(root_eeg, sub_eeg, f'{sub_eeg}_EO.set')
        sub_list.append((eeg_EC, label_eeg, 'train'))
        sub_list.append((eeg_EO, label_eeg, 'train'))
    for train_sub, train_label in zip(train_subs, train_labels):
        train_EC = os.path.join(root_eeg, train_sub, f'{train_sub}_EC.set')
        train_EO = os.path.join(root_eeg, train_sub, f'{train_sub}_EO.set')
        sub_list.append((train_EC, train_label, 'train'))
        sub_list.append((train_EO, train_label, 'train'))
    for val_sub, val_label in zip(val_subs, val_labels):
        val_EC = os.path.join(root_eeg, val_sub, f'{val_sub}_EC.set')
        val_EO = os.path.join(root_eeg, val_sub, f'{val_sub}_EO.set')
        sub_list.append((val_EC, val_label, 'val'))
        sub_list.append((val_EO, val_label, 'val'))
    for test_sub, test_label in zip(test_subs, test_labels):
        test_EC = os.path.join(root_eeg, test_sub, f'{test_sub}_EC.set')
        test_EO = os.path.join(root_eeg, test_sub, f'{test_sub}_EO.set')
        sub_list.append((test_EC, test_label, 'test'))
        sub_list.append((test_EO, test_label, 'test'))

    # preprocess the eeg and save them
    for sub in sub_list:
        preprocessing(sub, save_path_eeg)

    # fMRI
    sub_list_fMRI, subjects_fMRI, labels_fMRI = [], [], []
    for sub in subs_fMRI_only:
        subjects_fMRI.append(sub)
        gender = demographics[demographics.iloc[:, 0] == sub].iloc[0, 1]
        age = demographics[demographics.iloc[:, 0] == sub].iloc[0, 2]
        labels_fMRI.append((age, gender))

    for sub_fMRI, label_fMRI in zip(subjects_fMRI, labels_fMRI):
        fMRI_file = os.path.join(root_fMRI, sub_fMRI, 'func', f'{sub_fMRI}_ses-01_task-rest_acq-AP_run-01_MNI2mm.nii.gz')
        sub_list_fMRI.append((fMRI_file, label_fMRI, 'train'))
    for train_sub, train_label in zip(train_subs, train_labels):
        train_file = os.path.join(root_fMRI, train_sub, 'func', f'{train_sub}_ses-01_task-rest_acq-AP_run-01_MNI2mm.nii.gz')
        sub_list_fMRI.append((train_file, train_label, 'train'))
    for val_sub, val_label in zip(val_subs, val_labels):
        val_file = os.path.join(root_fMRI, val_sub, 'func', f'{val_sub}_ses-01_task-rest_acq-AP_run-01_MNI2mm.nii.gz')
        sub_list_fMRI.append((val_file, val_label, 'val'))
    for test_sub, test_label in zip(test_subs, test_labels):
        test_file = os.path.join(root_fMRI, test_sub, 'func', f'{test_sub}_ses-01_task-rest_acq-AP_run-01_MNI2mm.nii.gz')
        sub_list_fMRI.append((test_file, test_label, 'test'))

    # prepare the fMRI and save them
    for sub in sub_list_fMRI:
        preparing(sub, save_path_fMRI)
