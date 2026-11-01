import os
import mne
import h5py
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union
from pathlib import Path
from sklearn.model_selection import train_test_split


ch_names = ['MEG0113', 'MEG0112', 'MEG0111', 'MEG0122', 'MEG0123', 'MEG0121', 'MEG0132',
            'MEG0133', 'MEG0131', 'MEG0143', 'MEG0142', 'MEG0141', 'MEG0213', 'MEG0212',
            'MEG0211', 'MEG0222', 'MEG0223', 'MEG0221', 'MEG0232', 'MEG0233', 'MEG0231',
            'MEG0243', 'MEG0242', 'MEG0241', 'MEG0313', 'MEG0312', 'MEG0311', 'MEG0322',
            'MEG0323', 'MEG0321', 'MEG0333', 'MEG0332', 'MEG0331', 'MEG0343', 'MEG0342',
            'MEG0341', 'MEG0413', 'MEG0412', 'MEG0411', 'MEG0422', 'MEG0423', 'MEG0421',
            'MEG0432', 'MEG0433', 'MEG0431', 'MEG0443', 'MEG0442', 'MEG0441', 'MEG0513',
            'MEG0512', 'MEG0511', 'MEG0523', 'MEG0522', 'MEG0521', 'MEG0532', 'MEG0533',
            'MEG0531', 'MEG0542', 'MEG0543', 'MEG0541', 'MEG0613', 'MEG0612', 'MEG0611',
            'MEG0622', 'MEG0623', 'MEG0621', 'MEG0633', 'MEG0632', 'MEG0631', 'MEG0642',
            'MEG0643', 'MEG0641', 'MEG0713', 'MEG0712', 'MEG0711', 'MEG0723', 'MEG0722',
            'MEG0721', 'MEG0733', 'MEG0732', 'MEG0731', 'MEG0743', 'MEG0742', 'MEG0741',
            'MEG0813', 'MEG0812', 'MEG0811', 'MEG0822', 'MEG0823', 'MEG0821', 'MEG0913',
            'MEG0912', 'MEG0911', 'MEG0923', 'MEG0922', 'MEG0921', 'MEG0932', 'MEG0933',
            'MEG0931', 'MEG0942', 'MEG0943', 'MEG0941', 'MEG1013', 'MEG1012', 'MEG1011',
            'MEG1023', 'MEG1022', 'MEG1021', 'MEG1032', 'MEG1033', 'MEG1031', 'MEG1043',
            'MEG1042', 'MEG1041', 'MEG1112', 'MEG1113', 'MEG1111', 'MEG1123', 'MEG1122',
            'MEG1121', 'MEG1133', 'MEG1132', 'MEG1131', 'MEG1142', 'MEG1143', 'MEG1141',
            'MEG1213', 'MEG1212', 'MEG1211', 'MEG1223', 'MEG1222', 'MEG1221', 'MEG1232',
            'MEG1233', 'MEG1231', 'MEG1243', 'MEG1242', 'MEG1241', 'MEG1312', 'MEG1313',
            'MEG1311', 'MEG1323', 'MEG1322', 'MEG1321', 'MEG1333', 'MEG1332', 'MEG1331',
            'MEG1342', 'MEG1343', 'MEG1341', 'MEG1412', 'MEG1413', 'MEG1411', 'MEG1423',
            'MEG1422', 'MEG1421', 'MEG1433', 'MEG1432', 'MEG1431', 'MEG1442', 'MEG1443',
            'MEG1441', 'MEG1512', 'MEG1513', 'MEG1511', 'MEG1522', 'MEG1523', 'MEG1521',
            'MEG1533', 'MEG1532', 'MEG1531', 'MEG1543', 'MEG1542', 'MEG1541', 'MEG1613',
            'MEG1612', 'MEG1611', 'MEG1622', 'MEG1623', 'MEG1621', 'MEG1632', 'MEG1633',
            'MEG1631', 'MEG1643', 'MEG1642', 'MEG1641', 'MEG1713', 'MEG1712', 'MEG1711',
            'MEG1722', 'MEG1723', 'MEG1721', 'MEG1732', 'MEG1733', 'MEG1731', 'MEG1743',
            'MEG1742', 'MEG1741', 'MEG1813', 'MEG1812', 'MEG1811', 'MEG1822', 'MEG1823',
            'MEG1821', 'MEG1832', 'MEG1833', 'MEG1831', 'MEG1843', 'MEG1842', 'MEG1841',
            'MEG1912', 'MEG1913', 'MEG1911', 'MEG1923', 'MEG1922', 'MEG1921', 'MEG1932',
            'MEG1933', 'MEG1931', 'MEG1943', 'MEG1942', 'MEG1941', 'MEG2013', 'MEG2012',
            'MEG2011', 'MEG2023', 'MEG2022', 'MEG2021', 'MEG2032', 'MEG2033', 'MEG2031',
            'MEG2042', 'MEG2043', 'MEG2041', 'MEG2113', 'MEG2112', 'MEG2111', 'MEG2122',
            'MEG2123', 'MEG2121', 'MEG2133', 'MEG2132', 'MEG2131', 'MEG2143', 'MEG2142',
            'MEG2141', 'MEG2212', 'MEG2213', 'MEG2211', 'MEG2223', 'MEG2222', 'MEG2221',
            'MEG2233', 'MEG2232', 'MEG2231', 'MEG2242', 'MEG2243', 'MEG2241', 'MEG2312',
            'MEG2313', 'MEG2311', 'MEG2323', 'MEG2322', 'MEG2321', 'MEG2332', 'MEG2333',
            'MEG2331', 'MEG2343', 'MEG2342', 'MEG2341', 'MEG2412', 'MEG2413', 'MEG2411',
            'MEG2423', 'MEG2422', 'MEG2421', 'MEG2433', 'MEG2432', 'MEG2431', 'MEG2442',
            'MEG2443', 'MEG2441', 'MEG2512', 'MEG2513', 'MEG2511', 'MEG2522', 'MEG2523',
            'MEG2521', 'MEG2533', 'MEG2532', 'MEG2531', 'MEG2543', 'MEG2542', 'MEG2541',
            'MEG2612', 'MEG2613', 'MEG2611', 'MEG2623', 'MEG2622', 'MEG2621', 'MEG2633',
            'MEG2632', 'MEG2631', 'MEG2642', 'MEG2643', 'MEG2641']


BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_3mm.nii.gz").get_fdata()


def preprocessing_rest(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, label, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw meg data
        meg_raw = mne.io.read_raw_fif(fname, preload=preload)
        # pick common channels between whole channel and current subject, and reorder those channel with specific order
        meg_raw.reorder_channels(ch_names)
        # FIR bandpass and notch filter for rest state
        meg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        # resample brain signal to 190 HZ
        meg_raw.resample(190, n_jobs=n_jobs)
        # discarding the first 10s and the last 10s time points in order to obtain the stationary signals
        meg_raw.crop(tmin=10, tmax=meg_raw.n_times / meg_raw.info['sfreq'] - 10)
        # set the 2-second length event
        meg_raw.sample_event = mne.make_fixed_length_events(meg_raw, id=10, duration=2, overlap=0)
        meg_raw.sample_event_id = {'2 seconds': 10}
        # segment the whole signals into 2 seconds sample
        meg_epochs = mne.Epochs(meg_raw, meg_raw.sample_event, meg_raw.sample_event_id, baseline=None, tmin=0, tmax=2, preload=True, reject=None, proj=False)

        signals = meg_epochs.get_data(units=dict(grad='fT/cm', mag='fT'))
        for idx, signal in enumerate(signals):
            save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            age, gender = label
            # Gender: 1=female 2=male
            if gender == 'FEMALE':
                gender = np.array([1, 0])
            elif gender == 'MALE':
                gender = np.array([0, 1])
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=np.array([(age, gender)], dtype=[('age', 'float'), ('gender', 'float', (2,))]))
    except:
        with open(os.path.join(save_path, "CamCAN_rMEG_error.log"), "a") as f:
            f.write(str(fname) + "\n")


def preprocessing_task(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw meg data
        meg_raw = mne.io.read_raw_fif(fname, preload=preload)
        # pick the events
        # pos = Path(fname).parts.index('BIDSsep')
        # sub_id = next((part for part in Path(fname).parts if part.startswith("sub-")), None)
        # event_path = os.path.join(str(Path(*Path(fname).parts[:pos + 1])), 'passive', sub_id, 'ses-passive', 'meg', f'{sub_id}_ses-passive_task-passive_events.tsv')
        # events = pd.read_csv(event_path, sep='\t')[['sample', 'value']].to_numpy()
        # events = np.insert(events, 1, values=0, axis=1)
        events = mne.find_events(meg_raw)
        # pick common channels between whole channel and current subject, and reorder those channel with specific order
        meg_raw.reorder_channels(ch_names)
        # FIR bandpass and notch filter for rest state
        meg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        # resample brain signal to 200 HZ (passive)
        meg_raw, events = meg_raw.resample(200, events=events, n_jobs=n_jobs)
        # resample brain signal to 150 HZ (smt)
        # meg_raw, events = meg_raw.resample(150, events=events, n_jobs=n_jobs)
        # segment the signals into 2.5 seconds sample for smt
        # events = mne.pick_events(events, include=[1, 2, 3, 4, 5])
        # meg_epochs = mne.Epochs(meg_raw, events, baseline=None, tmin=-1.25, tmax=1.25, preload=True, reject=None, proj=False)

        # passive
        signals = meg_raw.get_data(units=dict(grad='fT/cm', mag='fT'))
        first_samp = meg_raw.first_samp
        # smt
        # signals = meg_epochs.get_data(units=dict(grad='fT/cm', mag='fT'))
        for idx in range(len(events) - 1):  # passive
        # for idx, signal in enumerate(signals): # smt
            save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            # passive
            signal = signals[:, events[idx, 0]-first_samp:events[idx+1, 0]-first_samp]
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "CamCAN_tMEG_error.log"), "a") as f:
            f.write(str(fname) + "\n")


def preparing_rest(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
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

        signals = [fMRI_roi]
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            age, gender = label
            # Gender: 1=female 2=male
            if gender == 'FEMALE':
                gender = np.array([1, 0])
            elif gender == 'MALE':
                gender = np.array([0, 1])
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=np.array([(age, gender)], dtype=[('age', 'float'), ('gender', 'float', (2,))]))
    except:
        with open(os.path.join(save_path, "CamCAN_rfMRI_error.log"), "a") as f:
            f.write(fname + "\n")


def preparing_task(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
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
        with open(os.path.join(save_path, "CamCAN_tfMRI_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root_meg = "/path/to/CamCan/cc700/meg/pipeline/release005/BIDSsep"
    root_fMRI = "/path/to/CamCan/cc700/mri/pipeline/release004/data_fMRI_Smooth/aamod_smooth_00001"
    save_path_rmeg = "/path/to/BrainSignalDataset/CamCAN_MEG_Rest"
    save_path_tmeg = "/path/to/BrainSignalDataset/CamCAN_MEG_Task"
    save_path_rfMRI = "/path/to/BrainSignalDataset/CamCAN_fMRI_Rest"
    save_path_tfMRI = "/path/to/BrainSignalDataset/CamCAN_fMRI_Task"
    demographics = pd.read_csv("/path/to/CamCan/dataman/useraccess/processed/.../standard_data.csv")

    subs_meg_only = ['CC620527', 'CC410286', 'CC420244', 'CC320361', 'CC122016', 'CC721449', 'CC320861',
                     'CC221737', 'CC320445', 'CC720723', 'CC520436', 'CC220511', 'CC120212', 'CC320428',
                     'CC420364', 'CC120137', 'CC120184', 'CC312149', 'CC210174', 'CC720330', 'CC720774',
                     'CC710223']
    subs_fMRI_only = ['CC220806', 'CC221733', 'CC620044', 'CC710518', 'CC310407', 'CC420259', 'CC510568',
                      'CC710099', 'CC321087', 'CC510259', 'CC210314', 'CC120286', 'CC220901', 'CC710416',
                      'CC320269', 'CC620560', 'CC120987', 'CC120816', 'CC221580', 'CC510015', 'CC610614',
                      'CC120234', 'CC220394', 'CC221038', 'CC310402', 'CC722216', 'CC120123', 'CC720180',
                      'CC121194', 'CC210304', 'CC721618']

    # prepare the train, val and test dataset for rest meg
    sub_list, subjects_all, labels_all = [], [], []
    for sub in os.listdir(os.path.join(root_meg, "derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002")):
        if os.path.isfile(os.path.join(root_meg, "derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002", sub, f'mf2pt2_{sub}_ses-rest_task-rest_meg.fif')):
            sub = sub.replace('sub-', '')
            subjects_all.append(sub)
            age = demographics[demographics.iloc[:, 0] == sub].iloc[0, 1]
            gender = demographics[demographics.iloc[:, 0] == sub].iloc[0, 2]
            labels_all.append((age, gender))

    subjects_meg, labels_meg, subjects_common, labels_common = [], [], [], []
    for sub, label in zip(subjects_all, labels_all):
        if sub in subs_meg_only:
            subjects_meg.append(sub)
            labels_meg.append(label)
        else:
            subjects_common.append(sub)
            labels_common.append(label)

    _, genders = zip(*labels_common)
    train_subs, tmp_subs, train_labels, tmp_labels = train_test_split(subjects_common, labels_common, test_size=0.3, stratify=genders)
    _, tmp_genders = zip(*tmp_labels)
    val_subs, test_subs, val_labels, test_labels = train_test_split(tmp_subs, tmp_labels, test_size=0.5, stratify=tmp_genders)

    for sub_meg, label_meg in zip(subjects_meg, labels_meg):
        sub_meg = 'sub-' + sub_meg
        meg = os.path.join(root_meg, "derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002", sub_meg, f'mf2pt2_{sub_meg}_ses-rest_task-rest_meg.fif')
        sub_list.append((meg, label_meg, 'train'))
    for train_sub, train_label in zip(train_subs, train_labels):
        train_sub = 'sub-' + train_sub
        train_meg = os.path.join(root_meg, "derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002", train_sub, f'mf2pt2_{train_sub}_ses-rest_task-rest_meg.fif')
        sub_list.append((train_meg, train_label, 'train'))
    for val_sub, val_label in zip(val_subs, val_labels):
        val_sub = 'sub-' + val_sub
        val_meg = os.path.join(root_meg, "derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002", val_sub, f'mf2pt2_{val_sub}_ses-rest_task-rest_meg.fif')
        sub_list.append((val_meg, val_label, 'val'))
    for test_sub, test_label in zip(test_subs, test_labels):
        test_sub = 'sub-' + test_sub
        test_meg = os.path.join(root_meg, "derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002", test_sub, f'mf2pt2_{test_sub}_ses-rest_task-rest_meg.fif')
        sub_list.append((test_meg, test_label, 'test'))

    # preprocess the rest meg and save them
    for sub in sub_list:
        preprocessing_rest(sub, save_path_rmeg)

    # task meg
    sub_list_tmeg = []
    task = 'passive'
    # task = 'smt'
    for sub in os.listdir(os.path.join(root_meg, 'derivatives_' + task, 'aa/AA_movecomp/aamod_meg_maxfilt_00002')):
        sub_path = os.path.join(root_meg, 'derivatives_' + task, 'aa/AA_movecomp/aamod_meg_maxfilt_00002', sub, f'mf2pt2_{sub}_ses-{task}_task-{task}_meg.fif')
        if os.path.isfile(sub_path):
            sub_list_tmeg.append((sub_path, 'train'))

    # prepare the tmeg and save them
    for sub in sub_list_tmeg:
        preprocessing_task(sub, save_path_tmeg)

    # fMRI
    sub_list_fMRI, subjects_fMRI, labels_fMRI = [], [], []
    for sub in subs_fMRI_only:
        subjects_fMRI.append(sub)
        age = demographics[demographics.iloc[:, 0] == sub].iloc[0, 1]
        gender = demographics[demographics.iloc[:, 0] == sub].iloc[0, 2]
        labels_fMRI.append((age, gender))

    for sub_fMRI, label_fMRI in zip(subjects_fMRI, labels_fMRI):
        fMRI_file = glob.glob(os.path.join(root_fMRI, sub_fMRI, 'Rest', '*.nii'), recursive=True)[0]
        sub_list_fMRI.append((fMRI_file, label_fMRI, 'train'))
    for train_sub, train_label in zip(train_subs, train_labels):
        train_file = glob.glob(os.path.join(root_fMRI, train_sub, 'Rest', '*.nii'), recursive=True)[0]
        sub_list_fMRI.append((train_file, train_label, 'train'))
    for val_sub, val_label in zip(val_subs, val_labels):
        val_file = glob.glob(os.path.join(root_fMRI, val_sub, 'Rest', '*.nii'), recursive=True)[0]
        sub_list_fMRI.append((val_file, val_label, 'val'))
    for test_sub, test_label in zip(test_subs, test_labels):
        test_file = glob.glob(os.path.join(root_fMRI, test_sub, 'Rest', '*.nii'), recursive=True)[0]
        sub_list_fMRI.append((test_file, test_label, 'test'))

    # prepare the rfMRI and save them
    for sub in sub_list_fMRI:
        preparing_rest(sub, save_path_rfMRI)

    # tfMRI
    sub_list_tfMRI = []
    tasks = ['Movie', 'SMT']
    for sub in os.listdir(root_fMRI):
        for task in tasks:
            if os.path.exists(os.path.join(root_fMRI, sub, task)):
                sub_path = glob.glob(os.path.join(root_fMRI, sub, task, '*.nii'), recursive=True)[0]
                if os.path.isfile(sub_path):
                    sub_list_tfMRI.append((sub_path, 'train'))

    # prepare the tfMRI and save them
    for sub in sub_list_tfMRI:
        preparing_task(sub, save_path_tfMRI)
