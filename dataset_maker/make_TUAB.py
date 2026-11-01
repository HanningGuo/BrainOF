import os
import mne
import h5py
import numpy as np
from typing import Union
from pathlib import Path
from sklearn.model_selection import train_test_split

ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
            'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
ch_mapping = {
    'EEG FP1-REF': 'Fp1', 'EEG FP2-REF': 'Fp2', 'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4',
    'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4', 'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4',
    'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8',
    'EEG T3-REF': 'T3', 'EEG T4-REF': 'T4', 'EEG T5-REF': 'T5', 'EEG T6-REF': 'T6',
    'EEG FZ-REF': 'Fz', 'EEG CZ-REF': 'Cz', 'EEG PZ-REF': 'Pz'
}


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, label, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_edf(fname, preload=preload)
        # rename special names to standard names of the international 10/20 system
        eeg_raw.rename_channels(ch_mapping)
        # pick 19 common channels across the TUAB dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # discarding the first 10s and the last 10s time points in order to obtain the stationary signals
        eeg_raw.crop(tmin=10, tmax=eeg_raw.n_times / eeg_raw.info['sfreq'] - 10)
        # FIR bandpass and notch filter
        eeg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        eeg_raw.notch_filter(60.0, n_jobs=n_jobs)   # U.S. power line
        # resample brain signal to 200 HZ
        eeg_raw.resample(200, n_jobs=n_jobs)
        # set the 10-second length event
        eeg_raw.sample_event = mne.make_fixed_length_events(eeg_raw, id=10, duration=10, overlap=0)
        eeg_raw.sample_event_id = {'10 seconds': 10}
        # segment the whole signals into 10 seconds sample
        eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=10, preload=True, reject=None, proj=False)
        # set average reference
        eeg_epochs.set_eeg_reference(ref_channels='average', ch_type='eeg')

        signals = eeg_epochs.get_data(units='uV')
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            if label == 'abnormal':
                target = np.array([1, 0])
            elif label == 'normal':
                target = np.array([0, 1])
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "tuab_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/TUAB/edf"
    save_path = "/path/to/BrainSignalDataset/TUAB"

    # prepare the train and val dataset
    sub_list, sub_dirs, labels = [], [], []
    for sub in os.listdir(os.path.join(root, 'train', 'abnormal', '01_tcp_ar')):
        sub_path = os.path.join(root, 'train', 'abnormal', '01_tcp_ar', sub)
        if os.path.isfile(sub_path):
            sub_dirs.append(sub_path)
            labels.append('abnormal')
    for sub in os.listdir(os.path.join(root, 'train', 'normal', '01_tcp_ar')):
        sub_path = os.path.join(root, 'train', 'normal', '01_tcp_ar', sub)
        if os.path.isfile(sub_path):
            sub_dirs.append(sub_path)
            labels.append('normal')
    train_subs, val_subs, train_labels, val_labels = train_test_split(sub_dirs, labels, test_size=0.1, stratify=labels)
    for train_sub, train_label in zip(train_subs, train_labels):
        sub_list.append((train_sub, train_label, 'train'))
    for val_sub, val_label in zip(val_subs, val_labels):
        sub_list.append((val_sub, val_label, 'val'))

    # prepare the test dataset
    for sub in os.listdir(os.path.join(root, 'eval', 'abnormal', '01_tcp_ar')):
        sub_path = os.path.join(root, 'eval', 'abnormal', '01_tcp_ar', sub)
        if os.path.isfile(sub_path):
            sub_list.append((sub_path, 'abnormal', 'test'))
    for sub in os.listdir(os.path.join(root, 'eval', 'normal', '01_tcp_ar')):
        sub_path = os.path.join(root, 'eval', 'normal', '01_tcp_ar', sub)
        if os.path.isfile(sub_path):
            sub_list.append((sub_path, 'normal', 'test'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
