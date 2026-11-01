import os
import mne
import h5py
import numpy as np
from typing import Union
from pathlib import Path


ch_names = ['F1', 'F2', 'FC1', 'FC2', 'C1', 'C2', 'CP1', 'CP2', 'P1', 'P2',
            'FP1', 'FP2', 'AF3', 'AF4', 'F3', 'F4', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'P3', 'P4', 'PO3', 'PO4', 'O1', 'O2',
            'F5', 'F6', 'FC5',  'FC6', 'C5',  'C6', 'CP5',  'CP6', 'P5', 'P6', 'PO5', 'PO6',
            'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P8', 'PO7', 'PO8', 'CB1', 'CB2',
            'FPZ', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ']


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_cnt(fname, preload=preload)
        # pick 62 common channels across the SEED dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # pick the events
        events = []
        for i in range(0, len(eeg_raw.annotations.onset), 2):
            events.append(mne.make_fixed_length_events(eeg_raw, start=eeg_raw.annotations.onset[i], stop=eeg_raw.annotations.onset[i+1], id=10, duration=10, overlap=0))
        events = np.concatenate(events)

        # FIR bandpass and notch filter
        eeg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        # eeg_raw.notch_filter(50.0, n_jobs=n_jobs)   # already notch filter (China power line 50Hz)
        # resample brain signal to 200 HZ
        eeg_raw, events = eeg_raw.resample(200, events=events, n_jobs=n_jobs)
        eeg_raw.sample_event = events
        eeg_raw.sample_event_id = {'10 seconds': 10}

        # segment the whole signals into 10 second sample
        eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=10, preload=True, reject=None, proj=False)
        # set average reference
        eeg_epochs.set_eeg_reference(ref_channels='average', ch_type='eeg')

        signals = eeg_epochs.get_data(units='uV')
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ""

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "SEED_VII_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/SEED-VII/EEG_raw"
    save_path = "/path/to/BrainSignalDataset/SEED_VII"

    # prepare the SEED-VII dataset
    sub_list = []
    for sub in os.listdir(root):
        if sub.endswith("cnt") and not sub.startswith(".") and '9_20221111_3' not in sub:
            sub_path = os.path.join(root, sub)
            if os.path.isfile(sub_path):
                sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
