import os
import mne
import h5py
from typing import Union
from pathlib import Path


ch_names = ['F1', 'F2', 'FC1', 'FC2', 'C1', 'C2', 'CP1', 'CP2', 'P1', 'P2',
            'Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'F4', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'P3', 'P4', 'PO3', 'PO4', 'O1', 'O2',
            'F5', 'F6', 'FC5',  'FC6', 'C5',  'C6', 'CP5',  'CP6', 'P5', 'P6',
            'AF7', 'AF8', 'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P8', 'PO7', 'PO8',
            'FT9', 'FT10', 'TP9', 'TP10',
            'AFz', 'Fz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_brainvision(fname, preload=preload)
        # pick 63 common channels across the PREDICT dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # discarding the first 30s and the last 30s time points in order to obtain the stationary signals
        eeg_raw.crop(tmin=30, tmax=eeg_raw.n_times / eeg_raw.info['sfreq'] - 30)
        # FIR bandpass and notch filter
        eeg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        eeg_raw.notch_filter(50.0, n_jobs=n_jobs)   # Europe power line
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
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "PREDICT_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/PREDICT"
    save_path = "/path/to/BrainSignalDataset/PREDICT"

    # prepare the train and val dataset
    sub_list = []
    for sub in os.listdir(root):
        if sub.startswith("sub"):
            sess = os.listdir(os.path.join(root, sub))
            for ses in sess:
                sub_path = os.path.join(root, sub, ses, 'eeg', f'{sub}_{ses}_task-rest_eeg.vhdr')
                if os.path.isfile(sub_path):
                    sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
