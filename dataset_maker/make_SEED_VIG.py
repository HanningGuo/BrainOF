import os
import mne
import h5py
import pymatreader
from typing import Union
from pathlib import Path


ch_names = ['CP1', 'CP2', 'P1', 'P2',
            'PO3', 'PO4', 'O1', 'O2',
            'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
            'PZ', 'POZ', 'OZ']


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_mat = pymatreader.read_mat(fname)['EEG']
        # initialize the information about EEG device and create info object
        info = mne.create_info(eeg_mat['chn'], ch_types=['eeg'] * len(eeg_mat['chn']), sfreq=eeg_mat['sample_rate'])
        eeg_raw = mne.io.RawArray(eeg_mat['data'].T * 1e-6, info)  # convert the unit V to unit uV
        # reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # FIR bandpass and notch filter
        eeg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        # eeg_raw.notch_filter(50.0, n_jobs=n_jobs)   # China power line
        # resample brain signal to 200 HZ
        # eeg_raw.resample(200, n_jobs=n_jobs)
        # set the 8-second length event
        eeg_raw.sample_event = mne.make_fixed_length_events(eeg_raw, id=8, duration=8, overlap=0)
        eeg_raw.sample_event_id = {'8 seconds': 8}
        # segment the whole signals into 8 seconds sample
        eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=8, preload=True, reject=None, proj=False)
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
        with open(os.path.join(save_path, "seed_vig_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/SEED-VIG/Raw_Data"
    save_path = "/path/to/BrainSignalDataset/SEED_VIG"

    # prepare the train dataset
    sub_list = []
    for sub in os.listdir(root):
        if sub.endswith("mat") and not sub.startswith("."):
            sub_path = os.path.join(root, sub)
            if os.path.isfile(sub_path):
                sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
