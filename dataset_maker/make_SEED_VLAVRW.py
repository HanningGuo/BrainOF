import os
import mne
import h5py
from typing import Union
from pathlib import Path

ch_names = ['EEG Fp1 - Pz', 'EEG Fp2 - Pz', 'EEG F3 - Pz', 'EEG F4 - Pz', 'EEG C3 - Pz', 'EEG C4 - Pz','EEG P3 - Pz', 'EEG P4 - Pz', 'EEG O1 - Pz', 'EEG O2 - Pz',
            'EEG F7 - Pz', 'EEG F8 - Pz', 'EEG T3 - Pz', 'EEG T4 - Pz', 'EEG T5 - Pz', 'EEG T6 - Pz',
            'EEG Fz - Pz', 'EEG Cz - Pz']


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_edf(fname, preload=preload)
        # pick 18 common channels across the VLA dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # discarding the first 60s and the last 60s time points in order to obtain the stationary signals
        eeg_raw.crop(tmin=60, tmax=eeg_raw.n_times / eeg_raw.info['sfreq'] - 60)
        # FIR bandpass and notch filter
        eeg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        # eeg_raw.notch_filter(50.0, n_jobs=n_jobs)   # already notch filter (China power line 50Hz)
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
            if 'real' in fname:
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'real_sample{idx:04d}.h5'
            elif 'lab' in fname:
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'lab_sample{idx:04d}.h5'
            target = ""

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "seed_vla_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/VLA_VRW"
    save_path = "/path/to/BrainSignalDataset/SEED_VLA_VRW"

    # prepare the SEED-VLA dataset
    sub_list = []
    for exper in os.listdir(root):
        if os.path.isdir(os.path.join(root, exper)):
            for sub in os.listdir(os.path.join(root, exper, 'EEG')):
                if sub.endswith("edf") and not sub.startswith("."):
                    sub_path = os.path.join(root, exper, 'EEG', sub)
                    if os.path.isfile(sub_path):
                        sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
