import os
import mne
import h5py
from typing import Union
from pathlib import Path


ch_names = [
    'Fp1', 'Fp2', 'F3', 'F4', 'FC3', 'FC4', 'C3', 'C4', 'CP3',
    'CP4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
    'P7', 'P8', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'Oz'
]


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_brainvision(fname, preload=preload)
        # pick 26 common channels across the TDBRAIN dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # discarding the first 10s and the last 10s time points in order to obtain the stationary signals
        eeg_raw.crop(tmin=10, tmax=eeg_raw.n_times / eeg_raw.info['sfreq'] - 10)
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
        with open(os.path.join(save_path, "tdbrain_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/TD-BRAIN-DATASET"
    save_path = "/path/to/BrainSignalDataset/TDBRAIN"

    # prepare the train and val dataset
    sub_list = []
    for sub in os.listdir(root):
        if sub.startswith("sub"):
            sess = os.listdir(os.path.join(root, sub))
            for ses in sess:
                EC_path = os.path.join(root, sub, ses, 'eeg', f'{sub}_{ses}_task-restEC_eeg.vhdr')
                EO_path = os.path.join(root, sub, ses, 'eeg', f'{sub}_{ses}_task-restEO_eeg.vhdr')
                if os.path.isfile(EC_path):
                    sub_list.append((EC_path, 'train'))
                    sub_list.append((EO_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
