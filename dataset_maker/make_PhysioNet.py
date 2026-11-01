import os
import mne
import h5py
from typing import Union
from pathlib import Path


ch_names = ['F1..', 'F2..', 'Fc1.', 'Fc2.', 'C1..', 'C2..', 'Cp1.', 'Cp2.', 'P1..', 'P2..',
            'Fp1.', 'Fp2.', 'Af3.', 'Af4.', 'F3..', 'F4..', 'Fc3.', 'Fc4.', 'C3..', 'C4..', 'Cp3.', 'Cp4.', 'P3..', 'P4..', 'Po3.', 'Po4.', 'O1..', 'O2..',
            'F5..', 'F6..', 'Fc5.', 'Fc6.', 'C5..', 'C6..', 'Cp5.', 'Cp6.', 'P5..', 'P6..',
            'Af7.', 'Af8.', 'F7..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'Tp7.', 'Tp8.', 'P7..', 'P8..', 'Po7.', 'Po8.',
            'Fpz.', 'Afz.', 'Fz..', 'Fcz.', 'Cz..', 'Cpz.', 'Pz..', 'Poz.', 'Oz..']


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_edf(fname, preload=preload)
        # pick 61 common channels across the PhysioBank dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # FIR bandpass and notch filter
        if eeg_raw.info['sfreq'] < 160:
            eeg_raw.filter(l_freq=1, h_freq=50, n_jobs=n_jobs)
            tmax = 5.125
        else:
            eeg_raw.filter(l_freq=1, h_freq=75, n_jobs=n_jobs)
            eeg_raw.notch_filter(60.0, n_jobs=n_jobs)   # U.S. power line
            tmax = 4.1
        # pick the event
        eeg_raw.sample_event, eeg_raw.sample_event_id = mne.events_from_annotations(eeg_raw)
        eeg_raw.sample_event = mne.pick_events(eeg_raw.sample_event, include=[2, 3])
        del eeg_raw.sample_event_id['T0']
        # segment the whole signals into sample
        eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=tmax, preload=True, reject=None, proj=False)
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
        with open(os.path.join(save_path, "PhysioNet_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/eegmmidb/1.0.0"
    save_path = "/path/to/BrainSignalDataset/PhysioNet"

    # prepare the train and val dataset
    sub_list = []
    for sub in os.listdir(root):
        if sub.startswith("S") and not sub.endswith(".txt"):
            files = os.listdir(os.path.join(root, sub))
            for file in files:
                if file.endswith(".edf") and file.startswith("S") and "R01" not in file and "R02" not in file:
                    sub_path = os.path.join(root, sub, file)
                    sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
