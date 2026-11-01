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

labels = [
    [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]
]
label_names = ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy']

start_timestamp = [
    [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
    [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
    [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
]

end_timestamp = [
    [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
    [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
    [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
]


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname = subject
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_cnt(fname, preload=preload)
        # pick 62 common channels across the SEED dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # FIR bandpass and notch filter
        eeg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        # eeg_raw.notch_filter(50.0, n_jobs=n_jobs)   # already notch filter (China power line 50Hz)
        # resample brain signal to 200 HZ
        eeg_raw.resample(200, n_jobs=n_jobs)

        for trail in range(15):
            if trail < 5:
                split = 'train'
            elif trail < 10:
                split = 'val'
            else:
                split = 'test'
            sess = int(fname.split('_')[-2]) - 1
            # set the 1-second length event
            eeg_raw.sample_event = mne.make_fixed_length_events(eeg_raw, start=start_timestamp[sess][trail], stop=end_timestamp[sess][trail], id=labels[sess][trail], duration=1, overlap=0)
            eeg_raw.sample_event_id = {label_names[labels[sess][trail]]: labels[sess][trail]}
            # segment the whole signals into 1 second sample
            eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=1, preload=True, reject=None, proj=False)
            # set average reference
            eeg_epochs.set_eeg_reference(ref_channels='average', ch_type='eeg')

            signals = eeg_epochs.get_data(units='uV')
            for idx, signal in enumerate(signals):
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_{label_names[labels[sess][trail]]}{trail}_sample{idx:04d}.h5'
                if labels[sess][trail] == 0:
                    target = np.array([1, 0, 0, 0, 0])
                elif labels[sess][trail] == 1:
                    target = np.array([0, 1, 0, 0, 0])
                elif labels[sess][trail] == 2:
                    target = np.array([0, 0, 1, 0, 0])
                elif labels[sess][trail] == 3:
                    target = np.array([0, 0, 0, 1, 0])
                elif labels[sess][trail] == 4:
                    target = np.array([0, 0, 0, 0, 1])

                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "SEEDV_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/SEED-V/SEED-V/EEG_raw"
    save_path = "/path/to/BrainSignalDataset/SEEDV"

    # make directory
    for split in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(save_path, split)):
            os.makedirs(os.path.join(save_path, split))

    # prepare the SEED-V dataset
    sub_list = []
    for sub in os.listdir(root):
        if sub.endswith("cnt") and not sub.startswith("."):
            sub_path = os.path.join(root, sub)
            if os.path.isfile(sub_path):
                sub_list.append(sub_path)

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
