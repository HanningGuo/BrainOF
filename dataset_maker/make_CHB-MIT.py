import os
import re
import mne
import h5py
import numpy as np
from typing import Union
from pathlib import Path


ch_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3',
            'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8',
            'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10',
            'FT10-T8']
montage_subs = ['chb12_27', 'chb12_28', 'chb12_29']
channel_pairs = [(5, 1), (1, 2), (2, 3), (3, 9), (5, 6), (6, 7), (7, 8),
                 (8, 9), (15, 16), (16, 17), (17, 18), (18, 19), (15, 21), (21, 22),
                 (22, 23), (23, 19), (11, 12), (12, 13), (3, 2)]


def get_seizure_times(fname: Union[str, Path], target_fname: str):
    seizure_times = []
    with open(fname, 'r') as file:
        lines = file.readlines()

    inside_target_file = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect the file name line
        if line.startswith("File Name:"):
            inside_target_file = target_fname in line

        if inside_target_file:
            # Match any "Seizure Start Time" format (with or without number)
            start_match = re.match(r"Seizure(?: \d+)? Start Time:\s*(\d+)", line)
            end_match = re.match(r"Seizure(?: \d+)? End Time:\s*(\d+)", line)

            if start_match:
                start_time = int(start_match.group(1))
                # Look for matching end time in subsequent lines
                i += 1
                while i < len(lines):
                    end_line = lines[i].strip()
                    end_match = re.match(r"Seizure(?: \d+)? End Time:\s*(\d+)", end_line)
                    if end_match:
                        end_time = int(end_match.group(1))
                        seizure_times.append((start_time, end_time))
                        break
                    i += 1
        i += 1

        # Exit after the next file starts
        if inside_target_file and line.startswith("File Name:") and target_fname not in line:
            break

    return seizure_times


def change_reference_channel(eeg):
    sfreq = eeg.info['sfreq']
    for idx, (ch1, ch2) in enumerate(channel_pairs):
        info = mne.create_info([ch_names[idx]], sfreq, ch_types=['eeg'])
        # Create RawArray for the new channel
        channel = mne.io.RawArray(eeg[ch1-1][0] - eeg[ch2-1][0], info)
        # Add the new channel to the original raw object
        eeg.add_channels([channel], force_update_info=True)


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_edf(fname, preload=preload)   # you have to comment the age check line in the mne script
        # pick the seizure start and end time and set the annotations
        match = re.match(r'([a-zA-Z]+\d+)', fname.split(os.sep)[-1])
        fname_seizure_times = os.path.join(os.path.dirname(fname), match.group(1) + '-summary.txt')
        seizure_times = get_seizure_times(fname_seizure_times, fname.split(os.sep)[-1])
        onset, duration = [], []
        for start, end in seizure_times:
            # a trick to include previous 1s before seizure and next 1s after seizure
            onset.append(start - 1)
            duration.append(end - start + 1)
        seizure_annot = mne.Annotations(onset=onset, duration=duration, description=["BAD_seizure"] * len(onset))   # a trick to ignore the seizure when splitting normal segmentation
        eeg_raw.set_annotations(seizure_annot)
        # pick 22 common channels across the CHB-MIT dataset and reorder those channel with specific order
        if any(montage_sub in fname for montage_sub in montage_subs):
            change_reference_channel(eeg_raw)
        eeg_raw.reorder_channels(list(filter(lambda x: x not in (set(ch_names) - set(eeg_raw.ch_names)), ch_names)))
        # discarding the first 60s and the last 60s time points in order to obtain the stationary signals
        eeg_raw.crop(tmin=60, tmax=eeg_raw.n_times / eeg_raw.info['sfreq'] - 60)
        # FIR bandpass and notch filter
        eeg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        eeg_raw.notch_filter(60.0, n_jobs=n_jobs)   # U.S. power line
        # resample brain signal to 200 HZ
        eeg_raw.resample(200, n_jobs=n_jobs)

        # set the 10-second length seizure event
        eeg_raw.seizure_event = mne.events_from_annotations(eeg_raw, regexp='(?i)^.*seizure.*$', chunk_duration=5, tol=3)[0]
        eeg_raw.seizure_event_id = {'seizure': 1}
        # segment the seizure signals into 10 seconds sample with 5s overlap
        if len(eeg_raw.seizure_event) > 0:
            seizure_epochs = mne.Epochs(eeg_raw, eeg_raw.seizure_event, eeg_raw.seizure_event_id, baseline=None, tmin=0, tmax=10, preload=True, reject=None, proj=False, reject_by_annotation=False)

            seizures = seizure_epochs.get_data(units='uV')
            for idx, seizure in enumerate(seizures):
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_seizure_sample{idx:04d}.h5'
                target = np.array([1, 0])
                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=seizure)
                    hf.create_dataset("y", data=target)

        # set the 10-second length event
        eeg_raw.sample_event = mne.make_fixed_length_events(eeg_raw, id=10, duration=10, overlap=0)
        eeg_raw.sample_event_id = {'10 seconds': 10}
        # segment the whole signals into 10 seconds sample and exclude the seizure sample
        eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=10, preload=True, reject=None, proj=False, reject_by_annotation=True)

        signals = eeg_epochs.get_data(units='uV')
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = np.array([0, 1])
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "chb-mit_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/chbmit/1.0.0"
    save_path = "/path/to/BrainSignalDataset/CHB"
    train_subs = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07",
                  "chb08", "chb09", "chb10", "chb11", "chb12", "chb13", "chb14",
                  "chb15", "chb16", "chb17", "chb18", "chb19", "chb20"]
    val_subs = ["chb21", "chb22"]
    test_subs = ["chb23", "chb24"]

    # prepare the train, val and test dataset
    sub_list = []
    for sub in train_subs:
        for sub_path in os.listdir(os.path.join(root, sub)):
            if sub_path.endswith('.edf'):
                sub_list.append((os.path.join(root, sub, sub_path), 'train'))
    for sub in val_subs:
        for sub_path in os.listdir(os.path.join(root, sub)):
            if sub_path.endswith('.edf'):
                sub_list.append((os.path.join(root, sub, sub_path), 'val'))
    for sub in test_subs:
        for sub_path in os.listdir(os.path.join(root, sub)):
            if sub_path.endswith('.edf'):
                sub_list.append((os.path.join(root, sub, sub_path), 'test'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
