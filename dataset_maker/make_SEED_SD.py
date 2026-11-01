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


# Format: (start_time, end_time), unit: second
part1_timing = [
    [[68, 236], [304, 374], [454, 653], [702, 832], [881, 969], [1020, 1175], [1182, 1335], [1409, 1618], [1661, 1806], [1908, 2077], [2139, 2189], [2259, 2369], [2433, 2650], [2682, 2851], [2930, 3189], [3262, 3403], [3476, 3544], [3597, 3776], [3841, 3981], [4045, 4093], [4139, 4251], [4309, 4421], [4463, 4638], [4703, 4840]],
    [[102, 323], [381, 482], [555, 694], [763, 909], [974, 1188], [1246, 1357], [1410, 1549], [1614, 1798], [1848, 1986], [2060, 2143], [2222, 2462], [2515, 2565], [2623, 2769], [2815, 2923], [2982, 3158], [3223, 3284], [3424, 3611], [3674, 3870], [3916, 4098], [4150, 4193], [4243, 4392], [4476, 4652], [4715, 4813], [4887, 4963]],
    [[113, 283], [333, 463], [514, 606], [700, 882], [940, 1133], [1196, 1302], [1348, 1396], [1465, 1558], [1626, 1730], [1776, 1840], [1935, 2142], [2189, 2354], [2441, 2598], [2653, 2730], [2789, 2904], [2954, 3131], [3195, 3252], [3327, 3397], [3441, 3624], [3677, 3766], [3822, 3981], [4033, 4188], [4257, 4422], [4475, 4631]]
]

# Format: (start_time, end_time), unit: second
part2_timing = [
    [[30, 198], [266, 336], [416, 615], [664, 794], [843, 931], [982, 1137], [1144, 1297], [1371, 1580], [1623, 1768], [1870, 2039], [2101, 2151], [2221, 2331], [2395, 2612], [2644, 2813], [2892, 3151], [3224, 3365], [3438, 3506], [3559, 3738], [3803, 3943], [4007, 4055], [4101, 4213], [4271, 4383], [4425, 4600], [4665, 4802]],
    [[64, 285], [343, 444], [517, 656], [725, 871], [936, 1150], [1208, 1319], [1372, 1511], [1576, 1760], [1810, 1948], [2022, 2105], [2184, 2424], [2477, 2527], [2585, 2731], [2777, 2885], [2944, 3120], [3185, 3246], [3386, 3573], [3636, 3832], [3878, 4060], [4112, 4155], [4205, 4354], [4438, 4614], [4677, 4775], [4849, 4925]],
    [[75, 245], [295, 425], [476, 568], [662, 844], [902, 1095], [1158, 1264], [1310, 1358], [1427, 1520], [1588, 1692], [1738, 1802], [1897, 2104], [2151, 2316], [2403, 2560], [2615, 2692], [2751, 2866], [2916, 3093], [3157, 3214], [3289, 3359], [3403, 3586], [3639, 3728], [3784, 3943], [3995, 4150], [4219, 4384], [4437, 4593]]
]


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_cnt(fname, preload=preload)
        # pick 62 common channels across the SEED dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # define the timing
        if 'part1' in fname:
            if 'session_1' in fname:
                timing = part1_timing[0]
            elif 'session_2' in fname:
                timing = part1_timing[1]
            elif 'session_3' in fname:
                timing = part1_timing[2]
        elif 'part2' in fname:
            if 'session_1' in fname:
                timing = part2_timing[0]
            elif 'session_2' in fname:
                timing = part2_timing[1]
            elif 'session_3' in fname:
                timing = part2_timing[2]
        # pick the events
        events = []
        for i in range(0, len(timing)):
            events.append(mne.make_fixed_length_events(eeg_raw, start=timing[i][0], stop=timing[i][1], id=10, duration=10, overlap=0))
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
        with open(os.path.join(save_path, "SEED_SD_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/SEED-SD/eeg_raw"
    save_path = "/path/to/BrainSignalDataset/SEED_SD"

    # prepare the SEED-SD dataset
    sub_list = []
    for part in os.listdir(root):
        if os.path.isdir(os.path.join(root, part)):
            for sess in os.listdir(os.path.join(root, part)):
                if os.path.isdir(os.path.join(root, part, sess)):
                    for sub in os.listdir(os.path.join(root, part, sess)):
                        if sub.endswith("cnt") and not sub.startswith("."):
                            sub_path = os.path.join(root, part, sess, sub)
                            if os.path.isfile(sub_path):
                                sub_list.append((sub_path, 'train'))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
