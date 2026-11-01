import os
import mne
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union
from pathlib import Path


ch_names = ['MLC11-4304', 'MLC12-4304', 'MLC13-4304', 'MLC14-4304', 'MLC15-4304', 'MLC16-4304', 'MLC17-4304',
            'MLC21-4304', 'MLC22-4304', 'MLC23-4304', 'MLC24-4304', 'MLC25-4304', 'MLC31-4304', 'MLC32-4304',
            'MLC41-4304', 'MLC42-4304', 'MLC51-4304', 'MLC52-4304', 'MLC53-4304', 'MLC54-4304', 'MLC55-4304',
            'MLC61-4304', 'MLC62-4304', 'MLC63-4304', 'MLF11-4304', 'MLF12-4304', 'MLF13-4304', 'MLF14-4304',
            'MLF21-4304', 'MLF22-4304', 'MLF23-4304', 'MLF24-4304', 'MLF25-4304', 'MLF31-4304', 'MLF32-4304',
            'MLF33-4304', 'MLF34-4304', 'MLF35-4304', 'MLF41-4304', 'MLF42-4304', 'MLF43-4304', 'MLF44-4304',
            'MLF45-4304', 'MLF46-4304', 'MLF51-4304', 'MLF52-4304', 'MLF53-4304', 'MLF54-4304', 'MLF55-4304',
            'MLF56-4304', 'MLF61-4304', 'MLF62-4304', 'MLF63-4304', 'MLF64-4304', 'MLF65-4304', 'MLF66-4304',
            'MLF67-4304', 'MLO11-4304', 'MLO12-4304', 'MLO13-4304', 'MLO14-4304', 'MLO21-4304', 'MLO22-4304',
            'MLO23-4304', 'MLO24-4304', 'MLO31-4304', 'MLO32-4304', 'MLO33-4304', 'MLO34-4304', 'MLO41-4304',
            'MLO42-4304', 'MLO43-4304', 'MLO44-4304', 'MLO51-4304', 'MLO52-4304', 'MLO53-4304', 'MLP11-4304',
            'MLP12-4304', 'MLP21-4304', 'MLP22-4304', 'MLP23-4304', 'MLP31-4304', 'MLP32-4304', 'MLP33-4304',
            'MLP34-4304', 'MLP35-4304', 'MLP41-4304', 'MLP42-4304', 'MLP43-4304', 'MLP44-4304', 'MLP45-4304',
            'MLP51-4304', 'MLP52-4304', 'MLP53-4304', 'MLP54-4304', 'MLP55-4304', 'MLP56-4304', 'MLP57-4304',
            'MLT11-4304', 'MLT12-4304', 'MLT13-4304', 'MLT14-4304', 'MLT15-4304', 'MLT16-4304', 'MLT21-4304',
            'MLT22-4304', 'MLT23-4304', 'MLT24-4304', 'MLT25-4304', 'MLT26-4304', 'MLT27-4304', 'MLT31-4304',
            'MLT32-4304', 'MLT33-4304', 'MLT34-4304', 'MLT35-4304', 'MLT36-4304', 'MLT37-4304', 'MLT41-4304',
            'MLT42-4304', 'MLT43-4304', 'MLT44-4304', 'MLT45-4304', 'MLT46-4304', 'MLT47-4304', 'MLT51-4304',
            'MLT52-4304', 'MLT53-4304', 'MLT54-4304', 'MLT55-4304', 'MLT56-4304', 'MLT57-4304', 'MRC11-4304',
            'MRC12-4304', 'MRC13-4304', 'MRC14-4304', 'MRC15-4304', 'MRC16-4304', 'MRC17-4304', 'MRC21-4304',
            'MRC22-4304', 'MRC23-4304', 'MRC24-4304', 'MRC25-4304', 'MRC31-4304', 'MRC32-4304', 'MRC41-4304',
            'MRC42-4304', 'MRC51-4304', 'MRC52-4304', 'MRC53-4304', 'MRC54-4304', 'MRC55-4304', 'MRC61-4304',
            'MRC62-4304', 'MRC63-4304', 'MRF11-4304', 'MRF12-4304', 'MRF13-4304', 'MRF14-4304', 'MRF21-4304',
            'MRF22-4304', 'MRF23-4304', 'MRF24-4304', 'MRF25-4304', 'MRF31-4304', 'MRF32-4304', 'MRF33-4304',
            'MRF34-4304', 'MRF35-4304', 'MRF41-4304', 'MRF42-4304', 'MRF43-4304', 'MRF44-4304', 'MRF45-4304',
            'MRF46-4304', 'MRF51-4304', 'MRF52-4304', 'MRF53-4304', 'MRF54-4304', 'MRF55-4304', 'MRF56-4304',
            'MRF61-4304', 'MRF62-4304', 'MRF63-4304', 'MRF64-4304', 'MRF65-4304', 'MRF66-4304', 'MRF67-4304',
            'MRO11-4304', 'MRO12-4304', 'MRO13-4304', 'MRO14-4304', 'MRO21-4304', 'MRO22-4304', 'MRO23-4304',
            'MRO24-4304', 'MRO31-4304', 'MRO32-4304', 'MRO33-4304', 'MRO34-4304', 'MRO41-4304', 'MRO42-4304',
            'MRO43-4304', 'MRO44-4304', 'MRO51-4304', 'MRO52-4304', 'MRO53-4304', 'MRP11-4304', 'MRP12-4304',
            'MRP21-4304', 'MRP22-4304', 'MRP23-4304', 'MRP31-4304', 'MRP32-4304', 'MRP33-4304', 'MRP34-4304',
            'MRP35-4304', 'MRP41-4304', 'MRP42-4304', 'MRP43-4304', 'MRP44-4304', 'MRP45-4304', 'MRP51-4304',
            'MRP52-4304', 'MRP53-4304', 'MRP54-4304', 'MRP55-4304', 'MRP56-4304', 'MRP57-4304', 'MRT11-4304',
            'MRT12-4304', 'MRT13-4304', 'MRT14-4304', 'MRT15-4304', 'MRT16-4304', 'MRT21-4304', 'MRT22-4304',
            'MRT23-4304', 'MRT24-4304', 'MRT25-4304', 'MRT26-4304', 'MRT27-4304', 'MRT31-4304', 'MRT32-4304',
            'MRT33-4304', 'MRT34-4304', 'MRT35-4304', 'MRT36-4304', 'MRT37-4304', 'MRT41-4304', 'MRT42-4304',
            'MRT43-4304', 'MRT44-4304', 'MRT45-4304', 'MRT46-4304', 'MRT47-4304', 'MRT51-4304', 'MRT52-4304',
            'MRT53-4304', 'MRT54-4304', 'MRT55-4304', 'MRT56-4304', 'MRT57-4304', 'MZC01-4304', 'MZC02-4304',
            'MZC03-4304', 'MZC04-4304', 'MZF01-4304', 'MZF02-4304', 'MZF03-4304', 'MZO01-4304', 'MZO02-4304',
            'MZO03-4304', 'MZP01-4304']


BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_2mm.nii.gz").get_fdata()


def preprocessing_rest(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw meg data
        meg_raw = mne.io.read_raw_ctf(fname, preload=preload)
        # pick common channels between whole channel and current subject, and reorder those channel with specific order
        meg_raw.reorder_channels(list(filter(lambda x: x not in (set(ch_names) - set(meg_raw.ch_names)), ch_names)))
        # FIR bandpass and notch filter for rest state
        meg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        meg_raw.notch_filter(50.0, n_jobs=n_jobs)  # Europe power line
        # resample brain signal to 200 HZ
        meg_raw.resample(200, n_jobs=n_jobs)
        # set the 2-second length event
        meg_raw.sample_event = mne.make_fixed_length_events(meg_raw, id=10, duration=2, overlap=0)
        meg_raw.sample_event_id = {'2 seconds': 10}
        # segment the whole signals into 2 seconds sample
        meg_epochs = mne.Epochs(meg_raw, meg_raw.sample_event, meg_raw.sample_event_id, baseline=None, tmin=0, tmax=2, preload=True, reject=None, proj=False)

        signals = meg_epochs.get_data(units='fT')
        for idx, signal in enumerate(signals):
            save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "MOUS_rMEG_error.log"), "a") as f:
            f.write(str(fname) + "\n")


def preprocessing_task(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        meg_raw = mne.io.read_raw_ctf(fname, preload=preload)
        # pick common channels between whole channel and current subject, and reorder those channel with specific order
        meg_raw.reorder_channels(list(filter(lambda x: x not in (set(ch_names) - set(meg_raw.ch_names)), ch_names)))
        # FIR bandpass filter for visual task (half of the sample frequency)
        # meg_raw.filter(l_freq=1, h_freq=15, n_jobs=n_jobs)
        # FIR bandpass filter for auditory task (half of the sample frequency)
        meg_raw.filter(l_freq=1, h_freq=25, n_jobs=n_jobs)
        # read the events for task state MEG
        events_raw = pd.read_csv(str(fname).replace('_meg.ds', '_events.tsv'), sep='\t')
        # visual task
        # sample_idx = events_raw[(events_raw['type'] == 'frontpanel trigger') & (events_raw['value'] == '20')]['sample'].to_numpy() - 1  # python start from 0
        # events = np.vstack((sample_idx.astype(int), np.zeros(len(sample_idx), dtype=int), 20*np.ones(len(sample_idx), dtype=int))).T
        # auditory task
        sample_idx = events_raw[(events_raw['type'] == 'Sound') | (events_raw['value'] == '15 End of file')]['sample'].to_numpy() - 1  # python start from 0
        event_type = events_raw[(events_raw['type'] == 'Sound') | (events_raw['value'] == '15 End of file')]['type'].replace('Sound', 14).replace('Nothing', 15).to_numpy()
        events = np.vstack((sample_idx, np.zeros(len(sample_idx), dtype=int), event_type)).T
        # resample brain signal to 30 HZ for visual task
        # meg_raw, events = meg_raw.resample(30, events=events, n_jobs=n_jobs)
        # resample brain signal to 50 HZ for auditory task
        meg_raw, events = meg_raw.resample(50, events=events, n_jobs=n_jobs)

        signals = meg_raw.get_data(units='fT')
        # for idx in range(0, len(events)-1): # visual task
        for idx in range(len(events)//2):   # auditory task
            save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            # signal = signals[:, events[idx, 0]:events[idx+1, 0]]
            if events[2*idx,2] == 14 and events[2*idx+1, 2] == 15:
                signal = signals[:, events[2*idx, 0]:events[2*idx+1, 0]]
            target = ''

            if signal.shape[1] > 448 or signal.shape[1] < 50:
                del signal
                continue
                # raise ValueError(' the length must be lower than 448')

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
            del signal
    except:
        with open(os.path.join(save_path, "MOUS_tMEG_error.log"), "a") as f:
            f.write(str(fname) + "\n")


def preparing_rest(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI data
        fMRI_voxel = nib.load(fname).get_fdata()
        # parcel voxel signal to brain region
        fMRI_roi = np.zeros((int(BN_atlas.max()), fMRI_voxel.shape[-1]))
        roi = np.unique(BN_atlas)
        # exclude roi 0
        roi = roi[roi != 0]
        for i, value in enumerate(roi):
            fMRI_roi[i] = fMRI_voxel[BN_atlas == value].mean(axis=0)

        # read the events for task state MEG
        signals = np.stack([fMRI_roi])
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "Mous_rfMRI_error.log"), "a") as f:
            f.write(str(fname) + "\n")


def preparing_task(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI data
        fMRI_voxel = nib.load(fname).get_fdata()
        # parcel voxel signal to brain region
        fMRI_roi = np.zeros((int(BN_atlas.max()), fMRI_voxel.shape[-1]))
        roi = np.unique(BN_atlas)
        # exclude roi 0
        roi = roi[roi != 0]
        for i, value in enumerate(roi):
            fMRI_roi[i] = fMRI_voxel[BN_atlas == value].mean(axis=0)

        # read the events for task state MEG
        events_raw = pd.read_csv(str(fname).replace('derivatives/fmriprep', '').replace('space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', 'events.tsv'), sep='\t')
        sample_idx = events_raw[(events_raw['type'] == 'Picture') & (events_raw['value'] == 'WOORDEN')]['volume'].to_numpy() - 1  # python start from 0
        # auditory task
        ts_length = int(sample_idx[abs(sample_idx - 320).argmin()])
        signals = [fMRI_roi[:, :ts_length], fMRI_roi[:, ts_length:]]
        # visual task
        # ts_length = [int(sample_idx[abs(sample_idx - 290).argmin()]), int(sample_idx[abs(sample_idx - 580).argmin()])]
        # signals = [fMRI_roi[:, :ts_length[0]], fMRI_roi[:, ts_length[0]:ts_length[1]], fMRI_roi[:, ts_length[1]:]]
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "Mous_tfMRI_error.log"), "a") as f:
            f.write(str(fname) + "\n")


if __name__ == "__main__":
    root = "/path/to/MOUS"
    save_path_rmeg = "/path/to/BrainSignalDataset/MOUS_MEG_Rest"
    save_path_tmeg = "/path/to/BrainSignalDataset/MOUS_MEG_Task"
    save_path_rfMRI = "/path/to/BrainSignalDataset/MOUS_fMRI_Rest"
    save_path_tfMRI = "/path/to/BrainSignalDataset/MOUS_fMRI_Task"

    # prepare the train dataset for rest state MEG
    sub_list = []
    task = 'rest'
    for sub in os.listdir(root):
        meg_path = os.path.join(root, sub, 'meg')
        sub_paths = list(Path(meg_path).glob('*.ds'))
        for sub_path in sub_paths:
            if task in str(sub_path) and '._' not in str(sub_path):
                sub_list.append((sub_path, 'train'))

    # preprocess the meg and save them
    for sub in sub_list:
        preprocessing_rest(sub, save_path_rmeg)

    # prepare the train dataset for task meg
    sub_list = []
    task = 'auditory'
    # task = 'visual'
    for sub in os.listdir(root):
        meg_path = os.path.join(root, sub, 'meg')
        sub_paths = list(Path(meg_path).glob('*.ds'))
        for sub_path in sub_paths:
            if task in str(sub_path) and '._' not in str(sub_path):
                sub_list.append((sub_path, 'train'))

    # preprocess the eeg and save them
    for sub in sub_list:
        preprocessing_task(sub, save_path_tmeg)

    # fMRI
    sub_list = []
    for sub in os.listdir(os.path.join(root, 'derivatives', 'fmriprep')):
        sub_path = os.path.join(root, 'derivatives', 'fmriprep', sub, 'func', f'{sub}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        if os.path.isfile(sub_path):
            sub_list.append((sub_path, 'train'))

    # prepare the fMRI and save them
    for sub in sub_list:
        preparing_rest(sub, save_path_rfMRI)

    sub_list = []
    task = 'auditory'
    # task = 'visual'
    for sub in os.listdir(os.path.join(root, 'derivatives', 'fmriprep')):
        sub_path = os.path.join(root, 'derivatives', 'fmriprep', sub, 'func', f'{sub}_task-{task}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        if os.path.isfile(sub_path):
            sub_list.append((sub_path, 'train'))

    # prepare the fMRI and save them
    for sub in sub_list:
        preparing_task(sub, save_path_tfMRI)
