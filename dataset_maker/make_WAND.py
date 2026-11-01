import os
import mne
import h5py
import numpy as np
import nibabel as nib
from typing import Union
from pathlib import Path


ch_names = [
    'MLC11-3305', 'MLC12-3305', 'MLC13-3305', 'MLC14-3305', 'MLC15-3305', 'MLC16-3305',
    'MLC17-3305', 'MLC21-3305', 'MLC22-3305', 'MLC23-3305', 'MLC24-3305', 'MLC25-3305',
    'MLC31-3305', 'MLC32-3305', 'MLC41-3305', 'MLC42-3305', 'MLC51-3305', 'MLC52-3305',
    'MLC53-3305', 'MLC54-3305', 'MLC55-3305', 'MLC61-3305', 'MLC62-3305', 'MLC63-3305',
    'MLF11-3305', 'MLF12-3305', 'MLF13-3305', 'MLF14-3305', 'MLF21-3305', 'MLF22-3305',
    'MLF23-3305', 'MLF24-3305', 'MLF25-3305', 'MLF31-3305', 'MLF32-3305', 'MLF33-3305',
    'MLF34-3305', 'MLF35-3305', 'MLF41-3305', 'MLF42-3305', 'MLF43-3305', 'MLF44-3305',
    'MLF45-3305', 'MLF46-3305', 'MLF51-3305', 'MLF52-3305', 'MLF53-3305', 'MLF54-3305',
    'MLF55-3305', 'MLF56-3305', 'MLF61-3305', 'MLF62-3305', 'MLF63-3305', 'MLF64-3305',
    'MLF65-3305', 'MLF66-3305', 'MLF67-3305', 'MLO11-3305', 'MLO12-3305', 'MLO13-3305',
    'MLO14-3305', 'MLO21-3305', 'MLO22-3305', 'MLO23-3305', 'MLO24-3305', 'MLO31-3305',
    'MLO32-3305', 'MLO33-3305', 'MLO34-3305', 'MLO41-3305', 'MLO42-3305', 'MLO43-3305',
    'MLO44-3305', 'MLO51-3305', 'MLO52-3305', 'MLO53-3305', 'MLP11-3305', 'MLP12-3305',
    'MLP21-3305', 'MLP22-3305', 'MLP23-3305', 'MLP31-3305', 'MLP32-3305', 'MLP33-3305',
    'MLP34-3305', 'MLP35-3305', 'MLP41-3305', 'MLP42-3305', 'MLP43-3305', 'MLP44-3305',
    'MLP45-3305', 'MLP51-3305', 'MLP52-3305', 'MLP53-3305', 'MLP54-3305', 'MLP55-3305',
    'MLP56-3305', 'MLP57-3305', 'MLT11-3305', 'MLT12-3305', 'MLT13-3305', 'MLT14-3305',
    'MLT15-3305', 'MLT16-3305', 'MLT21-3305', 'MLT22-3305', 'MLT23-3305', 'MLT24-3305',
    'MLT25-3305', 'MLT26-3305', 'MLT27-3305', 'MLT31-3305', 'MLT32-3305', 'MLT33-3305',
    'MLT34-3305', 'MLT35-3305', 'MLT36-3305', 'MLT37-3305', 'MLT41-3305', 'MLT42-3305',
    'MLT43-3305', 'MLT44-3305', 'MLT45-3305', 'MLT46-3305', 'MLT47-3305', 'MLT51-3305',
    'MLT52-3305', 'MLT53-3305', 'MLT54-3305', 'MLT55-3305', 'MLT56-3305', 'MLT57-3305',
    'MRC11-3305', 'MRC12-3305', 'MRC13-3305', 'MRC14-3305', 'MRC15-3305', 'MRC16-3305',
    'MRC17-3305', 'MRC21-3305', 'MRC22-3305', 'MRC23-3305', 'MRC24-3305', 'MRC25-3305',
    'MRC31-3305', 'MRC32-3305', 'MRC41-3305', 'MRC42-3305', 'MRC51-3305', 'MRC52-3305',
    'MRC53-3305', 'MRC54-3305', 'MRC55-3305', 'MRC61-3305', 'MRC62-3305', 'MRC63-3305',
    'MRF11-3305', 'MRF12-3305', 'MRF13-3305', 'MRF14-3305', 'MRF21-3305', 'MRF22-3305',
    'MRF23-3305', 'MRF24-3305', 'MRF25-3305', 'MRF31-3305', 'MRF32-3305', 'MRF33-3305',
    'MRF34-3305', 'MRF35-3305', 'MRF41-3305', 'MRF42-3305', 'MRF43-3305', 'MRF44-3305',
    'MRF45-3305', 'MRF46-3305', 'MRF51-3305', 'MRF52-3305', 'MRF53-3305', 'MRF54-3305',
    'MRF55-3305', 'MRF56-3305', 'MRF61-3305', 'MRF62-3305', 'MRF63-3305', 'MRF64-3305',
    'MRF65-3305', 'MRF66-3305', 'MRF67-3305', 'MRO11-3305', 'MRO12-3305', 'MRO13-3305',
    'MRO14-3305', 'MRO21-3305', 'MRO22-3305', 'MRO23-3305', 'MRO24-3305', 'MRO31-3305',
    'MRO32-3305', 'MRO33-3305', 'MRO34-3305', 'MRO41-3305', 'MRO42-3305', 'MRO43-3305',
    'MRO44-3305', 'MRO51-3305', 'MRO52-3305', 'MRO53-3305', 'MRP11-3305', 'MRP12-3305',
    'MRP21-3305', 'MRP22-3305', 'MRP23-3305', 'MRP31-3305', 'MRP32-3305', 'MRP33-3305',
    'MRP34-3305', 'MRP35-3305', 'MRP41-3305', 'MRP42-3305', 'MRP43-3305', 'MRP44-3305',
    'MRP45-3305', 'MRP51-3305', 'MRP52-3305', 'MRP53-3305', 'MRP54-3305', 'MRP55-3305',
    'MRP56-3305', 'MRP57-3305', 'MRT11-3305', 'MRT12-3305', 'MRT13-3305', 'MRT14-3305',
    'MRT15-3305', 'MRT16-3305', 'MRT21-3305', 'MRT22-3305', 'MRT23-3305', 'MRT24-3305',
    'MRT25-3305', 'MRT26-3305', 'MRT27-3305', 'MRT31-3305', 'MRT32-3305', 'MRT33-3305',
    'MRT34-3305', 'MRT35-3305', 'MRT36-3305', 'MRT37-3305', 'MRT41-3305', 'MRT42-3305',
    'MRT43-3305', 'MRT44-3305', 'MRT45-3305', 'MRT46-3305', 'MRT47-3305', 'MRT51-3305',
    'MRT52-3305', 'MRT53-3305', 'MRT54-3305', 'MRT55-3305', 'MRT56-3305', 'MRT57-3305',
    'MZC01-3305', 'MZC02-3305', 'MZC03-3305', 'MZC04-3305', 'MZF01-3305', 'MZF02-3305',
    'MZF03-3305', 'MZO01-3305', 'MZO02-3305', 'MZO03-3305', 'MZP01-3305'
]

BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_2mm.nii.gz").get_fdata()
BN_atlas_7T = nib.load("./Brainnetome Atlas/BN_Atlas_246_1.6mm.nii.gz").get_fdata()


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
        # discarding the first 10s and the last 10s time points in order to obtain the stationary signals
        meg_raw.crop(tmin=10, tmax=meg_raw.n_times / meg_raw.info['sfreq'] - 10)
        # set the 2-second length event
        meg_raw.sample_event = mne.make_fixed_length_events(meg_raw, id=10, duration=2, overlap=0)
        meg_raw.sample_event_id = {'2 seconds': 10}
        # segment the whole signals into 2 seconds sample
        meg_epochs = mne.Epochs(meg_raw, meg_raw.sample_event, meg_raw.sample_event_id, baseline=None, tmin=0, tmax=2, preload=True, reject=None, proj=False)

        signals = meg_epochs.get_data(units=dict(grad='fT/cm', mag='fT'))
        for idx, signal in enumerate(signals):
            save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ""
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "WAND_rMEG_error.log"), "a") as f:
            f.write(str(fname) + "\n")


def preprocessing_task(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split, task = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw meg data
        meg_raw = mne.io.read_raw_ctf(fname, preload=preload)
        # pick the events
        events, evnet_id = mne.events_from_annotations(meg_raw)
        # pick common channels between whole channel and current subject, and reorder those channel with specific order
        meg_raw.reorder_channels(list(filter(lambda x: x not in (set(ch_names) - set(meg_raw.ch_names)), ch_names)))
        # FIR bandpass and notch filter for rest state
        meg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        meg_raw.notch_filter(50.0, n_jobs=n_jobs)  # Europe power line

        if task == 'visual':
            # resample brain signal to 200 HZ
            meg_raw, events = meg_raw.resample(200, events=events, n_jobs=n_jobs)
            # pick the events
            picked_events = mne.pick_events(events, include=1)
            # segment the whole signals into sample
            meg_epochs = mne.Epochs(meg_raw, picked_events, baseline=None, tmin=-1, tmax=1, preload=True, reject=None, proj=False)

            signals = meg_epochs.get_data(units=dict(grad='fT/cm', mag='fT'))

            # save signals
            for idx, signal in enumerate(signals):
                save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
                target = ''
                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
        elif task == 'simon':
            # resample brain signal to 200 HZ
            meg_raw, events = meg_raw.resample(200, events=events, n_jobs=n_jobs)
            # pick the events
            picked_events = mne.pick_events(events, include=1)

            signals = meg_raw.get_data(units=dict(grad='fT/cm', mag='fT'))

            # save signals
            for idx in range(len(picked_events) - 1):
                save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
                signal = signals[:, picked_events[idx, 0]:picked_events[idx+1, 0]]
                if signal.shape[1] > 448:
                    continue
                target = ''
                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
        elif task == 'auditorymotor':
            # resample brain signal to 100 HZ
            meg_raw, events = meg_raw.resample(100, events=events, n_jobs=n_jobs)
            # segment the whole signals into sample
            meg_epochs = mne.Epochs(meg_raw, events, baseline=None, tmin=0, tmax=4, preload=True, reject=None, proj=False)

            signals = meg_epochs.get_data(units=dict(grad='fT/cm', mag='fT'))

            # save signals
            for idx, signal in enumerate(signals):
                save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
                target = ''
                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
        elif task == 'mmn':
            # resample brain signal to 200 HZ
            meg_raw, events = meg_raw.resample(200, events=events, n_jobs=n_jobs)

            signals = meg_raw.get_data(units=dict(grad='fT/cm', mag='fT'))

            # save signals
            for idx in range(0, len(events) - 4, 4):
                save_fname = str(fname).split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
                signal = signals[:, events[idx, 0]:events[idx + 4, 0]]
                if signal.shape[1] > 448:
                    continue
                target = ''
                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "WAND_tMEG_error.log"), "a") as f:
            f.write(str(fname) + "\n")


def preparing_rest(subject: tuple, save_path: Union[str, Path], atlas):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI data
        fMRI_voxel = nib.load(fname).get_fdata()
        # parcel voxel signal to brain region
        fMRI_roi = np.zeros((int(atlas.max()), fMRI_voxel.shape[-1]))
        roi = np.unique(atlas)
        # exclude roi 0
        roi = roi[roi != 0]
        for i, value in enumerate(roi):
            fMRI_roi[i] = fMRI_voxel[atlas == value].mean(axis=0)

        signals = [fMRI_roi]
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "WAND_rfMRI_error.log"), "a") as f:
            f.write(fname + "\n")


def preparing_task(subject: tuple, save_path: Union[str, Path]):
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

        signals = [fMRI_roi]
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "WAND_tfMRI_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root_meg = "/path/to/WAND"
    root_fMRI = "/path/to/WAND/derivatives/fmriprep"
    save_path_rmeg = "/path/to/BrainSignalDataset/WAND_MEG_Rest"
    save_path_tmeg = "/path/to/BrainSignalDataset/WAND_MEG_Task"
    save_path_rfMRI = "/path/to/BrainSignalDataset/WAND_fMRI_Rest"
    save_path_tfMRI = "/path/to/BrainSignalDataset/WAND_fMRI_Task"

    # rest meg
    sub_list_rmeg = []
    for sub in os.listdir(root_meg):
        if sub.startswith('sub-'):
            if os.path.exists(os.path.join(root_meg, sub, 'ses-01', 'meg')):
                sub_path = os.path.join(root_meg, sub, 'ses-01', 'meg', f'{sub}_ses-01_task-resting.ds')
                if os.path.exists(sub_path):
                    sub_list_rmeg.append((sub_path, 'train'))

    # preprocess the rest meg and save them
    for sub in sub_list_rmeg:
        preprocessing_rest(sub, save_path_rmeg)

    # task meg
    sub_list_tmeg = []
    tasks = ['visual', 'simon', 'auditorymotor', 'mmn']
    for sub in os.listdir(root_meg):
        if sub.startswith('sub-'):
            if os.path.exists(os.path.join(root_meg, sub, 'ses-01', 'meg')):
                for task in tasks:
                    sub_path = os.path.join(root_meg, sub, 'ses-01', 'meg', f'{sub}_ses-01_task-{task}.ds')
                    if os.path.exists(sub_path):
                        sub_list_tmeg.append((sub_path, 'train', task))

    # prepare the tmeg and save them
    for sub in sub_list_tmeg:
        preprocessing_task(sub, save_path_tmeg)

    # rest fMRI
    sub_list_rfMRI = []
    for sub in os.listdir(root_fMRI):
        if sub.startswith('sub-'):
            if os.path.exists(os.path.join(root_fMRI, sub, 'ses-03', 'func')):
                sub_path = os.path.join(root_fMRI, sub, 'ses-03', 'func', f'{sub}_ses-03_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                if os.path.isfile(sub_path):
                    sub_list_rfMRI.append((sub_path, 'train'))

    # prepare the rfMRI and save them
    for sub in sub_list_rfMRI:
        preparing_rest(sub, save_path_rfMRI, BN_atlas)

    # tfMRI
    sub_list_tfMRI = []
    tasks = ['categorylocaliser_run-1', 'categorylocaliser_run-2', 'reversallearning']
    for sub in os.listdir(root_fMRI):
        if sub.startswith('sub-'):
            if os.path.exists(os.path.join(root_fMRI, sub, 'ses-03', 'func')):
                for task in tasks:
                    sub_path = os.path.join(root_fMRI, sub, 'ses-03', 'func', f'{sub}_ses-03_task-{task}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                    if os.path.isfile(sub_path):
                        sub_list_tfMRI.append((sub_path, 'train'))

    # prepare the tfMRI and save them
    for sub in sub_list_tfMRI:
        preparing_task(sub, save_path_tfMRI)

    # 7T rest fMRI
    sub_list_rfMRI = []
    for sub in os.listdir(root_fMRI):
        if sub.startswith('sub-'):
            if os.path.exists(os.path.join(root_fMRI, sub, 'ses-06', 'func')):
                sub_path = os.path.join(root_fMRI, sub, 'ses-06', 'func', f'{sub}_ses-06_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                if os.path.isfile(sub_path):
                    sub_list_rfMRI.append((sub_path, 'train'))

    # prepare the rfMRI and save them
    for sub in sub_list_rfMRI:
        preparing_rest(sub, save_path_rfMRI, BN_atlas_7T)
