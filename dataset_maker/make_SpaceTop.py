import os
import h5py
import json
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union
from pathlib import Path
from functools import partial
from multiprocessing import Pool

BN_atlas = nib.load("./Brainnetome Atlas/BN_Atlas_246_2mm.nii.gz").get_fdata()


def preparing(subject: tuple, save_path: Union[str, Path]):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split), exist_ok=True)
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

        signals = []
        task = fname.split('task-')[1].split('_')[0]
        event_idx = find_events(fname, task)
        for i in range(len(event_idx) - 1):
            signals.append(fMRI_roi[:, event_idx[i]:event_idx[i+1]])
        signals.append(fMRI_roi[:, event_idx[-1]:])

        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''
            if 512 < signal.shape[-1] < 620:
                signal = signal[:, :512]
            if signal.shape[-1] < 80:
                continue

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "SpaceTop_error.log"), "a") as f:
            f.write(fname + "\n")


def find_events(fname, task):
    path = fname.split(os.sep)
    path.remove('derivatives')
    path.remove('fmriprep')
    fname_events = '/' + os.path.join(*path[:-1], path[-1].split('space')[0] + 'events.tsv')
    events = pd.read_csv(fname_events, sep="\t")
    fname_json = '/' + os.path.join(*path[:-1], path[-1].split('space')[0] + 'bold.json')
    with open(fname_json, "r", encoding="utf-8") as f:
        rt = json.load(f)['RepetitionTime']

    if 'alignvideo' in task:
        onset_timing = []
        videos = events[events['trial_type'] == 'video']
        for _, video in videos.iterrows():
            if video['duration'] < 300:
                onset_timing.append(video['onset'])
            else:
                onset_timing.append(video['onset'])
                onset_timing.append(video['onset'] + video['duration'] / 2)
        onset_timing = np.array(onset_timing)
        onset_idx = np.around(onset_timing / rt).astype(int)
    elif 'social' in task:
        onset_timing = events.loc[(events['trial_index'] == 1).idxmax()]['onset']
        onset_timing = np.append(onset_timing, events.loc[(events['trial_index'] == 7).idxmax()]['onset'])
        onset_idx = np.around(onset_timing / rt).astype(int)
    elif 'faces' in task:
        onset_timing = events['onset'].to_numpy()[0]
        onset_timing = np.append(onset_timing, events['onset'].to_numpy()[-1] / 2)
        onset_idx = np.around(onset_timing / rt).astype(int)
    elif 'narratives' in task:
        context = events.index[events['context'].ne(events['context'].shift())].tolist()
        onset_timing = []
        for i in range(0, len(context), 2):
            onset_timing.append(events.loc[context[i]]['onset'])
        onset_timing = np.array(onset_timing)
        onset_idx = np.around(onset_timing / rt).astype(int)
    elif 'shortvideo' in task:
        block = events.index[events['block_condition'].ne(events['block_condition'].shift())].tolist()
        onset_timing = []
        step = int(len(block) / 4)
        for i in range(0, len(block), step):
            onset_timing.append(events.loc[block[i]]['onset'])
        onset_timing = np.array(onset_timing)
        onset_idx = np.around(onset_timing / rt).astype(int)
    elif 'fractional' in task:
        if events['subtask_type'][0] == 'memory':
            onset_timing = events['onset'].to_numpy()[0]
            onset_timing = np.append(onset_timing, events['onset'].to_numpy()[-1] / 3)
            onset_timing = np.append(onset_timing, 2 * events['onset'].to_numpy()[-1] / 3)
            onset_idx = np.around(onset_timing / rt).astype(int)
        elif events['subtask_type'][0] in ['tomsaxe', 'posner', 'tomspunt']:
            step = np.around(events['trial_index'].to_numpy()[-1] / 3)
            onset_timing = events['onset'].to_numpy()[0]
            onset_timing = np.append(onset_timing, events.loc[(events['trial_index'] == (step+1)).idxmax()]['onset'])
            onset_timing = np.append(onset_timing, events.loc[(events['trial_index'] == (2*step+1)).idxmax()]['onset'])
            onset_idx = np.around(onset_timing / rt).astype(int)
        else:
            raise ValueError(f"Unknown subtask type: {events['subtask_type']}")
    else:
        raise ValueError(f"Unknown task type: {task}")

    return onset_idx


if __name__ == "__main__":
    root = "/path/to/SpaceTop"
    save_path = "/path/to/BrainSignalDataset/SpaceTop"

    # prepare the train dataset
    sub_list = []
    root = os.path.join(root, 'derivatives', 'fmriprep')
    for sub in os.listdir(root):
        if sub.startswith('sub') and not sub.endswith('.html'):
            for sess in os.listdir(os.path.join(root, sub)):
                if sess.startswith("ses"):
                    if os.path.exists(os.path.join(root, sub, sess, 'func')):
                        for file in os.listdir(os.path.join(root, sub, sess, 'func')):
                            if 'space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz' in file:
                                sub_path = os.path.join(root, sub, sess, 'func', file)
                                if os.path.isfile(sub_path):
                                    sub_list.append((sub_path, 'train'))

    # prepare the fMRI and save them
    with Pool(processes=8) as pool:
        preparing_fn = partial(preparing, save_path=save_path)
        pool.map(preparing_fn, sub_list)
    # for sub in sub_list:
    #     preparing(sub, save_path)
