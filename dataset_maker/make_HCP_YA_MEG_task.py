import os
import mne
import h5py
import numpy as np
import pymatreader
from typing import Union
from pathlib import Path


ch_names = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12',
    'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23',
    'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34',
    'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45',
    'A46', 'A47', 'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56',
    'A57', 'A58', 'A59', 'A60', 'A61', 'A62', 'A63', 'A64', 'A65', 'A66', 'A67',
    'A68', 'A69', 'A70', 'A71', 'A72', 'A73', 'A74', 'A75', 'A76', 'A77', 'A78',
    'A79', 'A80', 'A81', 'A82', 'A83', 'A84', 'A85', 'A86', 'A87', 'A88', 'A89',
    'A90', 'A91', 'A92', 'A93', 'A94', 'A95', 'A96', 'A97', 'A98', 'A99', 'A100',
    'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 'A107', 'A108', 'A109', 'A110',
    'A111', 'A112', 'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120',
    'A121', 'A122', 'A123', 'A124', 'A125', 'A126', 'A127', 'A128', 'A129', 'A130',
    'A131', 'A132', 'A133', 'A134', 'A135', 'A136', 'A137', 'A138', 'A139', 'A140',
    'A141', 'A142', 'A143', 'A144', 'A145', 'A146', 'A147', 'A148', 'A149', 'A150',
    'A151', 'A152', 'A153', 'A154', 'A155', 'A156', 'A157', 'A158', 'A159', 'A160',
    'A161', 'A162', 'A163', 'A164', 'A165', 'A166', 'A167', 'A168', 'A169', 'A170',
    'A171', 'A172', 'A173', 'A174', 'A175', 'A176', 'A177', 'A178', 'A179', 'A180',
    'A181', 'A182', 'A183', 'A184', 'A185', 'A186', 'A187', 'A188', 'A189', 'A190',
    'A191', 'A192', 'A193', 'A194', 'A195', 'A196', 'A197', 'A198', 'A199', 'A200',
    'A201', 'A202', 'A203', 'A204', 'A205', 'A206', 'A207', 'A208', 'A209', 'A210',
    'A211', 'A212', 'A213', 'A214', 'A215', 'A216', 'A217', 'A218', 'A219', 'A220',
    'A221', 'A222', 'A223', 'A224', 'A225', 'A226', 'A227', 'A228', 'A229', 'A230',
    'A231', 'A232', 'A233', 'A234', 'A235', 'A236', 'A237', 'A238', 'A239', 'A240',
    'A241', 'A242', 'A243', 'A244', 'A245', 'A246', 'A247', 'A248'
]


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8):
    fname, split, task = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the MEG from FieldTrip mat using pymatreader because of the inhomogeneous dimension
        meg_data = pymatreader.read_mat(fname)['data']

        # initialize the information about MEG
        loaded_ch_names = meg_data['label']
        ch_types = ['mag'] * len(loaded_ch_names)
        sampling_freq = meg_data['fsample']

        # create info object
        info = mne.create_info(loaded_ch_names, ch_types=ch_types, sfreq=sampling_freq)
        for idx, trial in enumerate(meg_data['trial']):
            # create raw object for MEG
            meg_epochs = mne.io.RawArray(trial, info)

            # reorder those channel with specific order
            meg_epochs.reorder_channels(list(filter(lambda x: x not in (set(ch_names) - set(meg_epochs.ch_names)), ch_names)))

            match task:
                case "TFLA" | "TEMG":
                    # resample brain signal to 200 HZ for TFLA and TEMG Task
                    meg_epochs.resample(200, n_jobs=n_jobs)
                case "TIM" | "TRESP":
                    # resample brain signal to 120 HZ for TIM and TRESP Task
                    meg_epochs.resample(120, n_jobs=n_jobs)
                case "TEV":
                    # resample brain signal to 80 HZ for TEV Task
                    meg_epochs.resample(80, n_jobs=n_jobs)
                case "BSENT":
                    # resample brain signal to 50 HZ for BSENT Task
                    meg_epochs.resample(50, n_jobs=n_jobs)

            signal = meg_epochs.get_data(units='fT')
            if np.any(np.isnan(signal)):
                continue
            if signal.shape[1] > 512:
                continue
                # raise ValueError(' the length must be lower than 512')

            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            target = ''
            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "HCP_YA_MEG_Task_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/HCP_YA_MEG_Task"
    save_path = "/path/to/BrainSignalDataset/HCP_YA_MEG_Task"

    sub_list = []
    # prepare the train dataset for Motor task
    experiments = ['Motort', 'Wrkmem', 'StoryM']
    tasks = {'Motort': ['TFLA', 'TEMG'], 'Wrkmem': ['TIM', 'TRESP'], 'StoryM': ['TEV', 'TRESP', 'BSENT']}
    task_sess = {'Motort': [10, 11], 'Wrkmem': [6, 7], 'StoryM': [8, 9]}

    for experiment in experiments:
        for sub in os.listdir(os.path.join(root, experiment)):
            if '.' not in sub:
                mat_path = os.path.join(root, experiment, sub, 'MEG', experiment, 'tmegpreproc')
                for task in tasks[experiment]:
                    for task_ses in task_sess[experiment]:
                        sub_path = os.path.join(mat_path, f'{sub}_MEG_{task_ses}-{experiment}_tmegpreproc_{task}.mat')
                        if os.path.isfile(sub_path):
                            sub_list.append((sub_path, 'train', task))

    # preprocess the data and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
