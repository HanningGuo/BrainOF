import os
import io
import mne
import h5py
import lmdb
import pickle
import tarfile
from multiprocessing import Pool, cpu_count


ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
            'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
ch_mapping_ar = {
    'EEG FP1-REF': 'Fp1', 'EEG FP2-REF': 'Fp2', 'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4',
    'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4', 'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4',
    'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8',
    'EEG T3-REF': 'T3', 'EEG T4-REF': 'T4', 'EEG T5-REF': 'T5', 'EEG T6-REF': 'T6',
    'EEG FZ-REF': 'Fz', 'EEG CZ-REF': 'Cz', 'EEG PZ-REF': 'Pz'
}
ch_mapping_le = {
    'EEG FP1-LE': 'Fp1', 'EEG FP2-LE': 'Fp2', 'EEG F3-LE': 'F3', 'EEG F4-LE': 'F4',
    'EEG C3-LE': 'C3', 'EEG C4-LE': 'C4', 'EEG P3-LE': 'P3', 'EEG P4-LE': 'P4',
    'EEG O1-LE': 'O1', 'EEG O2-LE': 'O2', 'EEG F7-LE': 'F7', 'EEG F8-LE': 'F8',
    'EEG T3-LE': 'T3', 'EEG T4-LE': 'T4', 'EEG T5-LE': 'T5', 'EEG T6-LE': 'T6',
    'EEG FZ-LE': 'Fz', 'EEG CZ-LE': 'Cz', 'EEG PZ-LE': 'Pz'
}


def preprocessing(subject: tuple, n_jobs: int = 1, preload: bool = True):
    fname, split, save_path = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split), exist_ok=True)
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_edf(fname, preload=preload)
        # exclude too short eeg data
        if eeg_raw.duration < 200:
            return
        # rename special names to standard names of the international 10/20 system
        if '01_tcp_ar' in fname or '03_tcp_ar_a' in fname:
            eeg_raw.rename_channels(ch_mapping_ar)
        elif '02_tcp_le' in fname or '04_tcp_le_a' in fname:
            eeg_raw.rename_channels(ch_mapping_le)
        # pick 19 common channels across the TUEG dataset and reorder those channel with specific order
        eeg_raw.reorder_channels(ch_names)
        # discarding the first 60s and the last 60s time points in order to obtain the stationary signals
        eeg_raw.crop(tmin=60, tmax=eeg_raw.n_times / eeg_raw.info['sfreq'] - 60)
        # FIR bandpass and notch filter
        eeg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        eeg_raw.notch_filter(60.0, n_jobs=n_jobs)   # U.S. power line
        # resample brain signal to 200 HZ
        eeg_raw.resample(200, n_jobs=n_jobs)
        # set the 10-second length event
        eeg_raw.sample_event = mne.make_fixed_length_events(eeg_raw, id=10, duration=10, overlap=0)
        eeg_raw.sample_event_id = {'10 seconds': 10}
        # segment the whole signals into 10 seconds sample
        reject_criteria = dict(eeg=150e-6)  # 150 µV
        flat_criteria = dict(eeg=1e-6)  # 1 fT  # 1 fT/cm  # 1 µV
        eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=10, preload=True, reject=reject_criteria, flat=flat_criteria, proj=False)
        # set average reference
        eeg_epochs.set_eeg_reference(ref_channels='average', ch_type='eeg')

        signals = eeg_epochs.get_data(units='uV')
        tar_path = os.path.join(save_path, split, f"{fname.split(os.sep)[-1].split('.')[0]}.tar")
        with tarfile.open(tar_path, "w") as tar:
            for idx, signal in enumerate(signals):
                save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
                target = ''
                with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                    hf.create_dataset("signal", data=signal)
                    hf.create_dataset("y", data=target)
                tar.add(os.path.join(save_path, split, save_fname), arcname=save_fname)
                os.remove(os.path.join(save_path, split, save_fname))
    except:
        with open(os.path.join(save_path, "tueg_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = '/path/to/data/TUEG/v2.0.1/edf'
    root_tuab = '/path/to/data/TUAB/edf'
    save_path = "/path/to/BrainSignalDataset/TUEG_tmp"
    save_path_merge = '/path/to/BrainSignalDataset/TUEG'

    sub_tuab = []
    # prepare the subjects in the TUAB test dataset in order to exclude them in the pretraining dataset
    for diagnose in os.listdir(os.path.join(root_tuab, 'eval')):
        if os.path.exists(os.path.join(root_tuab, 'eval', diagnose, '01_tcp_ar')):
            subs = os.listdir(os.path.join(root_tuab, 'eval', diagnose, '01_tcp_ar'))
            for sub in subs:
                if os.path.exists(os.path.join(root_tuab, 'eval', diagnose, '01_tcp_ar', sub)):
                    sub_tuab.append(sub.split('_')[0])

    # prepare the train and val dataset
    sub_list = []
    for designator in os.listdir(root):
        sub_ids = os.listdir(os.path.join(root, designator))
        for sub_id in sub_ids:
            if sub_id not in sub_tuab:
                for sess in os.listdir(os.path.join(root, designator, sub_id)):
                    for montage in os.listdir(os.path.join(root, designator, sub_id, sess)):
                        for sub_path in os.listdir(os.path.join(root, designator, sub_id, sess, montage)):
                            if sub_path.endswith('.edf'):
                                sub_list.append((os.path.join(root, designator, sub_id, sess, montage, sub_path), 'train', save_path))

    # preprocess the data and save them
    with Pool(int(cpu_count() / 2)) as pool:
        pool.map(preprocessing, sub_list)

    file_tars = sorted(os.listdir(os.path.join(save_path, 'train')))
    if not os.path.exists(os.path.join(save_path_merge, 'train')):
        os.makedirs(os.path.join(save_path_merge, 'train'))

    env = lmdb.open(os.path.join(save_path_merge, 'train', 'BrainSignal.lmdb'), map_size=1 << 42)  # ~4TB max size
    key_list = []
    with env.begin(write=True) as txn:
        for idx_tar in range(len(file_tars)):
            if file_tars[idx_tar].endswith(".tar"):
                with tarfile.open(os.path.join(save_path, 'train', file_tars[idx_tar]), "r") as tar:
                    members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
                    for member in members:
                        with h5py.File(io.BytesIO(tar.extractfile(member).read()), 'r') as f:
                            sample = {"signal": f['signal'][()], "y": f['y'][()]}
                        txn.put(member.name.encode("ascii"), pickle.dumps(sample))
                        key_list.append({"dataset": 'TUEG', "key": member.name})
            print(f'{file_tars[idx_tar]} Done!')
        txn.put('__keys__'.encode("ascii"), pickle.dumps(key_list))

    # delete the intermediate files
    # os.remove(save_path)
