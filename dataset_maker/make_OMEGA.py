import os
import mne
import h5py
from typing import Union
from pathlib import Path


ch_names = [
    'MLC11-4409', 'MLC12-4409', 'MLC13-4409', 'MLC14-4409', 'MLC15-4409', 'MLC16-4409',
    'MLC17-4409', 'MLC21-4409', 'MLC22-4409', 'MLC23-4409', 'MLC24-4409', 'MLC25-4409',
    'MLC31-4409', 'MLC32-4409', 'MLC41-4409', 'MLC42-4409', 'MLC51-4409', 'MLC52-4409',
    'MLC53-4409', 'MLC54-4409', 'MLC55-4409', 'MLC61-4409', 'MLC62-4409', 'MLC63-4409',
    'MLF11-4409', 'MLF12-4409', 'MLF13-4409', 'MLF14-4409', 'MLF21-4409', 'MLF22-4409',
    'MLF23-4409', 'MLF24-4409', 'MLF25-4409', 'MLF31-4409', 'MLF32-4409', 'MLF33-4409',
    'MLF34-4409', 'MLF35-4409', 'MLF41-4409', 'MLF42-4409', 'MLF43-4409', 'MLF44-4409',
    'MLF45-4409', 'MLF46-4409', 'MLF51-4409', 'MLF52-4409', 'MLF53-4409', 'MLF54-4409',
    'MLF55-4409', 'MLF56-4409', 'MLF61-4409', 'MLF62-4409', 'MLF63-4409', 'MLF64-4409',
    'MLF65-4409', 'MLF66-4409', 'MLF67-4409', 'MLO11-4409', 'MLO12-4409', 'MLO13-4409',
    'MLO14-4409', 'MLO21-4409', 'MLO22-4409', 'MLO23-4409', 'MLO24-4409', 'MLO31-4409',
    'MLO32-4409', 'MLO33-4409', 'MLO34-4409', 'MLO41-4409', 'MLO42-4409', 'MLO43-4409',
    'MLO44-4409', 'MLO51-4409', 'MLO52-4409', 'MLO53-4409', 'MLP11-4409', 'MLP12-4409',
    'MLP21-4409', 'MLP22-4409', 'MLP23-4409', 'MLP31-4409', 'MLP32-4409', 'MLP33-4409',
    'MLP34-4409', 'MLP35-4409', 'MLP41-4409', 'MLP42-4409', 'MLP43-4409', 'MLP44-4409',
    'MLP45-4409', 'MLP51-4409', 'MLP52-4409', 'MLP53-4409', 'MLP54-4409', 'MLP55-4409',
    'MLP56-4409', 'MLP57-4409', 'MLT11-4409', 'MLT12-4409', 'MLT13-4409', 'MLT14-4409',
    'MLT15-4409', 'MLT16-4409', 'MLT21-4409', 'MLT22-4409', 'MLT23-4409', 'MLT24-4409',
    'MLT25-4409', 'MLT26-4409', 'MLT27-4409', 'MLT31-4409', 'MLT32-4409', 'MLT33-4409',
    'MLT34-4409', 'MLT35-4409', 'MLT36-4409', 'MLT37-4409', 'MLT41-4409', 'MLT42-4409',
    'MLT43-4409', 'MLT44-4409', 'MLT45-4409', 'MLT46-4409', 'MLT47-4409', 'MLT51-4409',
    'MLT52-4409', 'MLT53-4409', 'MLT54-4409', 'MLT55-4409', 'MLT56-4409', 'MLT57-4409',
    'MRC11-4409', 'MRC12-4409', 'MRC13-4409', 'MRC14-4409', 'MRC15-4409', 'MRC16-4409',
    'MRC17-4409', 'MRC21-4409', 'MRC22-4409', 'MRC23-4409', 'MRC24-4409', 'MRC25-4409',
    'MRC31-4409', 'MRC32-4409', 'MRC41-4409', 'MRC42-4409', 'MRC51-4409', 'MRC52-4409',
    'MRC53-4409', 'MRC54-4409', 'MRC55-4409', 'MRC61-4409', 'MRC62-4409', 'MRC63-4409',
    'MRF11-4409', 'MRF12-4409', 'MRF13-4409', 'MRF14-4409', 'MRF21-4409', 'MRF22-4409',
    'MRF23-4409', 'MRF24-4409', 'MRF25-4409', 'MRF31-4409', 'MRF32-4409', 'MRF33-4409',
    'MRF34-4409', 'MRF35-4409', 'MRF41-4409', 'MRF42-4409', 'MRF43-4409', 'MRF44-4409',
    'MRF45-4409', 'MRF46-4409', 'MRF51-4409', 'MRF52-4409', 'MRF53-4409', 'MRF54-4409',
    'MRF55-4409', 'MRF56-4409', 'MRF61-4409', 'MRF62-4409', 'MRF63-4409', 'MRF64-4409',
    'MRF65-4409', 'MRF66-4409', 'MRF67-4409', 'MRO11-4409', 'MRO12-4409', 'MRO13-4409',
    'MRO14-4409', 'MRO21-4409', 'MRO22-4409', 'MRO23-4409', 'MRO24-4409', 'MRO31-4409',
    'MRO32-4409', 'MRO33-4409', 'MRO34-4409', 'MRO41-4409', 'MRO42-4409', 'MRO43-4409',
    'MRO44-4409', 'MRO51-4409', 'MRO52-4409', 'MRO53-4409', 'MRP11-4409', 'MRP12-4409',
    'MRP21-4409', 'MRP22-4409', 'MRP23-4409', 'MRP31-4409', 'MRP32-4409', 'MRP33-4409',
    'MRP34-4409', 'MRP35-4409', 'MRP41-4409', 'MRP42-4409', 'MRP43-4409', 'MRP44-4409',
    'MRP45-4409', 'MRP51-4409', 'MRP52-4409', 'MRP53-4409', 'MRP54-4409', 'MRP55-4409',
    'MRP56-4409', 'MRP57-4409', 'MRT11-4409', 'MRT12-4409', 'MRT13-4409', 'MRT14-4409',
    'MRT15-4409', 'MRT16-4409', 'MRT21-4409', 'MRT22-4409', 'MRT23-4409', 'MRT24-4409',
    'MRT25-4409', 'MRT26-4409', 'MRT27-4409', 'MRT31-4409', 'MRT32-4409', 'MRT33-4409',
    'MRT34-4409', 'MRT35-4409', 'MRT36-4409', 'MRT37-4409', 'MRT41-4409', 'MRT42-4409',
    'MRT43-4409', 'MRT44-4409', 'MRT45-4409', 'MRT46-4409', 'MRT47-4409', 'MRT51-4409',
    'MRT52-4409', 'MRT53-4409', 'MRT54-4409', 'MRT55-4409', 'MRT56-4409', 'MRT57-4409',
    'MZC01-4409', 'MZC02-4409', 'MZC03-4409', 'MZC04-4409', 'MZF01-4409', 'MZF02-4409',
    'MZF03-4409', 'MZO01-4409', 'MZO02-4409', 'MZO03-4409', 'MZP01-4409'
]


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw meg data
        meg_raw = mne.io.read_raw_ctf(fname, preload=preload)
        # pick common channels between whole channel and current subject, and reorder those channel with specific order
        chs = [ch.split('-')[0] for ch in ch_names]
        reorder_chs = []
        for ch in meg_raw.ch_names:
            if ch.split('-')[0] in chs:
                reorder_chs.append(ch)
        meg_raw.reorder_channels(reorder_chs)
        # FIR bandpass and notch filter for rest state
        meg_raw.filter(l_freq=1, h_freq=80, n_jobs=n_jobs)
        meg_raw.notch_filter(60.0, n_jobs=n_jobs)  # Canada power line
        # resample brain signal to 200 HZ
        meg_raw.resample(200, n_jobs=n_jobs)
        # discarding the first 60s and the last 60s time points in order to obtain the stationary signals
        meg_raw.crop(tmin=60, tmax=meg_raw.n_times / meg_raw.info['sfreq'] - 60)
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
        with open(os.path.join(save_path, "OMEGA_error.log"), "a") as f:
            f.write(str(fname) + "\n")


if __name__ == "__main__":
    root = "/path/to/OMEGA"
    save_path = "/path/to/BrainSignalDataset/OMEGA"

    sub_list = []
    for sub in os.listdir(root):
        if sub.startswith('sub-'):
            for ses in os.listdir(os.path.join(root, sub)):
                if os.path.exists(os.path.join(root, sub, ses, 'meg')):
                    for file in os.listdir(os.path.join(root, sub, ses, 'meg')):
                        if file.endswith('.ds') and 'rest' in file:
                            sub_path = os.path.join(root, sub, ses, 'meg', file)
                            if os.path.exists(sub_path):
                                sub_list.append((sub_path, 'train'))

    # preprocess the rest meg and save them
    for sub in sub_list:
        preprocessing(sub, save_path)
