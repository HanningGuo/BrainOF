import os
import h5py
import numpy as np
import nibabel as nib
from typing import Union
from pathlib import Path

offset = {
    "CSI1": np.array([370, 370, 370, 333, 370, 333, 370, 333, 333, 370, 333, 333, 333, 333, 370]),
    "CSI2": np.array([370, 333, 370, 370, 370, 333, 333, 333, 333, 333, 370, 370, 333, 370, 333]),
    "CSI3": np.array([370, 333, 370, 333, 370, 370, 370, 333, 333, 333, 370, 333, 333, 333, 370]),
    "CSI4": np.array([370, 333, 333, 370, 333, 333, 333, 333, 370]),
}


def preparing(subject: tuple, save_path: Union[str, Path]):
    fname, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI beta series data
        fMRI_vertex = nib.load(fname)
        signals = fMRI_vertex.get_fdata()

        # read the label name
        p = Path(fname)
        sub = p.name.split("_")[0]
        with open(os.path.join(p.parent, f"{sub}_imgnames.txt"), "r") as f:
            img_names = [line.strip() for line in f if line.strip()]

        sess = int(p.name.split(".")[0].split("-")[-1])
        cursor = np.sum(offset[sub][:(sess-1)])

        # read the brain mask
        brain_mask = nib.load(os.path.join(p.parent, f"{sub}_brainmask.nii.gz")).get_fdata() > 0
        signals = signals[brain_mask].T

        for idx, signal in enumerate(signals):
            signal = np.expand_dims(np.nan_to_num(signal, nan=0.0), axis=0)
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'

            img_name = img_names[cursor + idx].lower()
            if img_name.startswith("coco_"):
                target = [1, 0, 0]  # COCO
            elif img_name.startswith("n") and img_name.endswith((".jpeg", ".jpg")):
                target = [0, 1, 0]  # ImageNet
            else:
                target = [0, 0, 1]  # Scene/SUN

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=target)
    except:
        with open(os.path.join(save_path, "BOLD5000_error.log"), "a") as f:
            f.write(fname + "\n")


if __name__ == "__main__":
    root = "/path/to/BOLD5000"
    save_path = "/path/to/BrainSignalDataset/BOLD5000"

    # dataset splits
    train_subs = ["CSI1", "CSI2"]
    val_subs = ["CSI3"]
    test_subs = ["CSI4"]

    # prepare the train dataset
    sub_list = []
    for file in os.listdir(root):
        if "GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-" in file:
            sub = file.split("_")[0]
            if sub in train_subs:
                split = 'train'
            elif sub in val_subs:
                split = 'val'
            elif sub in test_subs:
                split = 'test'
            else:
                raise ValueError(f"{sub} is not in the splitting!")
            sub_path = os.path.join(root, file)
            sub_list.append((sub_path, split))

    # prepare the fMRI and save them
    for sub in sub_list:
        preparing(sub, save_path)
