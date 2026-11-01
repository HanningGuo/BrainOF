import os
import h5py
import lmdb
import pickle


# Settings
root = "/path/to/BrainSignalDataset"
save_path = "/path/to/BrainSignalDataset"
# datasets = ["ABIDE1", "ABIDE2", "ABRIM", ..., "WBCIC"]
datasets = ["YOUR DATASETS"]


for dataset in datasets:
    splits = os.listdir(os.path.join(root, dataset))
    for split in splits:
        if not os.path.exists(os.path.join(save_path, dataset, split)):
            os.makedirs(os.path.join(save_path, dataset, split))

        env = lmdb.open(os.path.join(save_path, dataset, split, 'BrainSignal.lmdb'), map_size=1 << 40)  # ~1TB max size
        key_list = []
        with env.begin(write=True) as txn:
            for sample_name in os.listdir(os.path.join(root, dataset, split)):
                if os.path.isfile(os.path.join(root, dataset, split, sample_name)):
                    with h5py.File(os.path.join(root, dataset, split, sample_name), 'r') as f:
                        sample = {"signal": f['signal'][()], "y": f['y'][()]}
                    txn.put(sample_name.encode("ascii"), pickle.dumps(sample))
                    key_list.append({"dataset": dataset, "key": sample_name})
            txn.put('__keys__'.encode("ascii"), pickle.dumps(key_list))
    print(f'{dataset} Done!')
