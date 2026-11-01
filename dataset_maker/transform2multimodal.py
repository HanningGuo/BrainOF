import os
import lmdb
import pickle
import numpy as np

# Settings
root = "/path/to/BrainSignalDataset"
save_path = "/path/to/BrainSignalDataset"


# Note: Modality order matters. The predominant modality should be the last element!
unimodal_datasets = ["CamCAN_fMRI_Rest", "CamCAN_MEG_Rest"]
# unimodal_datasets = ["LEMON_fMRI", "LEMON_EEG"]
multimodal_datasets = "CamCAN_Rest"
# multimodal_datasets = "LEMON"


for split in ['train', 'val', 'test']:
    if not os.path.exists(os.path.join(save_path, multimodal_datasets, split)):
        os.makedirs(os.path.join(save_path, multimodal_datasets, split))
    multimodal_env = lmdb.open(os.path.join(save_path, multimodal_datasets, split, 'BrainSignal.lmdb'), map_size=1 << 40)  # ~1TB max size

    # prepare the first dataset
    path1 = os.path.join(root, unimodal_datasets[0], split, 'BrainSignal.lmdb')
    env1 = lmdb.open(path1, readonly=True, lock=False, readahead=False, meminit=False)
    with env1.begin(write=False) as txn1:
        keys1 = pickle.loads(txn1.get('__keys__'.encode("ascii")))
    # prepare the second dataset
    path2 = os.path.join(root, unimodal_datasets[1], split, 'BrainSignal.lmdb')
    env2 = lmdb.open(path2, readonly=True, lock=False, readahead=False, meminit=False)
    with env2.begin(write=False) as txn2:
        keys2 = pickle.loads(txn2.get('__keys__'.encode("ascii")))

    # find the brain signal pair
    key_list = []
    with multimodal_env.begin(write=True) as multimodal_txn:
        for key1 in keys1:
            # CamCAN
            sub1 = key1['key'].split('_')[1].split('-')[1]
            indices = [idx for idx, key2 in enumerate(keys2) if key2['key'].split('_')[1].split('-')[0] == sub1]
            # LEMON
            # sub1 = key1['key'].split('_')[0]
            # indices = [idx for idx, key2 in enumerate(keys2) if key2['key'].split('_')[0] == sub1]

            # exist matched pair
            if indices:
                paired_idx = np.random.choice(indices)
                with env1.begin(write=False) as unimodal_txn1:
                    sample1 = pickle.loads(unimodal_txn1.get(key1['key'].encode("ascii")))
                    signal1 = sample1['signal']
                    target1 = sample1['y']
                with env2.begin(write=False) as unimodal_txn2:
                    sample2 = pickle.loads(unimodal_txn2.get(keys2[paired_idx]['key'].encode("ascii")))
                    signal2 = sample2['signal']
                    target2 = sample2['y']

                paired_sample = {"signal": (signal1, signal2), "y": target1}
                sample_name = key1['key'].split('.')[0] + ' & ' + keys2[paired_idx]['key'].split('.')[0]
                multimodal_txn.put(sample_name.encode("ascii"), pickle.dumps(paired_sample))
                key_list.append({"dataset": multimodal_datasets, "key": sample_name})
        multimodal_txn.put('__keys__'.encode("ascii"), pickle.dumps(key_list))

print(f'{multimodal_datasets} Done!')
