import numpy as np
from einops import rearrange


class DataTransform:
    def __init__(
            self,
            signal_size: list,
            patch_size: int,
            max_seq_length: int = 2048,
            normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization
        self.signal_size = signal_size
        self.patch_size = patch_size
        self.max_seq_length = max_seq_length

    def __call__(self, signal):
        if self.normalization:
            mean = signal.mean(axis=-1, keepdims=True)
            std = signal.std(axis=-1, keepdims=True)
            signal = (signal - mean) / (std + 1e-8)

        # clip the length and magnitude of signal
        orig_num_region, orig_signal_length = signal.shape
        num_region, signal_length = self.signal_size
        if signal_length < orig_signal_length:
            signal = signal[:, :signal_length]
            orig_signal_length = signal_length

        # make the signal_length an integer multiple of patch size
        padding_size_l = (-orig_signal_length % self.patch_size + 1) // 2  # int
        padding_size_r = (-orig_signal_length % self.patch_size) - padding_size_l
        signal = np.pad(signal, pad_width=((0, 0), (padding_size_l, padding_size_r)), mode='constant', constant_values=0)
        signal = rearrange(signal, 'N (P D) -> (N P) D', D=self.patch_size)
        valid_tokens = len(signal)

        # pad
        signal = np.pad(signal, pad_width=((0, self.max_seq_length - valid_tokens), (0, 0)), mode='constant', constant_values=0)
        mask = np.pad(np.ones(valid_tokens), pad_width=((0, self.max_seq_length - valid_tokens)), mode='constant', constant_values=0)

        return signal, mask, (orig_num_region, padding_size_l, padding_size_r)
