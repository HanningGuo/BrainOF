import torch
import warnings
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from utils import bf16_safe_rfft, bf16_safe_irfft


class TimeFrequencyLoss(nn.Module):
    """
    Time Frequency Loss
    """
    def __init__(self, alpha=0.8, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x_recs, targets, attn_masks, tmp_masks, freq_masks, mods, meta_data):
        rec_masks = attn_masks ^ tmp_masks  # valid tokens that were hidden from the encoder
        loss_tmp, num_tmp, loss_freq = 0, 0, 0
        num_regions, padding_size_ls, padding_size_rs = meta_data
        for x_rec, target, attn_mask, rec_mask, freq_mask, mod, num_region, padding_size_l, padding_size_r in zip(x_recs, targets, attn_masks, rec_masks, freq_masks, mods, num_regions, padding_size_ls, padding_size_rs):
            if padding_size_l != padding_size_r:
                padding_size_l = padding_size_r  # making the length of signal multiple of 2
            meta = (num_region, padding_size_l, padding_size_r)
            signal_rec = _signal_transform(x_rec, attn_mask, meta).to(torch.float32)
            signal_target = _signal_transform(target, attn_mask, meta).to(torch.float32)
            point_mask = _signal_transform(rec_mask.unsqueeze(-1).expand_as(x_rec), attn_mask, meta)

            if mod == 0:
                dim = 0  # spatial-frequency for fMRI
            elif mod == 1 or mod == 2:
                dim = 1  # temporal-frequency for M/EEG
            else:
                dim = 1
                warnings.warn(f"The modality is unknown (not fMRI, EEG and MEG). Please note that default pretraining method is masked temporal-frequency!", UserWarning)

            freq_rec = bf16_safe_rfft(signal_rec, dim=dim)
            freq_target = bf16_safe_rfft(signal_target, dim=dim)

            # temporal loss on the kept bands
            kept = freq_mask.to(freq_rec.dtype)
            tmp_rec = bf16_safe_irfft(freq_rec * kept, dim=dim)
            tmp_target = bf16_safe_irfft(freq_target * kept, dim=dim)
            loss_tmp += F.smooth_l1_loss(tmp_rec[point_mask], tmp_target[point_mask], reduction='sum', beta=self.beta)
            num_tmp += point_mask.sum()

            # frequency loss on the magnitudes of the masked bands
            loss_freq +=  F.smooth_l1_loss(freq_rec[freq_mask.logical_not()].abs(), freq_target[freq_mask.logical_not()].abs(), beta=self.beta)
        loss_tmp = loss_tmp / num_tmp.clamp(min=1)
        loss_freq = loss_freq / x_recs.shape[0]

        loss = (1 - self.alpha) * loss_tmp + self.alpha * loss_freq
        return loss


def _signal_transform(x, attn_mask, meta_data):
    num_region, padding_size_l, padding_size_r = meta_data
    signal = rearrange(x[attn_mask], '(N P) D -> N (P D)', N=num_region)
    signal = signal[:, padding_size_l:-padding_size_r] if padding_size_r != 0 else signal[:, padding_size_l:]
    return signal
