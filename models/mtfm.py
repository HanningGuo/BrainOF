import torch
import warnings
import torch.nn as nn
import scipy.stats as stats
from einops import rearrange
from torch.nn import RMSNorm
from functools import partial
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Callable, List, Optional
from utils import bf16_safe_rfft, bf16_safe_irfft
from .transformer import DintTransformer, ARNESS


def _signal_transform(x, attn_mask, meta_data):
    num_region, padding_size_l, padding_size_r = meta_data
    signal = rearrange(x[attn_mask], '(N P) D -> N (P D)', N=num_region)
    signal = signal[:, padding_size_l:-padding_size_r] if padding_size_r != 0 else signal[:, padding_size_l:]
    return signal


class DintTransformerForMTFM(DintTransformer):
    def __init__(self, mask_ratio_min, mask_ratio_max, mask_ratio_mu, mask_ratio_std, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

    def forward(self, x: torch.Tensor, attn_masks: torch.Tensor, mods: torch.Tensor, meta_data: tuple = None):
        """
        Args:
            x (torch.Tensor): brain signal data of shape B x N x PD
                shape (B, N, PD)
            attn_masks (torch.Tensor): masks for padded signal (i.e., x)
                shape (B, N)
            mods (torch.Tensor): modality index for modality-wise fft
                shape (B, 1)
            meta_data (tuple): data for recovering the signal
        """
        B, N, PD = x.shape

        x, tmp_masks, freq_masks, visible_masks = self.mask_generator(x, attn_masks, mods, meta_data)
        pos_ids = torch.arange(N, device=x.device).expand(B, N)[visible_masks].reshape(B, -1)
        x = x[visible_masks].reshape(B, -1, PD)
        attn_masks = attn_masks[visible_masks].reshape(B, -1)

        # encode the brain signal
        x = self.patch_embed(x)
        # map heterogeneous brain signal to the unified latent space
        x = self.arness(x, attn_masks, pos_ids=pos_ids)

        if self.use_checkpoint and self.checkpoint_blk_ids is None:
            self.checkpoint_blk_ids = [blk_id for blk_id in range(len(self.DintTransformerEncoder)) if blk_id % self.checkpoint_every_n == 0]
        if self.checkpoint_blk_ids:
            self.checkpoint_blk_ids = set(self.checkpoint_blk_ids)
        for layer_id, layer in enumerate(self.DintTransformerEncoder):
            if self.use_checkpoint and layer_id in self.checkpoint_blk_ids:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x, tmp_masks, freq_masks

    def mask_generator(self, orig_x, attn_masks, mods, meta_data):
        tmp_masks = attn_masks.clone()
        padded_visible_masks = attn_masks.clone()
        num_tokens = tmp_masks.sum(dim=-1)
        mask_ratio = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = (num_tokens * mask_ratio).ceil().to(torch.int)
        # number of padded tokens for reconstruction
        num_visible_tokens = num_tokens - num_masked_tokens
        num_padded_tokens = num_visible_tokens.max() - num_visible_tokens

        # generate the random mask for temporal domain
        for i in range(len(num_tokens)):
            mask_idx = torch.randperm(num_tokens[i].int())[:num_masked_tokens[i]]
            tmp_masks[i, mask_idx] = False
            padded_visible_masks[i, mask_idx] = False
            padded_visible_masks[i, num_tokens[i]:num_tokens[i]+num_padded_tokens[i]] = True

        # mask out the frequency
        x, freq_masks = self.masked_frequency(orig_x, attn_masks, mods, meta_data, mask_ratio)
        return x, tmp_masks, freq_masks, padded_visible_masks

    def masked_frequency(self, orig_x, attn_masks, mods, meta_data, mask_ratio):
        masked_x, freq_masks = [], []
        num_regions, padding_size_ls, padding_size_rs = meta_data
        for x, attn_mask, mod, num_region, padding_size_l, padding_size_r in zip(orig_x, attn_masks, mods, num_regions, padding_size_ls, padding_size_rs):
            # recover the normal signal shape
            if padding_size_l != padding_size_r:
                padding_size_l = padding_size_r
            signal = _signal_transform(x, attn_mask, (num_region, padding_size_l, padding_size_r))

            # transform to frequency domain
            if mod == 0:
                dim = 0     # spatial-frequency for fMRI
            elif mod == 1 or mod == 2:
                dim = 1     # temporal-frequency for M/EEG
            else:
                dim = 1
                warnings.warn(f"The modality is unknown (not fMRI, EEG and MEG). Please note that default pretraining method is masked temporal-frequency!", UserWarning)

            signal_freq = bf16_safe_rfft(signal, dim=dim)
            # mask out the frequency
            num_masked_freq = (torch.tensor(signal_freq.shape[dim]) * mask_ratio).ceil().to(torch.int)
            mask_idx = torch.randperm(signal_freq.shape[dim])[:num_masked_freq].to(signal_freq.device)
            freq_mask = torch.ones_like(signal_freq)
            freq_mask = torch.scatter(freq_mask, dim, mask_idx.unsqueeze(1-dim).repeat_interleave(signal_freq.shape[1-dim], 1-dim), 0)
            masked_freq = signal_freq * freq_mask

            # back to temporal domain
            masked_signal = bf16_safe_irfft(masked_freq, dim=dim)
            masked_signal = F.pad(masked_signal, pad=(padding_size_l, padding_size_r), mode="constant", value=0)
            x = rearrange(masked_signal, 'N (P D) -> (N P) D', D=self.patch_size)
            masked_x.append(F.pad(x, pad=(0, 0, 0, (attn_mask == False).sum()), mode="constant", value=0))
            freq_masks.append(freq_mask.bool())
        return torch.stack(masked_x), freq_masks


class MTFM(nn.Module):
    def __init__(
            self,
            mask_ratio_min: float,
            mask_ratio_max: float,
            mask_ratio_mu: float,
            mask_ratio_std: float,
            patch_size: int,
            embed_dim: int,
            num_layers: int,
            num_layers_pr: int,
            num_heads: int,
            num_experts: int,
            num_activated_experts: int,
            num_shared_experts: int,
            num_latents: int = 128,
            num_dense_layer: int = 1,
            num_classes: int = 0,
            drop_path_rate: float = 0.0,
            drop_path_type: str = "progressive",
            norm_layer: Callable = RMSNorm,
            mlp_ratio: int = 4,
            max_seq_length: int = 2048,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            moe_bias: bool = False,
            ffn_dropout_rate: float = 0.0,
            attn_dropout_rate: float = 0.0,
            layer_scale_type: Optional[str] = None,  # from cait; possible values are None, "per_channel", "scalar"
            layer_scale_init_value: float = 1e-4,  # from cait; float
            use_checkpoint: bool = False,
            checkpoint_every_n: int = 1,
            checkpoint_blk_ids: Optional[List[int]] = None,
            weight_init_style: str = "pytorch",  # possible values jax or pytorch
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.patch_size = patch_size
        self.encoder = DintTransformerForMTFM(
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            mask_ratio_mu=mask_ratio_mu,
            mask_ratio_std=mask_ratio_std,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_layers_pr=num_layers_pr,
            num_heads=num_heads,
            num_experts=num_experts,
            num_activated_experts=num_activated_experts,
            num_shared_experts=num_shared_experts,
            num_latents=num_latents,
            num_dense_layer=num_dense_layer,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            drop_path_type=drop_path_type,
            norm_layer=norm_layer,
            mlp_ratio=mlp_ratio,
            max_seq_length=max_seq_length,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            moe_bias=moe_bias,
            ffn_dropout_rate=ffn_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            layer_scale_type=layer_scale_type,
            layer_scale_init_value=layer_scale_init_value,
            use_checkpoint=use_checkpoint,
            checkpoint_every_n=checkpoint_every_n,
            checkpoint_blk_ids=checkpoint_blk_ids,
            weight_init_style=weight_init_style,
        )
        self.decoder = ARNESS(
            embed_dim=embed_dim,
            num_layers=1,
            num_heads=num_heads,
            num_experts=num_experts,
            num_activated_experts=num_activated_experts,
            num_shared_experts=num_shared_experts,
            num_latents=max_seq_length,
            num_dense_layer=0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            moe_bias=moe_bias,
            norm_layer=norm_layer,
            ffn_dropout_rate=ffn_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            drop_path_rate=drop_path_rate,
            drop_path_type=drop_path_type,
            layer_scale_type=layer_scale_type,
            layer_scale_init_value=layer_scale_init_value,
            use_checkpoint=use_checkpoint,
            checkpoint_every_n=checkpoint_every_n,
            checkpoint_blk_ids=checkpoint_blk_ids,
        )
        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_proj = nn.Linear(embed_dim, patch_size, bias=False)

        for m in (self.decoder, self.decoder_norm, self.decoder_proj):
            m.apply(self.encoder._init_weights)

    def forward(self, x: torch.Tensor, attn_masks: torch.Tensor, mods: torch.Tensor, meta_data: tuple):
        """
        Args:
            x (torch.Tensor): brain signal data of shape B x N x PD
                shape (B, N, PD)
            attn_masks (torch.Tensor): masks for padded signal (i.e., x)
                shape (B, N, PD)
            mods (torch.Tensor): modality index for modality token
                shape (B, num_modality + 1)
            meta_data (tuple): metadata about signal
                tuple (orig_num_region, padding_size_l, padding_size_r)
        """
        z, tmp_masks, freq_masks = self.encoder(x, attn_masks, mods, meta_data)
        z_masks = torch.ones((z.shape[0], z.shape[-2]), dtype=z.dtype, device=z.device).bool()
        x_rec = self.decoder(z, z_masks, attn_masks)
        x_rec = self.decoder_proj(self.decoder_norm(x_rec))

        return x_rec, tmp_masks, freq_masks


def brainof_base_50M_A22M(**kwargs):
    model = MTFM(
        patch_size=64, embed_dim=256, num_layers=12, num_layers_pr=4, num_latents=128, num_heads=4,
        num_experts=8, num_activated_experts=2, num_shared_experts=1, num_dense_layer=0,
        norm_layer=partial(RMSNorm, eps=1e-6), mlp_ratio=4, qkv_bias=False, moe_bias=False, **kwargs)
    return model


def brainof_large_331M_A150M(**kwargs):
    model = MTFM(
        patch_size=64, embed_dim=512, num_layers=24, num_layers_pr=4, num_latents=128, num_heads=8,
        num_experts=16, num_activated_experts=4, num_shared_experts=1, num_dense_layer=0,
        norm_layer=partial(RMSNorm, eps=1e-6), mlp_ratio=4, qkv_bias=False, moe_bias=False, **kwargs)
    return model


def brainof_huge_2B_A500M(**kwargs):
    model = MTFM(
        patch_size=64, embed_dim=768, num_layers=36, num_layers_pr=6, num_latents=128, num_heads=12,
        num_experts=28, num_activated_experts=4, num_shared_experts=1, num_dense_layer=0,
        norm_layer=partial(RMSNorm, eps=1e-6), mlp_ratio=4, qkv_bias=False, moe_bias=False, **kwargs)
    return model


def brainof_giant_9B_A1_2B(**kwargs):
    model = MTFM(
        patch_size=64, embed_dim=1024, num_layers=48, num_layers_pr=8, num_latents=128, num_heads=16,
        num_experts=64, num_activated_experts=4, num_shared_experts=1, num_dense_layer=0,
        norm_layer=partial(RMSNorm, eps=1e-6), mlp_ratio=4, qkv_bias=False, moe_bias=False, **kwargs)
    return model
