import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import RMSNorm
from functools import partial
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from typing import Callable, List, Optional
from timm.layers import DropPath, trunc_normal_
from models.transformer import LoRALinear, ARNESSAttention, ARNESSBlock, DintAttention, FeedForward, Router, SparseMoE, LayerScale, DintTransformerBlock, TemporalConv


class ARNESS(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_layers: int,
            num_heads: int,
            num_experts: int,
            num_activated_experts: int,
            num_shared_experts: int,
            num_latents: int = 128,
            num_dense_layer: int = 1,
            mlp_ratio: int = 4,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            moe_bias: bool = False,
            norm_layer: Callable = RMSNorm,
            ffn_dropout_rate: float = 0.0,
            attn_dropout_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            drop_path_type: str = "progressive",
            layer_scale_type: Optional[str] = None,
            layer_scale_init_value: float = 1e-4,
            use_checkpoint: bool = False,
            checkpoint_every_n: int = 1,
            checkpoint_blk_ids: Optional[List[int]] = None,
    ):
        """
        Any-Resolution Neural Signal Sampler (ARNESS) which takes in time series features and outputs patched tokens.

        Args:
            embed_dim (int): dimension of the hidden features.
            num_layers (int): number of layers.
            num_heads (int): number of heads.
            num_experts (int): total number of experts in the model.
            num_activated_experts (int): number of experts activated for each input.
            num_shared_experts (int): shared experts applied to all inputs.
            num_latents (int, optional): number of latent tokens to use in the ARNESS. Defaults to 128.
            num_dense_layer (int, optional): number of the dense layer
            mlp_ratio (int, optional): mlp ratio. Defaults to 4.
            qkv_bias (bool, optional): whether to include bias in the query, key, and value linear layers. Defaults to False.
            qk_scale (float, optional): scale factor for query and key. Defaults to None.
            moe_bias (bool, optional): whether to include bias in the linear layers. Defaults to False.
            norm_layer (Callable, optional): norm layer. Defaults to RmsNorm.
            ffn_dropout_rate (float, optional): dropout rate for ffn. Defaults to 0.0.
            attn_dropout_rate (float, optional): dropout rate for attention weights. Defaults to 0.0.
            drop_path_rate (float, optional): drop path rate for the layer. Defaults to 0.0.
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.checkpoint_every_n = checkpoint_every_n
        self.checkpoint_blk_ids = checkpoint_blk_ids
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_layers)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

        self.ARNESSLayers = nn.ModuleList(
            [
                ARNESSBlock(
                    dim=embed_dim,
                    depth=i,
                    num_heads=num_heads,
                    num_experts=num_experts,
                    num_activated_experts=num_activated_experts,
                    num_shared_experts=num_shared_experts,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    moe_bias=moe_bias,
                    norm_layer=norm_layer,
                    ffn_dropout_rate=ffn_dropout_rate,
                    attn_dropout_rate=attn_dropout_rate,
                    drop_path=dpr[i],
                    use_dense_layer=i < num_dense_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_layers)
            ]
        )

    def add_lora(self, add_lora_attn: bool = True, add_lora_ffn: bool = False, add_lora_proj: bool = False, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        for layer in self.ARNESSLayers:
            if hasattr(layer, "add_lora"):
                layer.add_lora(add_lora_attn=add_lora_attn, add_lora_ffn=add_lora_ffn, add_lora_proj=add_lora_proj, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: torch.Tensor, attn_masks: torch.Tensor, latents:torch.Tensor = None, latent_masks: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): brain signal features
                shape (B, N, PD)
            attn_masks (torch.Tensor): attention masks for signal patches (i.e., x)
                shape (B, N)
            latent_masks (torch.Tensor): latent masks for latent patches
                shape (B, NL) Defaults to None
        Returns:
            shape (B, NL, D) where NL is self.num_latents
        """
        B, NP, D = x.shape

        if latents is None:
            latents = self.latents
            latents = repeat(latents, "NL DL -> B NL DL", B=B)
        if self.use_checkpoint and self.checkpoint_blk_ids is None:
            self.checkpoint_blk_ids = [blk_id for blk_id in range(len(self.ARNESSLayers)) if blk_id % self.checkpoint_every_n == 0]
        if self.checkpoint_blk_ids:
            self.checkpoint_blk_ids = set(self.checkpoint_blk_ids)
        for layer_id, layer in enumerate(self.ARNESSLayers):
            if self.use_checkpoint and layer_id in self.checkpoint_blk_ids:
                latents = checkpoint.checkpoint(layer, x, latents, attn_masks, latent_masks, use_reentrant=False)
            else:
                latents = layer(x, latents, attn_masks, latent_masks)
        return latents


class DintTransformer(nn.Module):
    def __init__(
        self,
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
        num_classes: int = 2,
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
        classifier_head_init_scale: float = 0.001
    ):
        """
        DINT Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the shape of brain signal except that it is a 2D Tensor
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.checkpoint_every_n = checkpoint_every_n
        self.checkpoint_blk_ids = checkpoint_blk_ids
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        assert embed_dim % patch_size == 0
        self.patch_embed1 = TemporalConv(in_chans=1, hidden_chans=int(embed_dim/patch_size), out_chans=int(embed_dim/patch_size), patch_size=patch_size, drop=ffn_dropout_rate)
        self.patch_embed2 = TemporalConv(in_chans=1, hidden_chans=int(embed_dim/patch_size), out_chans=int(embed_dim/patch_size), patch_size=patch_size, drop=ffn_dropout_rate)
        self.arness1 = ARNESS(
            embed_dim=embed_dim,
            num_layers=num_layers_pr,
            num_heads=num_heads,
            num_experts=num_experts,
            num_activated_experts=num_activated_experts,
            num_shared_experts=num_shared_experts,
            num_latents=num_latents,
            num_dense_layer=num_dense_layer,
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
        )
        self.arness2 = ARNESS(
            embed_dim=embed_dim,
            num_layers=num_layers_pr,
            num_heads=num_heads,
            num_experts=num_experts,
            num_activated_experts=num_activated_experts,
            num_shared_experts=num_shared_experts,
            num_latents=num_latents,
            num_dense_layer=num_dense_layer,
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
        )

        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_layers)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")
        self.DintTransformerEncoder = nn.ModuleList(
            [
                DintTransformerBlock(
                    dim=embed_dim,
                    depth=i,
                    num_heads=num_heads,
                    num_experts=num_experts,
                    num_activated_experts=num_activated_experts,
                    num_shared_experts=num_shared_experts,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    moe_bias=moe_bias,
                    norm_layer=norm_layer,
                    ffn_dropout_rate=ffn_dropout_rate,
                    attn_dropout_rate=attn_dropout_rate,
                    drop_path=dpr[i],
                    use_dense_layer=i < num_dense_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(classifier_head_init_scale)
            self.head.bias.data.mul_(classifier_head_init_scale)

    def add_lora(self, add_lora_arness=True, add_lora_encoder=True, add_lora_attn=True, add_lora_ffn=False, add_lora_proj=False, lora_rank=8, lora_alpha=16, lora_dropout=0.0, encoder_layers=None):
        if add_lora_arness and hasattr(self.arness1, "add_lora") and hasattr(self.arness2, "add_lora"):
            self.arness1.add_lora(add_lora_attn=add_lora_attn, add_lora_ffn=add_lora_ffn, add_lora_proj=add_lora_proj, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.arness2.add_lora(add_lora_attn=add_lora_attn, add_lora_ffn=add_lora_ffn, add_lora_proj=add_lora_proj, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        if add_lora_encoder:
            layers = self.DintTransformerEncoder if encoder_layers is None else [
                self.DintTransformerEncoder[i] for i in encoder_layers
            ]
            for layer in layers:
                if hasattr(layer, "add_lora"):
                    layer.add_lora(add_lora_attn=add_lora_attn, add_lora_ffn=add_lora_ffn, add_lora_proj=add_lora_proj, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: tuple, attn_masks: tuple):
        """
        Args:
            x (tuple): brain signal data of shape B x N x PD
                shape (B, N, PD)
            attn_masks (tuple): masks for padded signal (i.e., x)
                shape (B, N)
        Returns:
            shape (B, N_Classes) where N_Classes is the number of classes
        """
        x1, x2 = x
        attn_masks1, attn_masks2 = attn_masks
        # encode the brain signal
        x1 = self.patch_embed1(x1)
        x2 = self.patch_embed2(x2)
        # map heterogeneous brain signal to the unified latent space
        latents = self.arness1(x1, attn_masks1)
        # fuse modality 2 to the latent space
        fused_x = self.arness2(x2, attn_masks2, latents)
        

        if self.use_checkpoint and self.checkpoint_blk_ids is None:
            self.checkpoint_blk_ids = [blk_id for blk_id in range(len(self.DintTransformerEncoder)) if blk_id % self.checkpoint_every_n == 0]
        if self.checkpoint_blk_ids:
            self.checkpoint_blk_ids = set(self.checkpoint_blk_ids)
        for layer_id, layer in enumerate(self.DintTransformerEncoder):
            if self.use_checkpoint and layer_id in self.checkpoint_blk_ids:
                fused_x = checkpoint.checkpoint(layer, fused_x, use_reentrant=False)
            else:
                fused_x = layer(fused_x)

        fused_x = self.norm(fused_x)    # B NP D
        fused_x = self.avgpool(fused_x.transpose(1, 2))  # B D 1
        fused_x = self.head(fused_x.flatten(1))
        return fused_x

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.DintTransformerEncoder):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if type(layer.ffn).__name__ == 'SparseMoE':
                for expert in layer.ffn.experts:
                    rescale(expert.w2.weight.data, layer_id + 1)
                rescale(layer.ffn.shared_experts.w2.weight.data, layer_id + 1)
            else:
                rescale(layer.ffn.w2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.weight_init_style == "jax":
                # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init_style == "pytorch":
                # PyTorch ViT uses trunc_normal_
                trunc_normal_(m.weight, std=0.02)
            elif self.weight_init_style == "allzero":
                # PyTorch ViT uses trunc_normal_
                torch.nn.init.constant_(m.weight, 0)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, RMSNorm):
            nn.init.constant_(m.weight, 1.0)


def brainof_base_50M_A22M(**kwargs):
    model = DintTransformer(
        patch_size=64, embed_dim=256, num_layers=12, num_layers_pr=4, num_latents=128, num_heads=4,
        num_experts=8, num_activated_experts=2, num_shared_experts=1, num_dense_layer=0,
        norm_layer=partial(RMSNorm, eps=1e-6), mlp_ratio=4, qkv_bias=False, moe_bias=False, **kwargs)
    return model


def brainof_large_331M_A150M(**kwargs):
    model = DintTransformer(
        patch_size=64, embed_dim=512, num_layers=24, num_layers_pr=4, num_latents=128, num_heads=8,
        num_experts=16, num_activated_experts=4, num_shared_experts=1, num_dense_layer=0,
        norm_layer=partial(RMSNorm, eps=1e-6), mlp_ratio=4, qkv_bias=False, moe_bias=False, **kwargs)
    return model


def brainof_huge_2B_A500M(**kwargs):
    model = DintTransformer(
        patch_size=64, embed_dim=768, num_layers=36, num_layers_pr=6, num_latents=128, num_heads=12,
        num_experts=28, num_activated_experts=4, num_shared_experts=1, num_dense_layer=0,
        norm_layer=partial(RMSNorm, eps=1e-6), mlp_ratio=4, qkv_bias=False, moe_bias=False, **kwargs)
    return model
