import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import RMSNorm
from functools import partial
import torch.nn.functional as F
from models import RotaryEmbedding
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from typing import Callable, List, Optional
from timm.layers import DropPath, trunc_normal_
from models.positional_encoding import apply_rotary_emb


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}.")
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def _apply_lora(base_out: torch.Tensor, lora: Optional[LoRALinear], lora_input: torch.Tensor) -> torch.Tensor:
    if lora is None:
        return base_out
    return base_out + lora(lora_input)


class ARNESSAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: Callable = RMSNorm,
            latent_pos_offset: int = 2048,
    ):
        """
        DINT Attention module used in the ARNESS module.

        Args:
            dim (int): The input dimension of the model.
            depth (int): The depth index of current layer.
            num_heads (int, optional): The number of DINT attention heads, half of the vanilla MHA head. Defaults to 8.
            qkv_bias (bool, optional): Whether to include bias in the query, key, and value linear layers. Defaults to False.
            qk_scale (float, optional): Scale factor for query and key. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
            proj_drop (float, optional): Dropout rate for the output of the projection layer. Defaults to 0.0.
            norm_layer (Callable, optional): The head norm in each head independently for improving gradient statistics of DINT attention. Defaults to RMSNorm.
        """
        super().__init__()

        assert dim % (2 * num_heads) == 0, 'dim should be divisible by 2 * num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads // 2
        self.scale = qk_scale or head_dim**-0.5
        self.depth = depth
        self.latent_pos_offset = latent_pos_offset

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.lora_q = None
        self.lora_k = None
        self.lora_v = None
        self.lora_proj = None
        self.rope = RotaryEmbedding(head_dim, freqs_for='lang', theta=10000)
        self.attn_drop = attn_drop
        self.head_norm = norm_layer(2 * head_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lambda_init = self.get_lambda_init()
        self.lambda_qk = nn.Parameter(torch.zeros(4 * head_dim).normal_(mean=0, std=0.1))

    def add_lora(self, add_lora_qkv: bool = True, add_lora_proj: bool = False, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        if add_lora_qkv:
            self.lora_q = LoRALinear(self.to_q.in_features, self.to_q.out_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
            self.lora_k = LoRALinear(self.to_kv.in_features, self.to_q.out_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
            self.lora_v = LoRALinear(self.to_kv.in_features, self.to_q.out_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        if add_lora_proj:
            self.lora_proj = LoRALinear(self.proj.in_features, self.proj.out_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    
    def forward(self, x: torch.Tensor, latents: torch.Tensor, attn_masks: torch.Tensor, latent_masks: torch.Tensor = None, pos_ids: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): signal tokens shape (b, n1, D)
            latents (torch.Tensor): latent features shape (b, n2, D)
            attn_masks (torch.Tensor): attention masks shape (b, n1)
            latent_masks (torch.Tensor): latent masks shape (b, n2). Defaults to None
        Returns:
            unified latent representation
        """
        num_tokens, num_latent = x.shape[-2], latents.shape[-2]
        if pos_ids is None:
            pos_ids = torch.arange(num_tokens, device=x.device).expand(x.shape[0], num_tokens)
        latent_pos = (self.latent_pos_offset + torch.arange(num_latent, device=x.device)).expand(x.shape[0], num_latent)

        q = _apply_lora(self.to_q(latents), self.lora_q, latents)
        kv_input = torch.cat((x, latents), dim=-2)

        if latent_masks is None:
            latent_masks = torch.ones((latents.shape[0], latents.shape[-2]), dtype=latents.dtype, device=latents.device).bool()
        if attn_masks is not None:
            attn_masks = torch.cat((attn_masks, latent_masks), dim=-1)

        b, n1, c = kv_input.shape
        kv = self.to_kv(kv_input)
        if self.lora_k is not None:
            kv[..., :c] = kv[..., :c] + self.lora_k(kv_input)
        if self.lora_v is not None:
            kv[..., c:] = kv[..., c:] + self.lora_v(kv_input)
        k, v = kv.reshape(b, n1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        b, n2, c = q.shape
        q = q.reshape(b, n2, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        q = rearrange(q, 'b h n2 (c1 c2) -> b (h c1) n2 c2', c1=2)
        k = rearrange(k, 'b h n1 (c1 c2) -> b (h c1) n1 c2', c1=2)

        q = self._rotate(q, latent_pos)
        k = self._rotate(k, torch.cat((pos_ids, latent_pos), dim=-1))
        v = torch.repeat_interleave(v, dim=1, repeats=2)

        # Apply attention mask here.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        attn_bias = torch.zeros((q.size(0), 1, q.size(-2), k.size(-2)), dtype=q.dtype, device=q.device)
        if attn_masks is not None:
            attn_masks = (latent_masks.unsqueeze(-1) & attn_masks.unsqueeze(1)).unsqueeze(1)
            attn_bias.masked_fill_(attn_masks.logical_not(), torch.finfo(torch.float16).min)

        lambda_qk = rearrange(self.lambda_qk, '(c1 c2 d) -> c1 c2 d', c1=2, c2=2)
        lambda_q, lambda_k = lambda_qk[0], lambda_qk[1]
        lambda_value = (lambda_q * lambda_k).sum(dim=-1).exp()
        lambda_full = lambda_value[0] - lambda_value[1] + self.lambda_init

        # Flash attention
        x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_bias, scale=self.scale, dropout_p=(self.attn_drop if self.training else 0.0))
        x = rearrange(x, 'b (h t) n1 n2 -> t b h n1 n2', t=2)
        valid_q = latent_masks[:, None, :, None].to(x.dtype)
        x_mean = (x[0] * valid_q).sum(dim=-2, keepdim=True) / valid_q.sum(dim=-2, keepdim=True).clamp(min=1)
        x = lambda_full * x_mean + x[0] - lambda_full * x[1]

        x = self.head_norm(x).transpose(1, 2).reshape(b, n2, c)
        x = _apply_lora(self.proj(x), self.lora_proj, x)
        x = self.proj_drop(x)
        return x

    def _rotate(self, t: torch.Tensor, pos: torch.Tensor):
        angles = pos.float().unsqueeze(-1) * self.rope.freqs_fp32
        angles = repeat(angles, 'b n f -> b 1 n (f r)', r=2)
        return apply_rotary_emb(angles, t)
    
    def get_lambda_init(self):
        return 0.8 - 0.6 * math.exp(-0.3 * self.depth)


class ARNESSBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            num_experts: int,
            num_activated_experts: int,
            num_shared_experts: int,
            mlp_ratio: int = 4,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            moe_bias: bool = False,
            norm_layer: Callable = RMSNorm,
            ffn_dropout_rate: float = 0.0,
            attn_dropout_rate: float = 0.0,
            drop_path: float = 0.0,
            use_dense_layer: bool = False,
            layer_scale_type: Optional[str] = None,
            layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()

        self.norm_1 = norm_layer(dim)
        self.norm_latents = norm_layer(dim)
        self.attn = ARNESSAttention(dim, depth, num_heads, qkv_bias, qk_scale, attn_dropout_rate, ffn_dropout_rate, norm_layer)
        self.ls1 = LayerScale(dim, layer_scale_init_value, layer_scale_type) if layer_scale_type else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(mlp_ratio * dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, drop=ffn_dropout_rate) if use_dense_layer else SparseMoE(dim=dim, hidden_dim=mlp_hidden_dim, num_experts=num_experts, num_activated_experts=num_activated_experts, num_shared_experts=num_shared_experts, dropout_rate=ffn_dropout_rate, moe_bias=moe_bias)
        self.norm_2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, layer_scale_init_value, layer_scale_type) if layer_scale_type else nn.Identity()

    def add_lora(self, add_lora_attn: bool = True, add_lora_ffn: bool = False, add_lora_proj: bool = False, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        if add_lora_attn or add_lora_proj:
            self.attn.add_lora(add_lora_qkv=add_lora_attn, add_lora_proj=add_lora_proj, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        if add_lora_ffn and hasattr(self.ffn, "add_lora"):
            self.ffn.add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: torch.Tensor, latents: torch.Tensor, attn_masks: torch.Tensor, latent_masks: torch.Tensor = None, pos_ids: torch.Tensor = None):
        latents = latents + self.drop_path(self.ls1(self.attn(self.norm_1(x), self.norm_latents(latents), attn_masks, latent_masks, pos_ids)))
        latents = latents + self.drop_path(self.ls2(self.ffn(self.norm_2(latents), latent_masks)))
        return latents


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

    def forward(self, x: torch.Tensor, attn_masks: torch.Tensor, latent_masks: torch.Tensor = None, pos_ids: torch.Tensor = None):
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

        latents = self.latents
        latents = repeat(latents, "NL DL -> B NL DL", B=B)
        if self.use_checkpoint and self.checkpoint_blk_ids is None:
            self.checkpoint_blk_ids = [blk_id for blk_id in range(len(self.ARNESSLayers)) if blk_id % self.checkpoint_every_n == 0]
        if self.checkpoint_blk_ids:
            self.checkpoint_blk_ids = set(self.checkpoint_blk_ids)
        for layer_id, layer in enumerate(self.ARNESSLayers):
            if self.use_checkpoint and layer_id in self.checkpoint_blk_ids:
                latents = checkpoint.checkpoint(layer, x, latents, attn_masks, latent_masks, pos_ids, use_reentrant=False)
            else:
                latents = layer(x, latents, attn_masks, latent_masks, pos_ids)
        return latents


class DintAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: Callable = RMSNorm,
    ):
        """
        DINT Attention (https://arxiv.org/pdf/2501.17486)

        Args:
            dim (int): The input dimension of the model.
            depth (int): The depth index of current layer.
            num_heads (int, optional): The number of DINT attention heads, half of the vanilla MHA head. Defaults to 8.
            qkv_bias (bool, optional): Whether to include bias in the query, key, and value linear layers. Defaults to False.
            qk_scale (float, optional): Scale factor for query and key. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
            proj_drop (float, optional): Dropout rate for the output of the projection layer. Defaults to 0.0.
            norm_layer (Callable, optional): The head norm in each head independently for improving gradient statistics of DINT attention. Defaults to RmsNorm.
        """
        super().__init__()
        assert dim % (2 * num_heads) == 0, 'dim should be divisible by 2 * num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads // 2
        self.scale = qk_scale or head_dim**-0.5
        self.depth = depth

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.lora_q = None
        self.lora_k = None
        self.lora_v = None
        self.lora_proj = None
        self.rope = RotaryEmbedding(head_dim, freqs_for='lang', theta=1000)     # set theta=1000 for short sequence 128
        self.attn_drop = attn_drop
        self.head_norm = norm_layer(2 * head_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lambda_init = self.get_lambda_init()
        self.lambda_qk = nn.Parameter(torch.zeros(4 * head_dim).normal_(mean=0, std=0.1))

    def add_lora(self, add_lora_qkv: bool = True, add_lora_proj: bool = False, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        if add_lora_qkv:
            self.lora_q = LoRALinear(self.qkv.in_features, self.qkv.in_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
            self.lora_k = LoRALinear(self.qkv.in_features, self.qkv.in_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
            self.lora_v = LoRALinear(self.qkv.in_features, self.qkv.in_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        if add_lora_proj:
            self.lora_proj = LoRALinear(self.proj.in_features, self.proj.out_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    def forward(self, x: torch.Tensor, attn_masks: torch.Tensor = None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        if self.lora_q is not None:
            qkv[..., :C] = qkv[..., :C] + self.lora_q(x)
        if self.lora_k is not None:
            qkv[..., C:2 * C] = qkv[..., C:2 * C] + self.lora_k(x)
        if self.lora_v is not None:
            qkv[..., 2 * C:] = qkv[..., 2 * C:] + self.lora_v(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rearrange(q, 'b h n (c1 c2) -> b (h c1) n c2', c1=2)
        k = rearrange(k, 'b h n (c1 c2) -> b (h c1) n c2', c1=2)

        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)
        v = torch.repeat_interleave(v, dim=1, repeats=2)

        attn_bias = torch.zeros((q.size(0), 1, q.size(-2), k.size(-2)), dtype=q.dtype, device=q.device)
        if attn_masks is not None:
            attn_masks = (attn_masks.unsqueeze(-1) & attn_masks.unsqueeze(1)).unsqueeze(1)
            attn_bias.masked_fill_(attn_masks.logical_not(), float("-inf"))

        lambda_qk = rearrange(self.lambda_qk, '(c1 c2 d) -> c1 c2 d', c1=2, c2=2)
        lambda_q, lambda_k = lambda_qk[0], lambda_qk[1]
        lambda_value = (lambda_q * lambda_k).sum(dim=-1).exp()
        lambda_full = lambda_value[0] - lambda_value[1] + self.lambda_init

        # Flash attention
        x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_bias, scale=self.scale, dropout_p=(self.attn_drop if self.training else 0.0))
        x = rearrange(x, 'b (h t) n1 n2 -> t b h n1 n2', t=2)
        x = lambda_full * x[0].mean(dim=-2, keepdim=True) + x[0] - lambda_full * x[1]

        x = self.head_norm(x).transpose(1, 2).reshape(B, N, C)
        x = _apply_lora(self.proj(x), self.lora_proj, x)
        x = self.proj_drop(x)
        return x

    def get_lambda_init(self):
        return 0.8 - 0.6 * math.exp(-0.3 * self.depth)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 32,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1g = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(drop)
        self.lora_w1g = None
        self.lora_w2 = None

    def add_lora(self, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        self.lora_w1g = LoRALinear(self.w1g.in_features, self.w1g.out_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self.lora_w2 = LoRALinear(self.w2.in_features, self.w2.out_features, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    def SwiGLU(self, x: torch.Tensor):
        assert x.shape[-1] % 2 == 0
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    def forward(self, x: torch.Tensor, valid: torch.Tensor = None):
        x = _apply_lora(self.w1g(x), self.lora_w1g, x)
        x = self.SwiGLU(x)
        x = self.drop(x)
        x = _apply_lora(self.w2(x), self.lora_w2, x)
        x = self.drop(x)
        return x


class Router(nn.Module):
    def __init__(
            self,
            dim: int,
            num_experts: int,
            num_activated_experts: int,
            norm_layer: Callable = RMSNorm,
            router_scale: float = 1.0,
            num_expert_groups: int = 1,
            num_limited_groups: int = 1,
            norm_topk_weights: bool = True,
            moe_bias: bool = False,
    ):
        """
        Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of the experts.
            num_activated_experts (int): Number of top experts activated for each input.            router_scale (float): Scaling factor for routing weights.
            router_scale (float, optional): Scaling factor for routing weights. Defaults to 1.0.
            num_expert_groups (int, optional): Number of groups for routing. Defaults to 1.
            num_limited_groups (int, optional): Number of groups to route inputs to. Defaults to 1.
            norm_topk_weights (bool, optional): Whether to normalize expert weights. Defaults to False.
            moe_bias (bool, optional): Whether to include bias in the linear layers. Defaults to False.
        """
        super().__init__()
        self.num_experts = num_experts
        self.topk = num_activated_experts
        self.router_scale = router_scale
        self.num_groups = num_expert_groups
        self.topk_groups = num_limited_groups
        self.norm_topk_weights = norm_topk_weights
        self.moe_bias = moe_bias
        self.gate = nn.Linear(dim, num_experts, bias=moe_bias)
        # self.probs_correction_bias = nn.Parameter(torch.empty(num_experts, dtype=torch.float32))
        self.register_buffer("probs_correction_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True)
        # self.norm = norm_layer(num_experts)
        # stats for updating the bias
        self.register_buffer("counts", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor):
        with torch.autocast(device_type=x.device.type, enabled=False):
            gate_bias = None if self.gate.bias is None else self.gate.bias.float()
            router_logits = F.linear(x.float(), self.gate.weight.float(), gate_bias)
        routing_probs = router_logits.sigmoid()
        original_probs = routing_probs
        routing_probs = routing_probs + self.probs_correction_bias
        if self.num_groups > 1:
            routing_probs = routing_probs.view(x.size(0), self.num_groups, -1)
            if not self.moe_bias:
                group_routing_probs = routing_probs.amax(dim=-1)
            else:
                group_routing_probs = routing_probs.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_routing_probs.topk(self.topk_groups, dim=-1)[1]
            mask = routing_probs.new_ones(x.size(0), self.num_groups, dtype=bool).scatter_(1, indices, False)
            routing_probs = routing_probs.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(routing_probs, self.topk, dim=-1)[1]
        routing_weights = original_probs.gather(1, indices)

        # record counts
        with torch.no_grad():
            counts = torch.bincount(indices.flatten(), minlength=self.num_experts).float()
            self.counts.copy_(counts)

        if self.topk > 1 and self.norm_topk_weights:
            # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = torch.softmax(F.logsigmoid(router_logits).gather(1, indices), dim=-1)
        routing_weights = routing_weights * self.router_scale
        return routing_weights.type_as(x), indices

    @torch.no_grad()
    def update_bias(self, aux_loss_free_alpha: float):
        """Call this once per training step, after optimizer.step()."""
        torch.distributed.all_reduce(self.counts, op=torch.distributed.ReduceOp.SUM)
        avg_counts = self.counts.mean()
        error = avg_counts - self.counts
        self.probs_correction_bias += aux_loss_free_alpha * torch.sign(error)
        # self.probs_correction_bias += aux_loss_free_alpha * self.norm(error)
        self.probs_correction_bias -= self.probs_correction_bias.mean()  # only the relative bias matters for top-k


class SparseMoE(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            num_experts: int,
            num_activated_experts: int,
            num_shared_experts: int,
            norm_layer: Callable = RMSNorm,
            dropout_rate: float = 0.0,
            router_scale: float = 1.0,
            num_expert_groups: int = 1,
            num_limited_groups: int = 1,
            norm_topk_weights: bool = True,
            moe_bias: bool = False,
    ):
        """
        Sparse Mixture-of-Experts (MoE) module with shared experts.

        Attributes:
            dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of intermediate features.
            num_experts (int): Total number of experts in the model.
            num_activated_experts (int): Number of experts activated for each input.
            num_shared_experts (int): Shared experts applied to all inputs.
            dropout_rate (float, optinal): Dropout rate. Defaults to 0.0.
            router_scale (float, optional): Scaling factor for routing weights. Defaults to 1.0.
            num_expert_groups (int, optional): Number of groups for routing. Defaults to 1.
            num_limited_groups (int, optional): Number of groups to route inputs to. Defaults to 1.
            norm_topk_weights (bool, optional): Whether to normalize expert weights. Defaults to False.
            moe_bias (bool, optional): Whether to include bias in the linear layers. Defaults to False.
        """
        super().__init__()
        self.num_experts = num_experts
        self.num_activated_experts = num_activated_experts
        self.router = Router(dim=dim, num_experts=num_experts, num_activated_experts=num_activated_experts, norm_layer=norm_layer, router_scale=router_scale, num_expert_groups=num_expert_groups, num_limited_groups=num_limited_groups, norm_topk_weights=norm_topk_weights, moe_bias=moe_bias)
        self.experts = nn.ModuleList([FeedForward(dim=dim, hidden_dim=hidden_dim // num_activated_experts, drop=dropout_rate) for _ in range(self.num_experts)])
        self.shared_experts = FeedForward(dim=dim, hidden_dim=num_shared_experts * hidden_dim, drop=dropout_rate)

    def add_lora(self, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0, expert_indices: Optional[List[int]] = None):
        if hasattr(self.shared_experts, "add_lora"):
            self.shared_experts.add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        indices = range(self.num_experts) if expert_indices is None else expert_indices
        for i in indices:
            if hasattr(self.experts[i], "add_lora"):
                self.experts[i].add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.size()
        x = x.reshape(-1, D)
        if valid is not None:                      # route / compute only real tokens; padded rows get 0
            keep = valid.reshape(-1)
            out = x.new_zeros(x.shape)
            out[keep] = self._moe(x[keep])
            return out.view(B, N, D)
        return self._moe(x).view(B, N, D)

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.experts[0]
        bf16 = e0.w1g.weight.dtype == torch.bfloat16 or (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16)
        if x.is_cuda and bf16 and (e0.drop.p == 0 or not self.training) and all(e.lora_w1g is None for e in self.experts):
            return self._moe_grouped(x)
        routing_weights, indices = self.router(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.num_experts).tolist()
        for i in range(self.num_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * routing_weights[idx, top, None]
        z = self.shared_experts(x)
        return y + z
    
    def _moe_grouped(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights, indices = self.router(x)
        flat = indices.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.num_experts)
        active = [i for i, c in enumerate(counts.tolist()) if c > 0]   # skip empty experts so their grad stays None, as in the loop
        if active:
            offs = torch.cumsum(counts[active], 0).to(torch.int32)
            w1g = torch.stack([self.experts[i].w1g.weight for i in active]).to(torch.bfloat16)
            w2 = torch.stack([self.experts[i].w2.weight for i in active]).to(torch.bfloat16)
            h = F.grouped_mm(x[order // indices.size(1)].to(torch.bfloat16), w1g.transpose(-2, -1), offs=offs)
            h = F.grouped_mm(self.experts[0].SwiGLU(h), w2.transpose(-2, -1), offs=offs)
            h = h[torch.argsort(order)].view(*indices.shape, -1)
            y = (h * routing_weights.unsqueeze(-1)).sum(1).to(x.dtype)
        else:
            y = torch.zeros_like(x)
        z = self.shared_experts(x)
        return y + z


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-4,
            layer_scale_type: str = 'per_channel',
            inplace: bool = False,
    ):
        super().__init__()
        self.inplace = inplace

        assert layer_scale_type in [
            "per_channel",
            "scalar",
        ], f"Found Layer scale type {self.layer_scale_type}"
        if layer_scale_type == "per_channel":
            # one gamma value per channel
            gamma_shape = [1, 1, dim]
        elif layer_scale_type == "scalar":
            # single gamma value for all channels
            gamma_shape = [1, 1, 1]
        # two gammas: for each part of the fwd in the encoder
        self.gamma = nn.Parameter(
            torch.ones(size=gamma_shape) * init_values,
            requires_grad=True,
        )

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DintTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        num_experts: int,
        num_activated_experts: int,
        num_shared_experts: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        moe_bias: bool = False,
        norm_layer: Callable = RMSNorm,
        ffn_dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        use_dense_layer: bool = False,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()

        self.norm_1 = norm_layer(dim)
        self.attn = DintAttention(dim, depth, num_heads, qkv_bias, qk_scale, attn_dropout_rate, ffn_dropout_rate, norm_layer)
        self.ls1 = LayerScale(dim, layer_scale_init_value, layer_scale_type) if layer_scale_type else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(mlp_ratio * dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, drop=ffn_dropout_rate) if use_dense_layer else SparseMoE(dim=dim, hidden_dim=mlp_hidden_dim, num_experts=num_experts, num_activated_experts=num_activated_experts, num_shared_experts=num_shared_experts, norm_layer=norm_layer, dropout_rate=ffn_dropout_rate, moe_bias=moe_bias)
        self.norm_2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, layer_scale_init_value, layer_scale_type) if layer_scale_type else nn.Identity()

    def add_lora(self, add_lora_attn: bool = True, add_lora_ffn: bool = False, add_lora_proj: bool = False, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        if add_lora_attn or add_lora_proj:
            self.attn.add_lora(add_lora_qkv=add_lora_attn, add_lora_proj=add_lora_proj, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        if add_lora_ffn and hasattr(self.ffn, "add_lora"):
            self.ffn.add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: torch.Tensor, attn_masks: torch.Tensor = None):
        x = x + self.drop_path(self.ls1(self.attn(self.norm_1(x), attn_masks)))
        x = x + self.drop_path(self.ls2(self.ffn(self.norm_2(x))))
        return x


class TemporalConv(nn.Module):
    """
    Brain Signal to Patch Embedding
    """
    def __init__(self, in_chans, hidden_chans, out_chans, patch_size, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, hidden_chans, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.gelu1 = nn.GELU()
        self.norm1 = RMSNorm(patch_size)
        self.drop1 = nn.Dropout(drop)

        self.conv2 = nn.Conv2d(hidden_chans, hidden_chans, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = RMSNorm(patch_size)
        self.drop2 = nn.Dropout(drop)

        self.conv3 = nn.Conv2d(hidden_chans, out_chans, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.norm3 = RMSNorm(patch_size)
        self.gelu3 = nn.GELU()
        self.drop3 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): patched brain signal features
                shape (B, N, D)
        Returns:
            shape (B, N, D) where N is the max sequence length
        """
        B, N, D = x.shape
        x = x.unsqueeze(1)
        x = self.drop1(self.gelu1(self.norm1(self.conv1(x))))
        x = self.drop2(self.gelu2(self.norm2(self.conv2(x))))
        x = self.drop3(self.gelu3(self.norm3(self.conv3(x))))
        x = rearrange(x, 'B C N D -> B N (D C)')
        return x


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
        self.patch_embed = TemporalConv(in_chans=1, hidden_chans=int(embed_dim/patch_size), out_chans=int(embed_dim/patch_size), patch_size=patch_size, drop=ffn_dropout_rate)
        self.arness = ARNESS(
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
        if add_lora_arness and hasattr(self.arness, "add_lora"):
            self.arness.add_lora(add_lora_attn=add_lora_attn, add_lora_ffn=add_lora_ffn, add_lora_proj=add_lora_proj, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        if add_lora_encoder:
            layers = self.DintTransformerEncoder if encoder_layers is None else [
                self.DintTransformerEncoder[i] for i in encoder_layers
            ]
            for layer in layers:
                if hasattr(layer, "add_lora"):
                    layer.add_lora(add_lora_attn=add_lora_attn, add_lora_ffn=add_lora_ffn, add_lora_proj=add_lora_proj, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: torch.Tensor, attn_masks: torch.Tensor):
        """
        Args:
            x (torch.Tensor): brain signal data of shape B x N x PD
                shape (B, N, PD)
            attn_masks (torch.Tensor): masks for padded signal (i.e., x)
                shape (B, N)
        Returns:
            shape (B, N_Classes) where N_Classes is the number of classes
        """
        B, N, PD = x.shape
        # encode the brain signal
        x = self.patch_embed(x)
        # map heterogeneous brain signal to the unified latent space
        x = self.arness(x, attn_masks)

        if self.use_checkpoint and self.checkpoint_blk_ids is None:
            self.checkpoint_blk_ids = [blk_id for blk_id in range(len(self.DintTransformerEncoder)) if blk_id % self.checkpoint_every_n == 0]
        if self.checkpoint_blk_ids:
            self.checkpoint_blk_ids = set(self.checkpoint_blk_ids)
        for layer_id, layer in enumerate(self.DintTransformerEncoder):
            if self.use_checkpoint and layer_id in self.checkpoint_blk_ids:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm(x)    # B NP D
        x = self.avgpool(x.transpose(1, 2))  # B D 1
        x = self.head(x.flatten(1))
        return x

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


def brainof_giant_9B_A1_2B(**kwargs):
    model = DintTransformer(
        patch_size=64, embed_dim=1024, num_layers=48, num_layers_pr=8, num_latents=128, num_heads=16,
        num_experts=64, num_activated_experts=4, num_shared_experts=1, num_dense_layer=0,
        norm_layer=partial(RMSNorm, eps=1e-6), mlp_ratio=4, qkv_bias=False, moe_bias=False, **kwargs)
    return model