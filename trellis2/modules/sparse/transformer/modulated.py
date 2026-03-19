import os
from typing import *
import torch
import torch.nn as nn
from ..basic import VarLenTensor, SparseTensor
from ..attention import SparseMultiHeadAttention
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


def _ai3d_raw_marker(marker: str) -> None:
    if os.environ.get('AI3D_SPARSE_ATTN_RAW_MARKERS') != '1':
        return
    try:
        os.write(2, f"[ai3d-raw] {marker}\n".encode('utf-8', errors='replace'))
    except Exception:
        pass


def _ai3d_use_diag_cross_modulation_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') in {'64', '128'}
    )


def _ai3d_use_diag_cross_residual_add_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_cross_mlp_modulation_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_cross_mlp_residual_add_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_apply_diag_cross_modulation(
    h: SparseTensor,
    scale_msa: torch.Tensor,
    shift_msa: torch.Tensor,
) -> SparseTensor:
    feats = h.feats
    batch_map = h.batch_boardcast_map

    scale_rows = scale_msa if scale_msa.dtype == feats.dtype else scale_msa.to(dtype=feats.dtype)
    shift_rows = shift_msa if shift_msa.dtype == feats.dtype else shift_msa.to(dtype=feats.dtype)

    _ai3d_raw_marker('pipeline_shape_slat_cross_modulation_before_one_plus_scale')
    scale_rows = scale_rows[batch_map]
    scale_rows.add_(1)
    _ai3d_raw_marker('pipeline_shape_slat_cross_modulation_after_one_plus_scale')

    _ai3d_raw_marker('pipeline_shape_slat_cross_modulation_before_mul')
    feats.mul_(scale_rows)
    _ai3d_raw_marker('pipeline_shape_slat_cross_modulation_after_mul')
    del scale_rows

    _ai3d_raw_marker('pipeline_shape_slat_cross_modulation_before_shift_add')
    feats.add_(shift_rows[batch_map])
    _ai3d_raw_marker('pipeline_shape_slat_cross_modulation_after_shift_add')
    return h


def _ai3d_apply_diag_cross_residual_add(x: SparseTensor, h: SparseTensor) -> SparseTensor:
    _ai3d_raw_marker('pipeline_shape_slat_cross_residual_add_before_add')
    feats = x.feats
    h_feats = h.feats
    if h_feats.dtype != feats.dtype:
        h_feats = h_feats.to(dtype=feats.dtype)
    feats.add_(h_feats)
    _ai3d_raw_marker('pipeline_shape_slat_cross_residual_add_after_add')

    result = x.replace(feats)
    _ai3d_raw_marker('pipeline_shape_slat_cross_residual_add_before_result_access')
    _ = result.feats
    _ai3d_raw_marker('pipeline_shape_slat_cross_residual_add_after_result_access')
    return result


def _ai3d_apply_diag_cross_mlp_modulation(
    h: SparseTensor,
    scale_mlp: torch.Tensor,
    shift_mlp: torch.Tensor,
) -> SparseTensor:
    feats = h.feats
    batch_map = h.batch_boardcast_map

    scale_rows = scale_mlp if scale_mlp.dtype == feats.dtype else scale_mlp.to(dtype=feats.dtype)
    shift_rows = shift_mlp if shift_mlp.dtype == feats.dtype else shift_mlp.to(dtype=feats.dtype)

    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_before_scale_materialize')
    scale_rows = scale_rows[batch_map]
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_after_scale_materialize')

    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_before_one_plus_scale')
    scale_rows.add_(1)
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_after_one_plus_scale')

    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_before_mul')
    feats.mul_(scale_rows)
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_after_mul')
    del scale_rows

    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_before_shift_materialize')
    shift_rows = shift_rows[batch_map]
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_after_shift_materialize')

    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_before_shift_add')
    feats.add_(shift_rows)
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_after_shift_add')
    del shift_rows

    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_before_result_access')
    _ = h.feats
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_modulation_after_result_access')
    return h


def _ai3d_apply_diag_cross_mlp_residual_add(x: SparseTensor, h: SparseTensor) -> SparseTensor:
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_residual_add_before_target_buffer_prepare')
    feats = x.feats
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_residual_add_after_target_buffer_prepare')

    h_feats = h.feats
    if h_feats.dtype != feats.dtype:
        h_feats = h_feats.to(dtype=feats.dtype)

    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_residual_add_before_add')
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_residual_add_before_inplace_add')
    feats.add_(h_feats)
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_residual_add_after_inplace_add')
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_residual_add_after_add')

    result = x.replace(feats)
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_residual_add_before_result_access')
    _ = result.feats
    _ai3d_raw_marker('pipeline_shape_slat_cross_mlp_residual_add_after_result_access')
    return result


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,

    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        if _ai3d_use_diag_cross_modulation_fix():
            h = _ai3d_apply_diag_cross_modulation(h, scale_msa, shift_msa)
        else:
            h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        if _ai3d_use_diag_cross_residual_add_fix():
            x = _ai3d_apply_diag_cross_residual_add(x, h)
        else:
            x = x + h
        _ai3d_raw_marker('pipeline_shape_slat_norm3_before_call')
        h = x.replace(self.norm3(x.feats))
        _ai3d_raw_marker('pipeline_shape_slat_norm3_after_call')
        if _ai3d_use_diag_cross_mlp_modulation_fix():
            h = _ai3d_apply_diag_cross_mlp_modulation(h, scale_mlp, shift_mlp)
        else:
            h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        if _ai3d_use_diag_cross_mlp_residual_add_fix():
            x = _ai3d_apply_diag_cross_mlp_residual_add(x, h)
        else:
            x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
