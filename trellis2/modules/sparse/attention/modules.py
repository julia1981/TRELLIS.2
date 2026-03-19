from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import VarLenTensor, SparseTensor
from ..linear import (
    ai3d_linear_row_chunk,
    ai3d_raw_marker,
    ai3d_raw_marker_once,
    ai3d_use_5070ti_quality_path,
    chunked_linear,
    chunked_rms_norm,
)
from .full_attn import sparse_scaled_dot_product_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from .rope import SparseRotaryPositionEmbedder


def _ai3d_skip_cross_attn_q_rms_float_cast() -> bool:
    return ai3d_use_5070ti_quality_path()


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(
        self,
        x: Union[VarLenTensor, torch.Tensor],
        *,
        marker_prefix: str = 'pipeline_shape_slat_q_rms_norm',
        cast_to_float: bool = True,
        row_chunk: int = 0,
    ) -> Union[VarLenTensor, torch.Tensor]:
        out_buffer = x.feats if isinstance(x, VarLenTensor) else x
        return chunked_rms_norm(
            x,
            gamma=self.gamma,
            scale=self.scale,
            cast_to_float=cast_to_float,
            row_chunk=row_chunk,
            marker_prefix=marker_prefix,
            output_buffer=out_buffer,
        )


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed", "double_windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed", "double_windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        if attn_mode == 'double_windowed':
            assert window_size % 2 == 0, "Window size must be even for double windowed attention"
            assert num_heads % 2 == 0, "Number of heads must be even for double windowed attention"
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)

    @staticmethod
    def _linear(
        module: nn.Linear,
        x: Union[VarLenTensor, torch.Tensor],
        *,
        marker_prefix: Optional[str] = None,
        row_chunk: int = 0,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> Union[VarLenTensor, torch.Tensor]:
        return chunked_linear(
            module,
            x,
            row_chunk=row_chunk,
            marker_prefix=marker_prefix,
            marker_style='attention',
            output_buffer=output_buffer,
            buffer_owner=module if ai3d_use_5070ti_quality_path() else None,
        )

    @staticmethod
    def _reshape_chs(x: Union[VarLenTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[VarLenTensor, torch.Tensor], num_fused: int) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, VarLenTensor) else x_feats

    def _attention_output_plan_kwargs(
        self,
        buffer_source: Union[VarLenTensor, torch.Tensor],
        *,
        marker_prefix: str,
        fused_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not ai3d_use_5070ti_quality_path():
            return {}
        if isinstance(buffer_source, VarLenTensor):
            feats = buffer_source.feats
        else:
            feats = buffer_source
        output_buffer = feats[:, fused_index] if fused_index is not None else feats
        return {
            'output_buffer': output_buffer,
            'scratch_owner': self,
            'marker_prefix': marker_prefix,
        }
    
    def forward(self, x: SparseTensor, context: Optional[Union[VarLenTensor, torch.Tensor]] = None) -> SparseTensor:
        if self._type == "self":
            if ai3d_use_5070ti_quality_path():
                qkv = self._linear(
                    self.to_qkv,
                    x,
                    marker_prefix='pipeline_shape_slat_self_attn_to_qkv',
                    row_chunk=ai3d_linear_row_chunk(),
                )
            else:
                qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.qk_rms_norm or self.use_rope:
                q, k, v = qkv.unbind(dim=-3)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(
                        q,
                        marker_prefix='pipeline_tex_slat_self_attn_q_rms_norm',
                        row_chunk=ai3d_linear_row_chunk() if ai3d_use_5070ti_quality_path() else 0,
                    )
                    k = self.k_rms_norm(
                        k,
                        marker_prefix='pipeline_tex_slat_self_attn_k_rms_norm',
                        row_chunk=ai3d_linear_row_chunk() if ai3d_use_5070ti_quality_path() else 0,
                    )
                if self.use_rope:
                    q, k = self.rope(q, k)
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_q_feats')
                q_feats = q.feats
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_q_feats')
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_k_feats')
                k_feats = k.feats
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_k_feats')
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_v_feats')
                v_feats = v.feats
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_v_feats')
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_stack')
                qkv_feats = qkv.feats
                qkv_feats[:, 0].copy_(q_feats)
                qkv_feats[:, 1].copy_(k_feats)
                qkv_feats[:, 2].copy_(v_feats)
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_stack')
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_replace')
                qkv = qkv.replace(qkv_feats)
                ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_replace')
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(
                    qkv,
                    **self._attention_output_plan_kwargs(
                        qkv,
                        marker_prefix='pipeline_shape_slat_self_attn_nonflash',
                        fused_index=0,
                    ),
                )
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
            elif self.attn_mode == "double_windowed":
                qkv0 = qkv.replace(qkv.feats[:, :, self.num_heads//2:])
                qkv1 = qkv.replace(qkv.feats[:, :, :self.num_heads//2])
                h0 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv0, self.window_size, shift_window=(0, 0, 0)
                )
                h1 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv1, self.window_size, shift_window=tuple([self.window_size//2] * 3)
                )
                h = qkv.replace(torch.cat([h0.feats, h1.feats], dim=1))
        else:
            if ai3d_use_5070ti_quality_path():
                q = self._linear(
                    self.to_q,
                    x,
                    marker_prefix='pipeline_shape_slat_cross_attn_to_q',
                    row_chunk=ai3d_linear_row_chunk(),
                    output_buffer=x.feats,
                )
            else:
                q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            if ai3d_use_5070ti_quality_path():
                kv_dense = self._linear(
                    self.to_kv,
                    context,
                    marker_prefix='pipeline_shape_slat_cross_attn_to_kv',
                    row_chunk=ai3d_linear_row_chunk(),
                )
                kv = kv_dense
            else:
                kv = self._linear(self.to_kv, context)
                kv_dense = None
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(
                    q,
                    marker_prefix='pipeline_shape_slat_cross_attn_q_rms_norm',
                    cast_to_float=not _ai3d_skip_cross_attn_q_rms_float_cast(),
                    row_chunk=ai3d_linear_row_chunk() if ai3d_use_5070ti_quality_path() else 0,
                )
                if ai3d_use_5070ti_quality_path():
                    ai3d_raw_marker('pipeline_shape_slat_cross_attn_to_kv_before_split')
                k, v = kv.unbind(dim=-3)
                if ai3d_use_5070ti_quality_path():
                    ai3d_raw_marker('pipeline_shape_slat_cross_attn_to_kv_after_split')
                k = self.k_rms_norm(
                    k,
                    marker_prefix='pipeline_shape_slat_cross_attn_k_rms_norm',
                    row_chunk=ai3d_linear_row_chunk() if ai3d_use_5070ti_quality_path() else 0,
                )
                if ai3d_use_5070ti_quality_path():
                    ai3d_raw_marker('pipeline_shape_slat_cross_attn_flash_attn_before_sparse_call')
                h = sparse_scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    **self._attention_output_plan_kwargs(
                        q,
                        marker_prefix='pipeline_shape_slat_cross_attn_flash_attn',
                    ),
                )
                if ai3d_use_5070ti_quality_path():
                    ai3d_raw_marker('pipeline_shape_slat_cross_attn_flash_attn_before_result_access')
            else:
                h = sparse_scaled_dot_product_attention(
                    q,
                    kv,
                    **self._attention_output_plan_kwargs(
                        q,
                        marker_prefix='pipeline_shape_slat_cross_attn_flash_attn',
                    ),
                )
        h = self._reshape_chs(h, (-1,))
        if self._type == "cross" and ai3d_use_5070ti_quality_path():
            ai3d_raw_marker('pipeline_shape_slat_cross_attn_flash_attn_after_result_access')
        if self._type == "self" and ai3d_use_5070ti_quality_path():
            h = self._linear(
                self.to_out,
                h,
                marker_prefix='pipeline_tex_slat_self_attn_to_out',
                row_chunk=ai3d_linear_row_chunk(),
                output_buffer=h.feats,
            )
        elif self._type == "cross" and ai3d_use_5070ti_quality_path():
            h = self._linear(
                self.to_out,
                h,
                marker_prefix='pipeline_shape_slat_cross_attn_to_out',
                row_chunk=ai3d_linear_row_chunk(),
                output_buffer=x.feats,
            )
        else:
            h = self._linear(self.to_out, h)
        return h
