import os
import math
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import VarLenTensor, SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from .rope import SparseRotaryPositionEmbedder


def _ai3d_raw_marker(marker: str) -> None:
    if os.environ.get('AI3D_SPARSE_ATTN_RAW_MARKERS') != '1':
        return
    try:
        os.write(2, f"[ai3d-raw] {marker}\n".encode('utf-8', errors='replace'))
    except Exception:
        pass


_AI3D_RAW_MARKER_ONCE = set()


def _ai3d_raw_marker_once(key: str, marker: str) -> None:
    if key in _AI3D_RAW_MARKER_ONCE:
        return
    _AI3D_RAW_MARKER_ONCE.add(key)
    _ai3d_raw_marker(marker)


def _ai3d_use_diag_cross_attn_q_rms_norm_fix() -> bool:
    return os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'


def _ai3d_use_diag_cross_attn_q_rms_norm_row_chunk_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_cross_attn_k_rms_norm_row_chunk_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_cross_attn_to_out_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
    )


def _ai3d_use_diag_cross_attn_flash_q_chunk_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_cross_attn_to_out_buffer_reuse_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_self_attn_to_qkv_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_texture_self_attn_q_rms_norm_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_texture_self_attn_k_rms_norm_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_cross_attn_to_q_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_cross_attn_to_kv_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_use_diag_cross_attn_to_kv_output_buffer_fix() -> bool:
    return _ai3d_use_diag_cross_attn_to_kv_fix()


def _ai3d_linear_row_chunk() -> int:
    raw = os.environ.get('AI3D_SPARSE_LINEAR_ROW_CHUNK', '').strip()
    if raw == '':
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


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
        def _maybe_emit_varlen_chunk_plan(feats: torch.Tensor) -> None:
            if marker_prefix not in {
                'pipeline_tex_slat_self_attn_q_rms_norm',
                'pipeline_tex_slat_self_attn_k_rms_norm',
            }:
                return
            shape_tuple = tuple(int(v) for v in feats.shape)
            token_axis = 0
            _ai3d_raw_marker_once(
                f'{marker_prefix}_cast_plan',
                (
                    f'{marker_prefix}_cast_plan='
                    f'cast_to_float:{int(cast_to_float)},shape:{shape_tuple},token_axis:{token_axis}'
                ),
            )
            if row_chunk > 0 and feats.shape[token_axis] > row_chunk:
                token_len = int(feats.shape[token_axis])
                num_chunks = math.ceil(token_len / row_chunk)
                first_end = min(row_chunk, token_len)
                last_start = ((token_len - 1) // row_chunk) * row_chunk
                _ai3d_raw_marker_once(
                    f'{marker_prefix}_chunk_ranges',
                    (
                        f'{marker_prefix}_chunk_ranges='
                        f'count:{num_chunks},first:0:{first_end},last:{last_start}:{token_len}'
                    ),
                )

        def _diag_token_axis(feats: torch.Tensor) -> int:
            if feats.ndim >= 4:
                return 1
            return 0

        def _maybe_emit_dense_chunk_plan(feats: torch.Tensor, effective_chunk_axis: int) -> None:
            if marker_prefix != 'pipeline_shape_slat_cross_attn_k_rms_norm':
                return
            current_chunk_axis = 0
            shape_tuple = tuple(int(v) for v in feats.shape)
            _ai3d_raw_marker_once(
                f'{marker_prefix}_shape',
                f'{marker_prefix}_shape={shape_tuple}',
            )
            _ai3d_raw_marker_once(
                f'{marker_prefix}_chunk_axes',
                (
                    f'{marker_prefix}_chunk_axes='
                    f'current:{current_chunk_axis},token:{effective_chunk_axis}'
                ),
            )
            token_len = int(feats.shape[effective_chunk_axis])
            if row_chunk > 0 and token_len > row_chunk:
                num_chunks = math.ceil(token_len / row_chunk)
                first_end = min(row_chunk, token_len)
                last_start = ((token_len - 1) // row_chunk) * row_chunk
                _ai3d_raw_marker_once(
                    f'{marker_prefix}_chunk_ranges',
                    (
                        f'{marker_prefix}_chunk_ranges='
                        f'count:{num_chunks},first:0:{first_end},last:{last_start}:{token_len}'
                    ),
                )

        x_type = x.dtype
        if isinstance(x, VarLenTensor):
            feats = x.feats
            _maybe_emit_varlen_chunk_plan(feats)
            if row_chunk > 0 and feats.shape[0] > row_chunk:
                gamma = self.gamma
                _ai3d_raw_marker(f'{marker_prefix}_before_first_chunk')
                for start in range(0, feats.shape[0], row_chunk):
                    end = min(start + row_chunk, feats.shape[0])
                    chunk = feats[start:end]
                    _ai3d_raw_marker(f'{marker_prefix}_before_cast')
                    work_chunk = chunk.float() if cast_to_float else chunk
                    _ai3d_raw_marker(f'{marker_prefix}_after_cast')
                    _ai3d_raw_marker(f'{marker_prefix}_before_normalize')
                    work_chunk = F.normalize(work_chunk, dim=-1)
                    _ai3d_raw_marker(f'{marker_prefix}_after_normalize')
                    _ai3d_raw_marker(f'{marker_prefix}_before_normalized_result_access')
                    _ai3d_raw_marker(f'{marker_prefix}_before_gamma_mul')
                    if gamma.dtype != work_chunk.dtype:
                        gamma = gamma.to(dtype=work_chunk.dtype)
                    work_chunk.mul_(gamma)
                    _ai3d_raw_marker(f'{marker_prefix}_after_gamma_mul')
                    _ai3d_raw_marker(f'{marker_prefix}_after_normalized_result_access')
                    _ai3d_raw_marker(f'{marker_prefix}_before_scale_mul')
                    work_chunk.mul_(self.scale)
                    _ai3d_raw_marker(f'{marker_prefix}_after_scale_mul')
                    if work_chunk.dtype != feats.dtype:
                        work_chunk = work_chunk.to(dtype=feats.dtype)
                    feats[start:end].copy_(work_chunk)
                    del work_chunk
                _ai3d_raw_marker(f'{marker_prefix}_after_last_chunk')
                x = x.replace(feats)
            else:
                _ai3d_raw_marker(f'{marker_prefix}_before_cast')
                feats = feats.float() if cast_to_float else feats
                _ai3d_raw_marker(f'{marker_prefix}_after_cast')
                _ai3d_raw_marker(f'{marker_prefix}_before_normalize')
                feats = F.normalize(feats, dim=-1)
                _ai3d_raw_marker(f'{marker_prefix}_after_normalize')
                _ai3d_raw_marker(f'{marker_prefix}_before_normalized_result_access')
                _ai3d_raw_marker(f'{marker_prefix}_before_gamma_mul')
                gamma = self.gamma
                if gamma.dtype != feats.dtype:
                    gamma = gamma.to(dtype=feats.dtype)
                feats.mul_(gamma)
                _ai3d_raw_marker(f'{marker_prefix}_after_gamma_mul')
                _ai3d_raw_marker(f'{marker_prefix}_after_normalized_result_access')
                _ai3d_raw_marker(f'{marker_prefix}_before_scale_mul')
                feats.mul_(self.scale)
                _ai3d_raw_marker(f'{marker_prefix}_after_scale_mul')
                x = x.replace(feats)
        else:
            feats = x.float() if cast_to_float else x
            effective_chunk_axis = _diag_token_axis(feats)
            _maybe_emit_dense_chunk_plan(feats, effective_chunk_axis)
            _ai3d_raw_marker(f'{marker_prefix}_before_normalize')
            if row_chunk > 0 and feats.shape[effective_chunk_axis] > row_chunk:
                _ai3d_raw_marker(f'{marker_prefix}_before_first_chunk')
                chunk_slices = [slice(None)] * feats.ndim
                for start in range(0, feats.shape[effective_chunk_axis], row_chunk):
                    end = min(start + row_chunk, feats.shape[effective_chunk_axis])
                    chunk_slices[effective_chunk_axis] = slice(start, end)
                    chunk_key = tuple(chunk_slices)
                    normalized_chunk = F.normalize(feats[chunk_key], dim=-1)
                    feats[chunk_key].copy_(normalized_chunk)
                    del normalized_chunk
                _ai3d_raw_marker(f'{marker_prefix}_after_last_chunk')
            else:
                feats = F.normalize(feats, dim=-1)
            _ai3d_raw_marker(f'{marker_prefix}_after_normalize')
            _ai3d_raw_marker(f'{marker_prefix}_before_normalized_result_access')
            _ai3d_raw_marker(f'{marker_prefix}_before_gamma_mul')
            gamma = self.gamma
            if gamma.dtype != feats.dtype:
                gamma = gamma.to(dtype=feats.dtype)
            feats.mul_(gamma)
            _ai3d_raw_marker(f'{marker_prefix}_after_gamma_mul')
            _ai3d_raw_marker(f'{marker_prefix}_after_normalized_result_access')
            _ai3d_raw_marker(f'{marker_prefix}_before_scale_mul')
            feats.mul_(self.scale)
            _ai3d_raw_marker(f'{marker_prefix}_after_scale_mul')
            x = feats
        return x if x.dtype == x_type else x.to(x_type)


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
        self._ai3d_diag_cross_attn_to_kv_output_buffer: Optional[torch.Tensor] = None

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
        if isinstance(x, VarLenTensor):
            feats = x.feats
            if marker_prefix is not None:
                _ai3d_raw_marker(f'{marker_prefix}_before_call')
            if row_chunk > 0 and feats.shape[0] > row_chunk:
                target_shape = (feats.shape[0], module.out_features)
                if marker_prefix is not None:
                    _ai3d_raw_marker(f'{marker_prefix}_before_output_alloc')
                if (
                    output_buffer is not None
                    and output_buffer.shape == target_shape
                    and output_buffer.device == feats.device
                ):
                    out_feats = output_buffer
                else:
                    out_feats = feats.new_empty(target_shape)
                if marker_prefix is not None:
                    _ai3d_raw_marker(f'{marker_prefix}_after_output_alloc')
                for start in range(0, feats.shape[0], row_chunk):
                    end = min(start + row_chunk, feats.shape[0])
                    chunk_out = F.linear(feats[start:end], module.weight, module.bias)
                    out_feats[start:end].copy_(chunk_out)
                    del chunk_out
            else:
                out_feats = module(feats)
            if marker_prefix is not None:
                _ai3d_raw_marker(f'{marker_prefix}_after_call')
                _ai3d_raw_marker(f'{marker_prefix}_before_result_access')
            out = x.replace(out_feats)
            if marker_prefix is not None:
                _ai3d_raw_marker(f'{marker_prefix}_after_result_access')
            return out
        else:
            if marker_prefix is not None:
                _ai3d_raw_marker(f'{marker_prefix}_before_call')
            effective_chunk_axis = 1 if x.ndim >= 3 else 0
            if row_chunk > 0 and x.shape[effective_chunk_axis] > row_chunk:
                target_shape = (*x.shape[:-1], module.out_features)
                if marker_prefix is not None:
                    _ai3d_raw_marker(f'{marker_prefix}_before_output_alloc')
                if (
                    output_buffer is not None
                    and output_buffer.shape == target_shape
                    and output_buffer.device == x.device
                    and output_buffer.dtype == x.dtype
                ):
                    out = output_buffer
                    if marker_prefix is not None:
                        _ai3d_raw_marker_once(
                            f'{marker_prefix}_output_buffer_info',
                            (
                                f'{marker_prefix}_output_buffer='
                                f'reused,shape:{tuple(int(v) for v in out.shape)},axis:{effective_chunk_axis}'
                            ),
                        )
                else:
                    out = x.new_empty(target_shape)
                if marker_prefix is not None:
                    _ai3d_raw_marker(f'{marker_prefix}_after_output_alloc')
                in_slices = [slice(None)] * x.ndim
                out_slices = [slice(None)] * out.ndim
                wrote_any_chunk = False
                for start in range(0, x.shape[effective_chunk_axis], row_chunk):
                    end = min(start + row_chunk, x.shape[effective_chunk_axis])
                    in_slices[effective_chunk_axis] = slice(start, end)
                    out_slices[effective_chunk_axis] = slice(start, end)
                    if marker_prefix is not None and not wrote_any_chunk:
                        _ai3d_raw_marker(f'{marker_prefix}_before_first_chunk_write')
                    chunk_out = F.linear(x[tuple(in_slices)], module.weight, module.bias)
                    out[tuple(out_slices)].copy_(chunk_out)
                    del chunk_out
                    wrote_any_chunk = True
                if marker_prefix is not None and wrote_any_chunk:
                    _ai3d_raw_marker(f'{marker_prefix}_after_last_chunk_write')
                result = out
            else:
                result = module(x)
            if marker_prefix is not None:
                _ai3d_raw_marker(f'{marker_prefix}_after_call')
                _ai3d_raw_marker(f'{marker_prefix}_before_result_access')
                _ = result.shape
                _ai3d_raw_marker(f'{marker_prefix}_after_result_access')
            return result

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
    
    def forward(self, x: SparseTensor, context: Optional[Union[VarLenTensor, torch.Tensor]] = None) -> SparseTensor:
        if self._type == "self":
            if _ai3d_use_diag_self_attn_to_qkv_fix():
                qkv = self._linear(
                    self.to_qkv,
                    x,
                    marker_prefix='pipeline_shape_slat_self_attn_to_qkv',
                    row_chunk=_ai3d_linear_row_chunk(),
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
                        row_chunk=_ai3d_linear_row_chunk() if _ai3d_use_diag_texture_self_attn_q_rms_norm_fix() else 0,
                    )
                    k = self.k_rms_norm(
                        k,
                        marker_prefix='pipeline_tex_slat_self_attn_k_rms_norm',
                        row_chunk=_ai3d_linear_row_chunk() if _ai3d_use_diag_texture_self_attn_k_rms_norm_fix() else 0,
                    )
                if self.use_rope:
                    q, k = self.rope(q, k)
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_q_feats')
                q_feats = q.feats
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_q_feats')
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_k_feats')
                k_feats = k.feats
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_k_feats')
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_v_feats')
                v_feats = v.feats
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_v_feats')
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_stack')
                qkv_feats = qkv.feats
                qkv_feats[:, 0].copy_(q_feats)
                qkv_feats[:, 1].copy_(k_feats)
                qkv_feats[:, 2].copy_(v_feats)
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_stack')
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_before_replace')
                qkv = qkv.replace(qkv_feats)
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_qkv_stack_after_replace')
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
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
            if _ai3d_use_diag_cross_attn_to_q_fix():
                q = self._linear(
                    self.to_q,
                    x,
                    marker_prefix='pipeline_shape_slat_cross_attn_to_q',
                    row_chunk=_ai3d_linear_row_chunk(),
                    output_buffer=x.feats,
                )
            else:
                q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv_output_buffer = None
            if (
                _ai3d_use_diag_cross_attn_to_kv_output_buffer_fix()
                and isinstance(context, torch.Tensor)
            ):
                kv_target_shape = (*context.shape[:-1], self.to_kv.out_features)
                _ai3d_raw_marker('pipeline_shape_slat_cross_attn_to_kv_before_buffer_select')
                candidate = self._ai3d_diag_cross_attn_to_kv_output_buffer
                if (
                    candidate is not None
                    and candidate.shape == kv_target_shape
                    and candidate.device == context.device
                    and candidate.dtype == context.dtype
                ):
                    kv_output_buffer = candidate
                    _ai3d_raw_marker_once(
                        'pipeline_shape_slat_cross_attn_to_kv_output_buffer_info',
                        (
                            'pipeline_shape_slat_cross_attn_to_kv_output_buffer='
                            f'module_cache,shape:{tuple(int(v) for v in candidate.shape)},axis:1'
                        ),
                    )
                _ai3d_raw_marker('pipeline_shape_slat_cross_attn_to_kv_after_buffer_select')
            if _ai3d_use_diag_cross_attn_to_kv_fix():
                kv_dense = self._linear(
                    self.to_kv,
                    context,
                    marker_prefix='pipeline_shape_slat_cross_attn_to_kv',
                    row_chunk=_ai3d_linear_row_chunk(),
                    output_buffer=kv_output_buffer,
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
                    cast_to_float=not _ai3d_use_diag_cross_attn_q_rms_norm_fix(),
                    row_chunk=_ai3d_linear_row_chunk() if _ai3d_use_diag_cross_attn_q_rms_norm_row_chunk_fix() else 0,
                )
                if _ai3d_use_diag_cross_attn_to_kv_fix():
                    _ai3d_raw_marker('pipeline_shape_slat_cross_attn_to_kv_before_split')
                k, v = kv.unbind(dim=-3)
                if _ai3d_use_diag_cross_attn_to_kv_fix():
                    _ai3d_raw_marker('pipeline_shape_slat_cross_attn_to_kv_after_split')
                k = self.k_rms_norm(
                    k,
                    marker_prefix='pipeline_shape_slat_cross_attn_k_rms_norm',
                    row_chunk=_ai3d_linear_row_chunk() if _ai3d_use_diag_cross_attn_k_rms_norm_row_chunk_fix() else 0,
                )
                if _ai3d_use_diag_cross_attn_flash_q_chunk_fix():
                    _ai3d_raw_marker('pipeline_shape_slat_cross_attn_flash_attn_before_sparse_call')
                h = sparse_scaled_dot_product_attention(q, k, v)
                if _ai3d_use_diag_cross_attn_flash_q_chunk_fix():
                    _ai3d_raw_marker('pipeline_shape_slat_cross_attn_flash_attn_before_result_access')
            else:
                h = sparse_scaled_dot_product_attention(q, kv)
            if _ai3d_use_diag_cross_attn_to_kv_output_buffer_fix() and kv_dense is not None:
                self._ai3d_diag_cross_attn_to_kv_output_buffer = kv_dense
        h = self._reshape_chs(h, (-1,))
        if self._type == "cross" and _ai3d_use_diag_cross_attn_flash_q_chunk_fix():
            _ai3d_raw_marker('pipeline_shape_slat_cross_attn_flash_attn_after_result_access')
        if self._type == "cross" and _ai3d_use_diag_cross_attn_to_out_fix():
            output_buffer = x.feats if _ai3d_use_diag_cross_attn_to_out_buffer_reuse_fix() else None
            h = self._linear(
                self.to_out,
                h,
                marker_prefix='pipeline_shape_slat_cross_attn_to_out',
                row_chunk=_ai3d_linear_row_chunk(),
                output_buffer=output_buffer,
            )
        else:
            h = self._linear(self.to_out, h)
        return h
