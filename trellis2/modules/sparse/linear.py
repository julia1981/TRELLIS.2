import os
from typing import Callable, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import VarLenTensor

__all__ = [
    "SparseLinear",
    "ai3d_raw_marker",
    "ai3d_raw_marker_once",
    "ai3d_linear_row_chunk",
    "ai3d_use_5070ti_quality_path",
    "chunked_linear",
    "chunked_rms_norm",
    "chunked_elementwise",
]


_AI3D_RAW_MARKER_ONCE = set()
_AI3D_SHARED_OUTPUT_BUFFERS: dict[tuple[str, str, str], torch.Tensor] = {}


def ai3d_raw_marker(marker: str) -> None:
    if os.environ.get("AI3D_SPARSE_ATTN_RAW_MARKERS") != "1":
        return
    try:
        os.write(2, f"[ai3d-raw] {marker}\n".encode("utf-8", errors="replace"))
    except Exception:
        pass


def ai3d_raw_marker_once(key: str, marker: str) -> None:
    if key in _AI3D_RAW_MARKER_ONCE:
        return
    _AI3D_RAW_MARKER_ONCE.add(key)
    ai3d_raw_marker(marker)


def _shared_buffer_key(marker_prefix: str, like: torch.Tensor) -> tuple[str, str, str]:
    return marker_prefix, str(like.device), str(like.dtype)


def ai3d_linear_row_chunk() -> int:
    raw = os.environ.get("AI3D_SPARSE_LINEAR_ROW_CHUNK", "").strip()
    if raw == "":
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def ai3d_use_5070ti_quality_path() -> bool:
    return (
        os.environ.get("AI3D_SELF_ATTN_NONFLASH_DIAG") == "1"
        and os.environ.get("AI3D_SELF_ATTN_NONFLASH_Q_CHUNK") == "64"
        and os.environ.get("AI3D_SELF_ATTN_NONFLASH_KV_CHUNK") == "512"
    )


def _dense_token_axis(x: torch.Tensor) -> int:
    return 1 if x.ndim >= 3 else 0


def _dense_rms_token_axis(x: torch.Tensor) -> int:
    return 1 if x.ndim >= 4 else 0


def _normalize_axis(ndim: int, axis: Optional[int], default: int) -> int:
    if axis is None:
        axis = default
    if axis < 0:
        axis += ndim
    return axis


def _slice_along_axis(x: torch.Tensor, axis: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(start, end)
    return x[tuple(slices)]


def _copy_along_axis(dst: torch.Tensor, src: torch.Tensor, axis: int, start: int, end: int) -> None:
    slices = [slice(None)] * dst.ndim
    slices[axis] = slice(start, end)
    dst[tuple(slices)].copy_(src)


def _matching_buffer(
    candidate: Optional[torch.Tensor],
    *,
    target_shape: tuple[int, ...],
    like: torch.Tensor,
) -> tuple[Optional[torch.Tensor], str]:
    if candidate is None:
        return None, "missing"
    if candidate.device != like.device:
        return None, "device_mismatch"
    if candidate.dtype != like.dtype:
        return None, "dtype_mismatch"
    if candidate.shape == target_shape:
        return candidate, "exact"
    target_numel = 1
    for dim in target_shape:
        target_numel *= int(dim)
    if candidate.numel() < target_numel:
        return None, f"shape_mismatch_numel_too_small:{tuple(int(v) for v in candidate.shape)}"
    if not candidate.is_contiguous():
        return None, "shape_mismatch_noncontiguous"
    return candidate.reshape(-1)[:target_numel].view(target_shape), "capacity_reuse"


def _acquire_output_buffer(
    like: torch.Tensor,
    *,
    target_shape: tuple[int, ...],
    output_buffer: Optional[torch.Tensor],
    buffer_owner: Optional[object],
    cache_attr: str,
    marker_prefix: Optional[str] = None,
) -> tuple[torch.Tensor, bool]:
    matched, output_reason = _matching_buffer(output_buffer, target_shape=target_shape, like=like)
    if matched is not None:
        if marker_prefix is not None:
            ai3d_raw_marker_once(
                f"{marker_prefix}_output_buffer_resolution",
                (
                    f"{marker_prefix}_output_buffer_resolution="
                    f"owner:direct_output_buffer,target_shape:{tuple(int(v) for v in target_shape)},"
                    f"like_shape:{tuple(int(v) for v in like.shape)},cache_present:0,reuse_mode:{output_reason}"
                ),
            )
        return matched, True

    cached_candidate = None
    if buffer_owner is not None:
        cached_candidate = getattr(buffer_owner, cache_attr, None)
        cached, cache_reason = _matching_buffer(cached_candidate, target_shape=target_shape, like=like)
        if cached is not None:
            if marker_prefix is not None:
                ai3d_raw_marker_once(
                    f"{marker_prefix}_output_buffer_resolution",
                    (
                        f"{marker_prefix}_output_buffer_resolution="
                        f"owner:{type(buffer_owner).__name__},target_shape:{tuple(int(v) for v in target_shape)},"
                        f"like_shape:{tuple(int(v) for v in like.shape)},cache_present:1,reuse_mode:{cache_reason}"
                    ),
                )
            return cached, True
    else:
        cache_reason = "no_buffer_owner"

    shared_candidate = None
    if marker_prefix is not None and ai3d_use_5070ti_quality_path():
        shared_candidate = _AI3D_SHARED_OUTPUT_BUFFERS.get(_shared_buffer_key(marker_prefix, like))
        shared, shared_reason = _matching_buffer(shared_candidate, target_shape=target_shape, like=like)
        if shared is not None:
            ai3d_raw_marker_once(
                f"{marker_prefix}_shared_output_buffer_resolution_{id(shared_candidate)}",
                (
                    f"{marker_prefix}_output_buffer_resolution="
                    f"owner:shared_marker_cache,target_shape:{tuple(int(v) for v in target_shape)},"
                    f"like_shape:{tuple(int(v) for v in like.shape)},cache_present:1,reuse_mode:{shared_reason}"
                ),
            )
            return shared, True
    else:
        shared_reason = "shared_cache_disabled"

    if marker_prefix is not None:
        ai3d_raw_marker_once(
            f"{marker_prefix}_output_buffer_resolution_{id(buffer_owner) if buffer_owner is not None else 'none'}",
            (
                f"{marker_prefix}_output_buffer_resolution="
                f"owner:{type(buffer_owner).__name__ if buffer_owner is not None else 'none'},"
                f"target_shape:{tuple(int(v) for v in target_shape)},"
                f"like_shape:{tuple(int(v) for v in like.shape)},"
                f"output_buffer_present:{int(output_buffer is not None)},"
                f"output_buffer_reason:{output_reason},"
                f"cache_present:{int(cached_candidate is not None)},"
                f"cache_reason:{cache_reason},"
                f"shared_cache_present:{int(shared_candidate is not None)},"
                f"shared_cache_reason:{shared_reason},"
                f"fallback:new_empty"
            ),
        )

    return like.new_empty(target_shape), False


def _remember_output_buffer(buffer_owner: Optional[object], cache_attr: str, tensor: torch.Tensor) -> None:
    if buffer_owner is None:
        return
    existing = getattr(buffer_owner, cache_attr, None)
    if (
        isinstance(existing, torch.Tensor)
        and existing.device == tensor.device
        and existing.dtype == tensor.dtype
        and existing.numel() >= tensor.numel()
    ):
        return
    setattr(buffer_owner, cache_attr, tensor)


def _remember_shared_output_buffer(marker_prefix: Optional[str], tensor: torch.Tensor) -> None:
    if marker_prefix is None or not ai3d_use_5070ti_quality_path():
        return
    key = _shared_buffer_key(marker_prefix, tensor)
    existing = _AI3D_SHARED_OUTPUT_BUFFERS.get(key)
    if existing is not None and existing.numel() >= tensor.numel():
        return
    _AI3D_SHARED_OUTPUT_BUFFERS[key] = tensor


def chunked_linear(
    module: nn.Linear,
    x: Union[VarLenTensor, torch.Tensor],
    *,
    row_chunk: int,
    marker_prefix: Optional[str] = None,
    marker_style: Literal["attention", "sparse_linear"] = "attention",
    token_axis: Optional[int] = None,
    output_buffer: Optional[torch.Tensor] = None,
    buffer_owner: Optional[object] = None,
    cache_attr: str = "_ai3d_quality_output_buffer",
) -> Union[VarLenTensor, torch.Tensor]:
    is_varlen = isinstance(x, VarLenTensor)
    feats = x.feats if is_varlen else x
    effective_token_axis = 0 if is_varlen else _normalize_axis(feats.ndim, token_axis, _dense_token_axis(feats))

    if marker_prefix is not None and marker_style == "attention":
        ai3d_raw_marker(f"{marker_prefix}_before_call")

    should_chunk = row_chunk > 0 and feats.shape[effective_token_axis] > row_chunk
    if should_chunk:
        target_shape = list(feats.shape)
        target_shape[-1] = module.out_features
        if marker_prefix is not None:
            ai3d_raw_marker(f"{marker_prefix}_before_output_alloc")
        out_feats, reused = _acquire_output_buffer(
            feats,
            target_shape=tuple(target_shape),
            output_buffer=output_buffer,
            buffer_owner=buffer_owner,
            cache_attr=cache_attr,
            marker_prefix=marker_prefix,
        )
        if reused and marker_prefix is not None and marker_style == "attention":
            ai3d_raw_marker_once(
                f"{marker_prefix}_output_buffer_info",
                (
                    f"{marker_prefix}_output_buffer="
                    f"reused,shape:{tuple(int(v) for v in out_feats.shape)},axis:{effective_token_axis}"
                ),
            )
        if marker_prefix is not None:
            ai3d_raw_marker(f"{marker_prefix}_after_output_alloc")
        wrote_any_chunk = False
        for start in range(0, feats.shape[effective_token_axis], row_chunk):
            end = min(start + row_chunk, feats.shape[effective_token_axis])
            if marker_prefix is not None:
                if marker_style == "attention":
                    if not wrote_any_chunk:
                        ai3d_raw_marker(f"{marker_prefix}_before_first_chunk_write")
                else:
                    ai3d_raw_marker(f"{marker_prefix}_before_mm")
            chunk_out = F.linear(
                _slice_along_axis(feats, effective_token_axis, start, end),
                module.weight,
                module.bias,
            )
            if marker_prefix is not None and marker_style == "sparse_linear":
                ai3d_raw_marker(f"{marker_prefix}_after_mm")
                ai3d_raw_marker(f"{marker_prefix}_before_result_access")
            _copy_along_axis(out_feats, chunk_out, effective_token_axis, start, end)
            if marker_prefix is not None and marker_style == "sparse_linear":
                ai3d_raw_marker(f"{marker_prefix}_after_result_access")
            del chunk_out
            wrote_any_chunk = True
        if marker_prefix is not None and marker_style == "attention" and wrote_any_chunk:
            ai3d_raw_marker(f"{marker_prefix}_after_last_chunk_write")
    else:
        if marker_prefix is not None and marker_style == "sparse_linear":
            ai3d_raw_marker(f"{marker_prefix}_before_mm")
        out_feats = F.linear(feats, module.weight, module.bias)
        if marker_prefix is not None and marker_style == "sparse_linear":
            ai3d_raw_marker(f"{marker_prefix}_after_mm")
            ai3d_raw_marker(f"{marker_prefix}_before_result_access")

    _remember_output_buffer(buffer_owner, cache_attr, out_feats)
    _remember_shared_output_buffer(marker_prefix, out_feats)
    result = x.replace(out_feats) if is_varlen else out_feats

    if marker_prefix is not None:
        if marker_style == "attention":
            ai3d_raw_marker(f"{marker_prefix}_after_call")
            ai3d_raw_marker(f"{marker_prefix}_before_result_access")
            _ = result.feats if is_varlen else result.shape
            ai3d_raw_marker(f"{marker_prefix}_after_result_access")
        else:
            ai3d_raw_marker(f"{marker_prefix}_after_result_access")

    return result


def chunked_elementwise(
    x: Union[VarLenTensor, torch.Tensor],
    *,
    op: Callable[[torch.Tensor], torch.Tensor],
    row_chunk: int,
    marker_prefix: Optional[str] = None,
    op_name: str = "op",
    token_axis: Optional[int] = None,
    output_buffer: Optional[torch.Tensor] = None,
    buffer_owner: Optional[object] = None,
    cache_attr: str = "_ai3d_quality_output_buffer",
) -> Union[VarLenTensor, torch.Tensor]:
    is_varlen = isinstance(x, VarLenTensor)
    feats = x.feats if is_varlen else x
    effective_token_axis = 0 if is_varlen else _normalize_axis(feats.ndim, token_axis, _dense_token_axis(feats))

    if marker_prefix is not None:
        ai3d_raw_marker(f"{marker_prefix}_before_call")
        ai3d_raw_marker_once(
            f"{marker_prefix}_plan",
            (
                f"{marker_prefix}_plan="
                f"shape:{tuple(int(v) for v in feats.shape)},token_axis:{effective_token_axis},row_chunk:{row_chunk}"
            ),
        )
        ai3d_raw_marker(f"{marker_prefix}_before_input_materialize")

    should_chunk = row_chunk > 0 and feats.shape[effective_token_axis] > row_chunk
    out_feats, _ = _acquire_output_buffer(
        feats,
        target_shape=tuple(int(v) for v in feats.shape),
        output_buffer=output_buffer,
        buffer_owner=buffer_owner,
        cache_attr=cache_attr,
    )

    if marker_prefix is not None:
        ai3d_raw_marker(f"{marker_prefix}_after_input_materialize")

    before_op = f"{marker_prefix}_before_{op_name}" if marker_prefix is not None else None
    after_op = f"{marker_prefix}_after_{op_name}" if marker_prefix is not None else None

    if should_chunk:
        for start in range(0, feats.shape[effective_token_axis], row_chunk):
            end = min(start + row_chunk, feats.shape[effective_token_axis])
            if before_op is not None:
                ai3d_raw_marker(before_op)
            chunk_out = op(_slice_along_axis(feats, effective_token_axis, start, end))
            if after_op is not None:
                ai3d_raw_marker(after_op)
            _copy_along_axis(out_feats, chunk_out, effective_token_axis, start, end)
            del chunk_out
    else:
        if before_op is not None:
            ai3d_raw_marker(before_op)
        out_feats = op(feats)
        if after_op is not None:
            ai3d_raw_marker(after_op)

    _remember_output_buffer(buffer_owner, cache_attr, out_feats)
    result = x.replace(out_feats) if is_varlen else out_feats

    if marker_prefix is not None:
        ai3d_raw_marker(f"{marker_prefix}_after_call")
        ai3d_raw_marker(f"{marker_prefix}_before_result_access")
        _ = result.feats if is_varlen else result.shape
        ai3d_raw_marker(f"{marker_prefix}_after_result_access")

    return result


def chunked_rms_norm(
    x: Union[VarLenTensor, torch.Tensor],
    *,
    gamma: torch.Tensor,
    scale: float,
    cast_to_float: bool,
    row_chunk: int,
    marker_prefix: Optional[str] = None,
    token_axis: Optional[int] = None,
    output_buffer: Optional[torch.Tensor] = None,
    buffer_owner: Optional[object] = None,
    cache_attr: str = "_ai3d_quality_output_buffer",
) -> Union[VarLenTensor, torch.Tensor]:
    is_varlen = isinstance(x, VarLenTensor)
    feats = x.feats if is_varlen else x
    effective_token_axis = 0 if is_varlen else _normalize_axis(feats.ndim, token_axis, _dense_rms_token_axis(feats))
    should_chunk = row_chunk > 0 and feats.shape[effective_token_axis] > row_chunk

    if marker_prefix is not None:
        ai3d_raw_marker_once(
            f"{marker_prefix}_cast_plan",
            (
                f"{marker_prefix}_cast_plan="
                f"cast_to_float:{int(cast_to_float)},shape:{tuple(int(v) for v in feats.shape)},token_axis:{effective_token_axis}"
            ),
        )
        if should_chunk:
            token_len = int(feats.shape[effective_token_axis])
            num_chunks = (token_len + row_chunk - 1) // row_chunk
            first_end = min(row_chunk, token_len)
            last_start = ((token_len - 1) // row_chunk) * row_chunk
            ai3d_raw_marker_once(
                f"{marker_prefix}_chunk_ranges",
                (
                    f"{marker_prefix}_chunk_ranges="
                    f"count:{num_chunks},first:0:{first_end},last:{last_start}:{token_len}"
                ),
            )

    out_feats, _ = _acquire_output_buffer(
        feats,
        target_shape=tuple(int(v) for v in feats.shape),
        output_buffer=output_buffer,
        buffer_owner=buffer_owner,
        cache_attr=cache_attr,
    )

    gamma_cache = gamma
    if should_chunk and marker_prefix is not None:
        ai3d_raw_marker(f"{marker_prefix}_before_first_chunk")

    if should_chunk:
        for start in range(0, feats.shape[effective_token_axis], row_chunk):
            end = min(start + row_chunk, feats.shape[effective_token_axis])
            if marker_prefix is not None:
                ai3d_raw_marker(f"{marker_prefix}_before_cast")
            work_chunk = _slice_along_axis(feats, effective_token_axis, start, end)
            work_chunk = work_chunk.float() if cast_to_float else work_chunk
            if marker_prefix is not None:
                ai3d_raw_marker(f"{marker_prefix}_after_cast")
                ai3d_raw_marker(f"{marker_prefix}_before_normalize")
            work_chunk = F.normalize(work_chunk, dim=-1)
            if marker_prefix is not None:
                ai3d_raw_marker(f"{marker_prefix}_after_normalize")
                ai3d_raw_marker(f"{marker_prefix}_before_normalized_result_access")
                ai3d_raw_marker(f"{marker_prefix}_before_gamma_mul")
            if gamma_cache.dtype != work_chunk.dtype:
                gamma_cache = gamma.to(dtype=work_chunk.dtype)
            work_chunk.mul_(gamma_cache)
            if marker_prefix is not None:
                ai3d_raw_marker(f"{marker_prefix}_after_gamma_mul")
                ai3d_raw_marker(f"{marker_prefix}_after_normalized_result_access")
                ai3d_raw_marker(f"{marker_prefix}_before_scale_mul")
            work_chunk.mul_(scale)
            if marker_prefix is not None:
                ai3d_raw_marker(f"{marker_prefix}_after_scale_mul")
            if work_chunk.dtype != out_feats.dtype:
                work_chunk = work_chunk.to(dtype=out_feats.dtype)
            _copy_along_axis(out_feats, work_chunk, effective_token_axis, start, end)
            del work_chunk
        if marker_prefix is not None:
            ai3d_raw_marker(f"{marker_prefix}_after_last_chunk")
    else:
        if marker_prefix is not None:
            ai3d_raw_marker(f"{marker_prefix}_before_cast")
        out_feats = feats.float() if cast_to_float else feats
        if marker_prefix is not None:
            ai3d_raw_marker(f"{marker_prefix}_after_cast")
            ai3d_raw_marker(f"{marker_prefix}_before_normalize")
        out_feats = F.normalize(out_feats, dim=-1)
        if marker_prefix is not None:
            ai3d_raw_marker(f"{marker_prefix}_after_normalize")
            ai3d_raw_marker(f"{marker_prefix}_before_normalized_result_access")
            ai3d_raw_marker(f"{marker_prefix}_before_gamma_mul")
        if gamma_cache.dtype != out_feats.dtype:
            gamma_cache = gamma.to(dtype=out_feats.dtype)
        out_feats.mul_(gamma_cache)
        if marker_prefix is not None:
            ai3d_raw_marker(f"{marker_prefix}_after_gamma_mul")
            ai3d_raw_marker(f"{marker_prefix}_after_normalized_result_access")
            ai3d_raw_marker(f"{marker_prefix}_before_scale_mul")
        out_feats.mul_(scale)
        if marker_prefix is not None:
            ai3d_raw_marker(f"{marker_prefix}_after_scale_mul")
        if out_feats.dtype != feats.dtype:
            out_feats = out_feats.to(dtype=feats.dtype)

    _remember_output_buffer(buffer_owner, cache_attr, out_feats)
    return x.replace(out_feats) if is_varlen else out_feats


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)
        self._ai3d_diag_output_buffer: Optional[torch.Tensor] = None

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return chunked_linear(
            self,
            input,
            row_chunk=ai3d_linear_row_chunk(),
            marker_prefix="pipeline_shape_slat_sparse_linear",
            marker_style="sparse_linear",
            token_axis=0,
            buffer_owner=self,
            cache_attr="_ai3d_diag_output_buffer",
        )
