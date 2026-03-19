import os
from contextlib import nullcontext
from itertools import accumulate
from typing import *
import torch
from .. import VarLenTensor
from .. import config
from ..linear import ai3d_use_5070ti_quality_path


__all__ = [
    'sparse_scaled_dot_product_attention',
]


_AI3D_RAW_MARKER_ONCE = set()
_AI3D_ATTENTION_OUTPUT_BUFFERS: Dict[Tuple[str, str, str], torch.Tensor] = {}


def _ai3d_raw_marker(marker: str) -> None:
    if os.environ.get('AI3D_SPARSE_ATTN_RAW_MARKERS') != '1':
        return
    try:
        os.write(2, f"[ai3d-raw] {marker}\n".encode('utf-8', errors='replace'))
    except Exception:
        pass


def _ai3d_raw_marker_once(key: str, marker: str) -> None:
    if key in _AI3D_RAW_MARKER_ONCE:
        return
    _AI3D_RAW_MARKER_ONCE.add(key)
    _ai3d_raw_marker(marker)


def _attention_output_buffer_key(marker_prefix: str, like: torch.Tensor) -> Tuple[str, str, str]:
    return marker_prefix, str(like.device), str(like.dtype)


def _matching_attention_output_buffer(
    candidate: Optional[torch.Tensor],
    *,
    target_shape: Tuple[int, ...],
    like: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], str]:
    if candidate is None:
        return None, 'missing'
    if candidate.device != like.device:
        return None, 'device_mismatch'
    if candidate.dtype != like.dtype:
        return None, 'dtype_mismatch'
    if candidate.shape == target_shape:
        return candidate, 'exact'
    target_numel = 1
    for dim in target_shape:
        target_numel *= int(dim)
    if candidate.numel() < target_numel:
        return None, f'shape_mismatch_numel_too_small:{tuple(int(v) for v in candidate.shape)}'
    if not candidate.is_contiguous():
        return None, 'shape_mismatch_noncontiguous'
    return candidate.reshape(-1)[:target_numel].view(target_shape), 'capacity_reuse'


def _get_attention_output_cache(scratch_owner: Optional[object]) -> Optional[dict]:
    if scratch_owner is None:
        return None
    cache = getattr(scratch_owner, '_ai3d_attention_output_buffers', None)
    if cache is None:
        cache = {}
        setattr(scratch_owner, '_ai3d_attention_output_buffers', cache)
    return cache


def _acquire_attention_output_buffer(
    like: torch.Tensor,
    *,
    marker_prefix: str,
    target_shape: Tuple[int, ...],
    output_buffer: Optional[torch.Tensor],
    scratch_owner: Optional[object],
    write_axis: int = 0,
) -> Tuple[torch.Tensor, str]:
    matched, output_reason = _matching_attention_output_buffer(
        output_buffer,
        target_shape=target_shape,
        like=like,
    )
    if matched is not None:
        _ai3d_raw_marker_once(
            f'{marker_prefix}_output_buffer_resolution',
            (
                f'{marker_prefix}_output_buffer_resolution='
                f'owner:direct_output_buffer,target_shape:{tuple(int(v) for v in target_shape)},'
                f'like_shape:{tuple(int(v) for v in like.shape)},reuse_mode:{output_reason}'
            ),
        )
        _ai3d_raw_marker_once(
            f'{marker_prefix}_output_buffer_info',
            (
                f'{marker_prefix}_output_buffer='
                f'reused,shape:{tuple(int(v) for v in matched.shape)},axis:{int(write_axis)}'
            ),
        )
        return matched, 'direct_output_buffer'

    owner_cache = _get_attention_output_cache(scratch_owner)
    owner_candidate = None if owner_cache is None else owner_cache.get(_attention_output_buffer_key(marker_prefix, like))
    owner_matched, owner_reason = _matching_attention_output_buffer(
        owner_candidate,
        target_shape=target_shape,
        like=like,
    )
    if owner_matched is not None:
        _ai3d_raw_marker_once(
            f'{marker_prefix}_output_buffer_resolution',
            (
                f'{marker_prefix}_output_buffer_resolution='
                f'owner:{type(scratch_owner).__name__ if scratch_owner is not None else "none"},'
                f'target_shape:{tuple(int(v) for v in target_shape)},'
                f'like_shape:{tuple(int(v) for v in like.shape)},reuse_mode:{owner_reason}'
            ),
        )
        _ai3d_raw_marker_once(
            f'{marker_prefix}_output_buffer_info',
            (
                f'{marker_prefix}_output_buffer='
                f'module_cache,shape:{tuple(int(v) for v in owner_matched.shape)},axis:{int(write_axis)}'
            ),
        )
        return owner_matched, 'owner_cache'

    shared_candidate = None
    if ai3d_use_5070ti_quality_path():
        shared_candidate = _AI3D_ATTENTION_OUTPUT_BUFFERS.get(_attention_output_buffer_key(marker_prefix, like))
    shared_matched, shared_reason = _matching_attention_output_buffer(
        shared_candidate,
        target_shape=target_shape,
        like=like,
    )
    if shared_matched is not None:
        _ai3d_raw_marker_once(
            f'{marker_prefix}_output_buffer_resolution',
            (
                f'{marker_prefix}_output_buffer_resolution='
                f'owner:shared_attention_cache,target_shape:{tuple(int(v) for v in target_shape)},'
                f'like_shape:{tuple(int(v) for v in like.shape)},reuse_mode:{shared_reason}'
            ),
        )
        _ai3d_raw_marker_once(
            f'{marker_prefix}_output_buffer_info',
            (
                f'{marker_prefix}_output_buffer='
                f'shared_cache,shape:{tuple(int(v) for v in shared_matched.shape)},axis:{int(write_axis)}'
            ),
        )
        return shared_matched, 'shared_cache'

    _ai3d_raw_marker_once(
        f'{marker_prefix}_output_buffer_resolution_{id(scratch_owner) if scratch_owner is not None else "none"}',
        (
            f'{marker_prefix}_output_buffer_resolution='
            f'owner:{type(scratch_owner).__name__ if scratch_owner is not None else "none"},'
            f'target_shape:{tuple(int(v) for v in target_shape)},'
            f'like_shape:{tuple(int(v) for v in like.shape)},'
            f'output_buffer_present:{int(output_buffer is not None)},'
            f'output_buffer_reason:{output_reason},'
            f'cache_present:{int(owner_candidate is not None)},'
            f'cache_reason:{owner_reason},'
            f'shared_cache_present:{int(shared_candidate is not None)},'
            f'shared_cache_reason:{shared_reason},'
            f'fallback:new_empty'
        ),
    )
    return like.new_empty(target_shape), 'fresh_alloc'


def _remember_attention_output_buffer(
    marker_prefix: str,
    scratch_owner: Optional[object],
    tensor: torch.Tensor,
) -> None:
    owner_cache = _get_attention_output_cache(scratch_owner)
    if owner_cache is not None:
        key = _attention_output_buffer_key(marker_prefix, tensor)
        existing = owner_cache.get(key)
        if existing is None or existing.numel() < tensor.numel():
            owner_cache[key] = tensor
    if ai3d_use_5070ti_quality_path():
        key = _attention_output_buffer_key(marker_prefix, tensor)
        existing = _AI3D_ATTENTION_OUTPUT_BUFFERS.get(key)
        if existing is None or existing.numel() < tensor.numel():
            _AI3D_ATTENTION_OUTPUT_BUFFERS[key] = tensor


class _AttentionOutputWriter:
    def __init__(
        self,
        out: torch.Tensor,
        *,
        marker_prefix: str,
        extra_first_write_marker: Optional[str] = None,
        extra_last_write_marker: Optional[str] = None,
    ):
        self.out = out
        self.marker_prefix = marker_prefix
        self.extra_first_write_marker = extra_first_write_marker
        self.extra_last_write_marker = extra_last_write_marker
        self._emitted_first_write = False
        self._emitted_extra_first_write = False
        self._emitted_any_write = False

    def write(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if not self._emitted_first_write:
            _ai3d_raw_marker(f'{self.marker_prefix}_before_first_output_write')
            self._emitted_first_write = True
        if self.extra_first_write_marker is not None and not self._emitted_extra_first_write:
            _ai3d_raw_marker(f'{self.marker_prefix}_{self.extra_first_write_marker}')
            self._emitted_extra_first_write = True
        dst.copy_(src)
        self._emitted_any_write = True

    def finalize(self) -> None:
        if self._emitted_any_write:
            _ai3d_raw_marker(f'{self.marker_prefix}_after_last_output_write')
        if self.extra_last_write_marker is not None and self._emitted_extra_first_write:
            _ai3d_raw_marker(f'{self.marker_prefix}_{self.extra_last_write_marker}')


def _build_cu_seqlens(seqlen: Sequence[int], device: torch.device, marker_prefix: str) -> torch.Tensor:
    # Build the final int32 tensor directly to avoid an extra CPU cat/int chain plus a device copy.
    _ai3d_raw_marker(f'{marker_prefix}_before_cat')
    cumulative = [0, *accumulate(seqlen)]
    _ai3d_raw_marker(f'{marker_prefix}_after_cat')
    _ai3d_raw_marker(f'{marker_prefix}_before_to_device')
    cu_seqlens = torch.tensor(cumulative, dtype=torch.int32, device=device)
    _ai3d_raw_marker(f'{marker_prefix}_after_to_device')
    return cu_seqlens


def _device_cache_key(device: torch.device) -> str:
    return f'{device.type}:{device.index if device.index is not None else -1}'


def _get_cached_cu_seqlens(
    cache_owner: Optional[VarLenTensor],
    seqlen: Sequence[int],
    device: torch.device,
    marker_prefix: str,
    cache_name: str,
) -> torch.Tensor:
    if cache_owner is None:
        return _build_cu_seqlens(seqlen, device, marker_prefix)

    cache_store = getattr(cache_owner, '_spatial_cache', None)
    if cache_store is None:
        cache_store = getattr(cache_owner, '_cache', None)
    if cache_store is None:
        cache_store = {}
        if hasattr(cache_owner, '_spatial_cache'):
            setattr(cache_owner, '_spatial_cache', cache_store)
        else:
            setattr(cache_owner, '_cache', cache_store)
    cache_bucket = cache_store.setdefault(cache_name, {})
    cache_key = (_device_cache_key(device), tuple(seqlen))
    cached = cache_bucket.get(cache_key)
    if cached is not None:
        _ai3d_raw_marker(f'{marker_prefix}_cache_hit')
        return cached

    _ai3d_raw_marker(f'{marker_prefix}_cache_miss')
    cu_seqlens = _build_cu_seqlens(seqlen, device, marker_prefix)
    cache_bucket[cache_key] = cu_seqlens
    return cu_seqlens


def _use_diag_unpacked_self_flash_attn() -> bool:
    return os.environ.get('AI3D_FLASH_ATTN_SELF_UNPACKED_DIAG') == '1'


def _use_diag_nonflash_self_attn() -> bool:
    return os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'


def _use_diag_cross_attn_flash_q_chunk() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _use_diag_nonflash_self_attn_recombine_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
    )


def _diag_nonflash_q_chunk_size() -> int:
    raw = os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK', '').strip()
    if raw == '':
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _diag_cross_attn_flash_q_chunk_size() -> int:
    if not _use_diag_cross_attn_flash_q_chunk():
        return 0
    return _diag_nonflash_q_chunk_size()


def _diag_nonflash_kv_chunk_size() -> int:
    raw = os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK', '').strip()
    if raw == '':
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _nonflash_math_sdp_context():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        pass

    cuda_backends = getattr(torch.backends, 'cuda', None)
    sdp_kernel = getattr(cuda_backends, 'sdp_kernel', None)
    if callable(sdp_kernel):
        try:
            return sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
                enable_cudnn=False,
            )
        except TypeError:
            try:
                return sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True,
                )
            except TypeError:
                pass
    return nullcontext()


def _run_diag_nonflash_self_attn(
    qkv: torch.Tensor,
    q_seqlen: Sequence[int],
    *,
    output_buffer: Optional[torch.Tensor] = None,
    scratch_owner: Optional[object] = None,
    marker_prefix: str = 'pipeline_shape_slat_self_attn_nonflash',
) -> torch.Tensor:
    _ai3d_raw_marker(f'{marker_prefix}_entered')
    _ai3d_raw_marker(f'{marker_prefix}_before_output_alloc')
    out, out_source = _acquire_attention_output_buffer(
        qkv[:, 0],
        marker_prefix=marker_prefix,
        target_shape=tuple(int(v) for v in qkv[:, 0].shape),
        output_buffer=output_buffer if ai3d_use_5070ti_quality_path() else None,
        scratch_owner=scratch_owner,
        write_axis=0,
    )
    _ai3d_raw_marker(f'{marker_prefix}_after_output_alloc')
    writer = _AttentionOutputWriter(out, marker_prefix=marker_prefix)
    _ai3d_raw_marker(f'{marker_prefix}_before_kernel')
    offset = 0
    chunk_size = _diag_nonflash_q_chunk_size()
    kv_chunk_size = _diag_nonflash_kv_chunk_size()
    use_recombine_fix = _use_diag_nonflash_self_attn_recombine_fix()
    with _nonflash_math_sdp_context():
        for length in q_seqlen:
            next_offset = offset + length
            q, k, v = qkv[offset:next_offset].unbind(dim=1)
            k = k.permute(1, 0, 2).unsqueeze(0)
            v = v.permute(1, 0, 2).unsqueeze(0)
            q = q.permute(1, 0, 2).unsqueeze(0)
            if chunk_size > 0 and chunk_size < length:
                for q_start in range(0, length, chunk_size):
                    q_end = min(q_start + chunk_size, length)
                    q_chunk = q[:, :, q_start:q_end, :]
                    if kv_chunk_size > 0 and kv_chunk_size < length:
                        scale = q_chunk.shape[-1] ** -0.5
                        q_chunk_float = q_chunk.float()
                        running_max = None
                        running_denom = None
                        running_out = None
                        for kv_start in range(0, length, kv_chunk_size):
                            kv_end = min(kv_start + kv_chunk_size, length)
                            k_chunk = k[:, :, kv_start:kv_end, :].float()
                            v_chunk = v[:, :, kv_start:kv_end, :].float()
                            scores = torch.matmul(q_chunk_float, k_chunk.transpose(-2, -1))
                            scores.mul_(scale)
                            max_chunk = scores.amax(dim=-1, keepdim=True)
                            probs = torch.exp(scores - max_chunk)
                            denom_chunk = probs.sum(dim=-1, keepdim=True)
                            weighted_values = torch.matmul(probs, v_chunk)
                            if running_max is None:
                                running_max = max_chunk
                                running_denom = denom_chunk
                                running_out = weighted_values
                            else:
                                next_max = torch.maximum(running_max, max_chunk)
                                running_scale = torch.exp(running_max - next_max)
                                chunk_scale = torch.exp(max_chunk - next_max)
                                running_out = running_out * running_scale + weighted_values * chunk_scale
                                running_denom = running_denom * running_scale + denom_chunk * chunk_scale
                                running_max = next_max
                            del k_chunk, v_chunk, scores, max_chunk, probs, denom_chunk, weighted_values
                        out_chunk = (running_out / running_denom.clamp_min(1e-12)).to(dtype=q_chunk.dtype)
                        del q_chunk_float, running_max, running_denom, running_out
                    else:
                        out_chunk = torch.nn.functional.scaled_dot_product_attention(
                            q_chunk,
                            k,
                            v,
                            dropout_p=0.0,
                            is_causal=False,
                        )
                    _ai3d_raw_marker(f'{marker_prefix}_after_kernel_return')
                    if use_recombine_fix:
                        _ai3d_raw_marker(f'{marker_prefix}_before_output_target_permute')
                        out_target = out[offset + q_start:offset + q_end].permute(1, 0, 2)
                        _ai3d_raw_marker(f'{marker_prefix}_after_output_target_permute')
                        _ai3d_raw_marker(f'{marker_prefix}_before_output_recombine')
                        writer.write(out_target, out_chunk[0])
                        _ai3d_raw_marker(f'{marker_prefix}_after_output_recombine')
                        del out_target
                    else:
                        _ai3d_raw_marker(f'{marker_prefix}_before_output_squeeze')
                        out_chunk = out_chunk.squeeze(0)
                        _ai3d_raw_marker(f'{marker_prefix}_after_output_squeeze')
                        _ai3d_raw_marker(f'{marker_prefix}_before_output_permute')
                        out_chunk = out_chunk.permute(1, 0, 2)
                        _ai3d_raw_marker(f'{marker_prefix}_after_output_permute')
                        _ai3d_raw_marker(f'{marker_prefix}_before_output_contiguous')
                        out_chunk = out_chunk.contiguous()
                        _ai3d_raw_marker(f'{marker_prefix}_after_output_contiguous')
                        _ai3d_raw_marker(f'{marker_prefix}_before_output_recombine')
                        writer.write(out[offset + q_start:offset + q_end], out_chunk)
                        _ai3d_raw_marker(f'{marker_prefix}_after_output_recombine')
                    del out_chunk
            else:
                out_chunk = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=False,
                )
                _ai3d_raw_marker(f'{marker_prefix}_after_kernel_return')
                if use_recombine_fix:
                    _ai3d_raw_marker(f'{marker_prefix}_before_output_target_permute')
                    out_target = out[offset:next_offset].permute(1, 0, 2)
                    _ai3d_raw_marker(f'{marker_prefix}_after_output_target_permute')
                    _ai3d_raw_marker(f'{marker_prefix}_before_output_recombine')
                    writer.write(out_target, out_chunk[0])
                    _ai3d_raw_marker(f'{marker_prefix}_after_output_recombine')
                    del out_target
                else:
                    _ai3d_raw_marker(f'{marker_prefix}_before_output_squeeze')
                    out_chunk = out_chunk.squeeze(0)
                    _ai3d_raw_marker(f'{marker_prefix}_after_output_squeeze')
                    _ai3d_raw_marker(f'{marker_prefix}_before_output_permute')
                    out_chunk = out_chunk.permute(1, 0, 2)
                    _ai3d_raw_marker(f'{marker_prefix}_after_output_permute')
                    _ai3d_raw_marker(f'{marker_prefix}_before_output_contiguous')
                    out_chunk = out_chunk.contiguous()
                    _ai3d_raw_marker(f'{marker_prefix}_after_output_contiguous')
                    _ai3d_raw_marker(f'{marker_prefix}_before_output_recombine')
                    writer.write(out[offset:next_offset], out_chunk)
                    _ai3d_raw_marker(f'{marker_prefix}_after_output_recombine')
                del out_chunk
            offset = next_offset
    writer.finalize()
    if out_source != 'direct_output_buffer':
        _remember_attention_output_buffer(marker_prefix, scratch_owner, out)
    _ai3d_raw_marker(f'{marker_prefix}_after_kernel')
    _ai3d_raw_marker(f'{marker_prefix}_before_result_access')
    _ = out.shape
    _ai3d_raw_marker(f'{marker_prefix}_after_result_access')
    return out


def _run_diag_cross_attn_flash_q_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: Sequence[int],
    kv_seqlen: Sequence[int],
    device: torch.device,
    *,
    output_buffer: Optional[torch.Tensor] = None,
    scratch_owner: Optional[object] = None,
    marker_prefix: str = 'pipeline_shape_slat_cross_attn_flash_attn',
) -> torch.Tensor:
    import flash_attn

    q_chunk_size = _diag_cross_attn_flash_q_chunk_size()
    assert q_chunk_size > 0, 'diagnostic cross-attn flash q chunk size must be positive'
    max_q_seqlen = max(q_seqlen)
    max_kv_seqlen = max(kv_seqlen)
    _ai3d_raw_marker_once(
        f'{marker_prefix}_q_shape',
        f'{marker_prefix}_q_shape=flat:{tuple(int(v) for v in q.shape)},max_q:{int(max_q_seqlen)},max_kv:{int(max_kv_seqlen)}',
    )
    _ai3d_raw_marker_once(
        f'{marker_prefix}_q_chunk_axis',
        f'{marker_prefix}_q_chunk_axis=0',
    )
    if max_q_seqlen > q_chunk_size:
        num_q_chunks = (max_q_seqlen + q_chunk_size - 1) // q_chunk_size
        first_end = min(q_chunk_size, max_q_seqlen)
        last_start = ((max_q_seqlen - 1) // q_chunk_size) * q_chunk_size
        _ai3d_raw_marker_once(
            f'{marker_prefix}_q_chunk_ranges',
            (
                f'{marker_prefix}_q_chunk_ranges='
                f'count:{num_q_chunks},first:0:{first_end},last:{last_start}:{int(max_q_seqlen)}'
            ),
        )

    _ai3d_raw_marker(f'{marker_prefix}_before_output_alloc')
    out, out_source = _acquire_attention_output_buffer(
        q,
        marker_prefix=marker_prefix,
        target_shape=tuple(int(v) for v in q.shape),
        output_buffer=output_buffer,
        scratch_owner=scratch_owner,
        write_axis=0,
    )
    _ai3d_raw_marker(f'{marker_prefix}_after_output_alloc')
    writer = _AttentionOutputWriter(
        out,
        marker_prefix=marker_prefix,
        extra_first_write_marker='before_first_q_chunk_write',
        extra_last_write_marker='after_last_q_chunk_write',
    )
    q_offsets = [0, *accumulate(q_seqlen)]
    kv_offsets = [0, *accumulate(kv_seqlen)]
    cu_cache: Dict[int, torch.Tensor] = {}
    emitted_first_q_chunk = False

    def _get_pair_cu(length: int) -> torch.Tensor:
        cached = cu_cache.get(length)
        if cached is None:
            cached = torch.tensor([0, length], dtype=torch.int32, device=device)
            cu_cache[length] = cached
        return cached

    for batch_idx, q_len in enumerate(q_seqlen):
        kv_len = kv_seqlen[batch_idx]
        q_start = q_offsets[batch_idx]
        q_end = q_offsets[batch_idx + 1]
        kv_start = kv_offsets[batch_idx]
        kv_end = kv_offsets[batch_idx + 1]
        q_seq = q[q_start:q_end]
        k_seq = k[kv_start:kv_end]
        v_seq = v[kv_start:kv_end]

        if q_chunk_size < q_len:
            if not emitted_first_q_chunk:
                _ai3d_raw_marker(f'{marker_prefix}_before_first_q_chunk')
                emitted_first_q_chunk = True
            for local_q_start in range(0, q_len, q_chunk_size):
                local_q_end = min(local_q_start + q_chunk_size, q_len)
                q_chunk = q_seq[local_q_start:local_q_end]
                cu_q_chunk = _get_pair_cu(local_q_end - local_q_start)
                cu_kv_seq = _get_pair_cu(kv_len)
                _ai3d_raw_marker(f'{marker_prefix}_before_kernel')
                out_chunk = flash_attn.flash_attn_varlen_func(
                    q_chunk,
                    k_seq,
                    v_seq,
                    cu_q_chunk,
                    cu_kv_seq,
                    local_q_end - local_q_start,
                    kv_len,
                )
                _ai3d_raw_marker(f'{marker_prefix}_after_kernel')
                writer.write(out[q_start + local_q_start:q_start + local_q_end], out_chunk)
                del q_chunk, out_chunk
        else:
            cu_q_seq = _get_pair_cu(q_len)
            cu_kv_seq = _get_pair_cu(kv_len)
            _ai3d_raw_marker(f'{marker_prefix}_before_kernel')
            out_chunk = flash_attn.flash_attn_varlen_func(
                q_seq,
                k_seq,
                v_seq,
                cu_q_seq,
                cu_kv_seq,
                q_len,
                kv_len,
            )
            _ai3d_raw_marker(f'{marker_prefix}_after_kernel')
            writer.write(out[q_start:q_end], out_chunk)
            del out_chunk

    if emitted_first_q_chunk:
        _ai3d_raw_marker(f'{marker_prefix}_after_last_q_chunk')
    writer.finalize()
    if out_source != 'direct_output_buffer':
        _remember_attention_output_buffer(marker_prefix, scratch_owner, out)
    return out


@overload
def sparse_scaled_dot_product_attention(qkv: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (VarLenTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, kv: Union[VarLenTensor, torch.Tensor]) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (VarLenTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, C] dense tensor containing Qs.
        kv (VarLenTensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: VarLenTensor, v: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: torch.Tensor, v: torch.Tensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: VarLenTensor, v: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    output_buffer = kwargs.pop('output_buffer', None)
    scratch_owner = kwargs.pop('scratch_owner', None)
    marker_prefix = kwargs.pop('marker_prefix', None)
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, VarLenTensor), f"qkv must be a VarLenTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device
        q_cache_owner = qkv
        kv_cache_owner = qkv

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, VarLenTensor) and isinstance(kv, (VarLenTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, VarLenTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            q_cache_owner = q
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            q_cache_owner = None
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, VarLenTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_cache_owner = kv
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            kv_cache_owner = None
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, VarLenTensor) and isinstance(k, (VarLenTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, VarLenTensor) and isinstance(v, VarLenTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            q_cache_owner = q
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            q_cache_owner = None
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, VarLenTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_cache_owner = k
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            kv_cache_owner = None
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        if num_all_args == 1 and _use_diag_nonflash_self_attn():
            out = _run_diag_nonflash_self_attn(
                qkv,
                q_seqlen,
                output_buffer=output_buffer,
                scratch_owner=scratch_owner,
                marker_prefix=marker_prefix or 'pipeline_shape_slat_self_attn_nonflash',
            )
            return s.replace(out) if s is not None else out
        cu_seqlens_q = _get_cached_cu_seqlens(
            q_cache_owner,
            q_seqlen,
            device,
            'pipeline_shape_slat_sparse_attn_cu_seqlens_q',
            'ai3d_flash_attn_cu_seqlens_q_int32',
        )
        if num_all_args in [2, 3]:
            kv_effective_cache_owner = kv_cache_owner if kv_cache_owner is not None else q_cache_owner
            cu_seqlens_kv = _get_cached_cu_seqlens(
                kv_effective_cache_owner,
                kv_seqlen,
                device,
                'pipeline_shape_slat_sparse_attn_cu_seqlens_kv',
                'ai3d_flash_attn_cu_seqlens_kv_int32',
            )
        if num_all_args == 1:
            if _use_diag_unpacked_self_flash_attn():
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_flash_attn_unpacked_before_kernel')
                q, k, v = qkv.unbind(dim=1)
                out = flash_attn.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_q,
                    max(q_seqlen),
                    max(q_seqlen),
                )
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_flash_attn_unpacked_after_kernel')
            else:
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_flash_attn_qkvpacked_before_kernel')
                out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
                _ai3d_raw_marker('pipeline_shape_slat_self_attn_flash_attn_qkvpacked_after_kernel')
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            if _use_diag_cross_attn_flash_q_chunk():
                out = _run_diag_cross_attn_flash_q_chunk(
                    q,
                    k,
                    v,
                    q_seqlen,
                    kv_seqlen,
                    device,
                    output_buffer=output_buffer,
                    scratch_owner=scratch_owner,
                    marker_prefix=marker_prefix or 'pipeline_shape_slat_cross_attn_flash_attn',
                )
            else:
                _ai3d_raw_marker('pipeline_shape_slat_cross_attn_flash_attn_before_kernel')
                out = flash_attn.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max(q_seqlen),
                    max(kv_seqlen),
                )
                _ai3d_raw_marker('pipeline_shape_slat_cross_attn_flash_attn_after_kernel')
    elif config.ATTN == 'flash_attn_3':
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
        cu_seqlens_q = _get_cached_cu_seqlens(
            q_cache_owner,
            q_seqlen,
            device,
            'pipeline_shape_slat_sparse_attn3_cu_seqlens_q',
            'ai3d_flash_attn3_cu_seqlens_q_int32',
        )
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            cu_seqlens_kv = cu_seqlens_q.clone()
            max_q_seqlen = max_kv_seqlen = max(q_seqlen)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
            cu_seqlens_kv = _build_cu_seqlens(kv_seqlen, device, 'pipeline_shape_slat_sparse_attn3_cu_seqlens_kv')
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        elif num_all_args == 3:
            cu_seqlens_kv = _build_cu_seqlens(kv_seqlen, device, 'pipeline_shape_slat_sparse_attn3_cu_seqlens_kv')
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")
    
    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
