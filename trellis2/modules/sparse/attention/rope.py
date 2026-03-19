import os
from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor


def _ai3d_raw_marker(marker: str) -> None:
    if os.environ.get('AI3D_SPARSE_ATTN_RAW_MARKERS') != '1':
        return
    try:
        os.write(2, f"[ai3d-raw] {marker}\n".encode('utf-8', errors='replace'))
    except Exception:
        pass


def _ai3d_use_diag_rope_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_diag_rope_row_chunk() -> int:
    raw = os.environ.get('AI3D_SPARSE_LINEAR_ROW_CHUNK', '').strip()
    if raw == '':
        return 1024
    try:
        return max(1, int(raw))
    except ValueError:
        return 1024


class SparseRotaryPositionEmbedder(nn.Module):
    def __init__(
        self, 
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0)
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        if _ai3d_use_diag_rope_fix():
            row_chunk = _ai3d_diag_rope_row_chunk()
            x_embed = torch.empty_like(x)
            for start in range(0, x.shape[0], row_chunk):
                end = min(start + row_chunk, x.shape[0])
                x_chunk = x[start:end]
                phase_chunk = phases[start:end]
                x_complex = torch.view_as_complex(x_chunk.float().reshape(*x_chunk.shape[:-1], -1, 2))
                x_rotated = x_complex * phase_chunk.unsqueeze(-2)
                _ai3d_raw_marker('pipeline_shape_slat_rope_before_view_as_real')
                x_real = torch.view_as_real(x_rotated)
                _ai3d_raw_marker('pipeline_shape_slat_rope_after_view_as_real')
                _ai3d_raw_marker('pipeline_shape_slat_rope_before_reshape')
                x_real = x_real.reshape(*x_rotated.shape[:-1], -1)
                _ai3d_raw_marker('pipeline_shape_slat_rope_after_reshape')
                _ai3d_raw_marker('pipeline_shape_slat_rope_before_to_dtype')
                x_embed[start:end].copy_(x_real)
                _ai3d_raw_marker('pipeline_shape_slat_rope_after_to_dtype')
                del x_complex, x_rotated, x_real
            return x_embed

        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        _ai3d_raw_marker('pipeline_shape_slat_rope_before_view_as_real')
        x_real = torch.view_as_real(x_rotated)
        _ai3d_raw_marker('pipeline_shape_slat_rope_after_view_as_real')
        _ai3d_raw_marker('pipeline_shape_slat_rope_before_reshape')
        x_real = x_real.reshape(*x_rotated.shape[:-1], -1)
        _ai3d_raw_marker('pipeline_shape_slat_rope_after_reshape')
        _ai3d_raw_marker('pipeline_shape_slat_rope_before_to_dtype')
        x_embed = x_real.to(x.dtype)
        _ai3d_raw_marker('pipeline_shape_slat_rope_after_to_dtype')
        return x_embed
        
    def forward(self, q: SparseTensor, k: Optional[SparseTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (SparseTensor): [..., N, H, D] tensor of queries
            k (SparseTensor): [..., N, H, D] tensor of keys
        """
        assert q.coords.shape[-1] == self.dim + 1, "Last dimension of coords must be equal to dim+1"
        phases_cache_name = f'rope_phase_{self.dim}d_freq{self.rope_freq[0]}-{self.rope_freq[1]}_hd{self.head_dim}'
        phases = q.get_spatial_cache(phases_cache_name)
        if phases is None:
            coords = q.coords[..., 1:]
            phases = self._get_phases(coords.reshape(-1)).reshape(*coords.shape[:-1], -1)
            if phases.shape[-1] < self.head_dim // 2:
                padn = self.head_dim // 2 - phases.shape[-1]
                phases = torch.cat([phases, torch.polar(
                    torch.ones(*phases.shape[:-1], padn, device=phases.device),
                    torch.zeros(*phases.shape[:-1], padn, device=phases.device)
                )], dim=-1)
            q.register_spatial_cache(phases_cache_name, phases)
        q_embed = q.replace(self._rotary_embedding(q.feats, phases))
        if k is None:
            return q_embed
        k_embed = k.replace(self._rotary_embedding(k.feats, phases))
        return q_embed, k_embed
