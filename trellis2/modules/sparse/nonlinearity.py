import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import VarLenTensor

__all__ = [
    'SparseReLU',
    'SparseSiLU',
    'SparseGELU',
    'SparseActivation'
]


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


def _ai3d_use_diag_texture_sparse_gelu_fix() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_row_chunk() -> int:
    raw = os.environ.get('AI3D_SPARSE_LINEAR_ROW_CHUNK', '').strip()
    if raw == '':
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


class SparseReLU(nn.ReLU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))
    

class SparseSiLU(nn.SiLU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseGELU(nn.GELU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_before_call')
        feats = input.feats
        if _ai3d_use_diag_texture_sparse_gelu_fix():
            row_chunk = _ai3d_row_chunk()
            _ai3d_raw_marker_once(
                'pipeline_tex_slat_sparse_gelu_plan',
                (
                    'pipeline_tex_slat_sparse_gelu_plan='
                    f'shape:{tuple(int(v) for v in feats.shape)},token_axis:0,row_chunk:{row_chunk}'
                ),
            )
            if row_chunk > 0 and feats.shape[0] > row_chunk:
                _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_before_input_materialize')
                work_feats = feats
                _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_after_input_materialize')
                for start in range(0, feats.shape[0], row_chunk):
                    end = min(start + row_chunk, feats.shape[0])
                    _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_before_gelu')
                    gelu_chunk = F.gelu(work_feats[start:end], approximate=self.approximate)
                    _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_after_gelu')
                    work_feats[start:end].copy_(gelu_chunk)
                    del gelu_chunk
                result = input.replace(work_feats)
            else:
                _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_before_input_materialize')
                materialized = feats
                _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_after_input_materialize')
                _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_before_gelu')
                out_feats = F.gelu(materialized, approximate=self.approximate)
                _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_after_gelu')
                result = input.replace(out_feats)
        else:
            result = input.replace(super().forward(feats))
        _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_after_call')
        _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_before_result_access')
        _ = result.feats
        _ai3d_raw_marker('pipeline_tex_slat_sparse_gelu_after_result_access')
        return result


class SparseActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(self.activation(input.feats))
    
