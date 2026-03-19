import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import VarLenTensor

__all__ = [
    'SparseLinear'
]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def _ai3d_raw_marker(self, marker: str) -> None:
        if os.environ.get('AI3D_SPARSE_ATTN_RAW_MARKERS') != '1':
            return
        try:
            os.write(2, f"[ai3d-raw] {marker}\n".encode('utf-8', errors='replace'))
        except Exception:
            pass

    def _ai3d_linear_chunk_rows(self) -> int:
        raw = os.environ.get('AI3D_SPARSE_LINEAR_ROW_CHUNK', '').strip()
        if raw == '':
            return 0
        try:
            return max(0, int(raw))
        except ValueError:
            return 0

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        feats = input.feats
        row_chunk = self._ai3d_linear_chunk_rows()

        if row_chunk > 0 and feats.shape[0] > row_chunk:
            self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_before_output_alloc')
            out_feats = feats.new_empty((feats.shape[0], self.out_features))
            self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_after_output_alloc')
            for start in range(0, feats.shape[0], row_chunk):
                end = min(start + row_chunk, feats.shape[0])
                self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_before_mm')
                chunk_out = F.linear(feats[start:end], self.weight, self.bias)
                self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_after_mm')
                self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_before_result_access')
                out_feats[start:end].copy_(chunk_out)
                self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_after_result_access')
                del chunk_out
            return input.replace(out_feats)

        self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_before_mm')
        out_feats = super().forward(feats)
        self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_after_mm')
        self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_before_result_access')
        out = input.replace(out_feats)
        self._ai3d_raw_marker('pipeline_shape_slat_sparse_linear_after_result_access')
        return out
