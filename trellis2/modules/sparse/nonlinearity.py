import torch
import torch.nn as nn
import torch.nn.functional as F

from . import VarLenTensor
from .linear import ai3d_linear_row_chunk, ai3d_use_5070ti_quality_path, chunked_elementwise

__all__ = [
    "SparseReLU",
    "SparseSiLU",
    "SparseGELU",
    "SparseActivation",
]


class SparseReLU(nn.ReLU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseSiLU(nn.SiLU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseGELU(nn.GELU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        if not ai3d_use_5070ti_quality_path():
            return input.replace(super().forward(input.feats))
        return chunked_elementwise(
            input,
            op=lambda chunk: F.gelu(chunk, approximate=self.approximate),
            row_chunk=ai3d_linear_row_chunk(),
            marker_prefix="pipeline_tex_slat_sparse_gelu",
            op_name="gelu",
            token_axis=0,
            output_buffer=input.feats,
        )


class SparseActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(self.activation(input.feats))
