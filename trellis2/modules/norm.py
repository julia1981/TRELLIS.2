import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import manual_cast


def _ai3d_raw_marker(marker: str) -> None:
    if os.environ.get('AI3D_SPARSE_ATTN_RAW_MARKERS') != '1':
        return
    try:
        os.write(2, f"[ai3d-raw] {marker}\n".encode('utf-8', errors='replace'))
    except Exception:
        pass


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        _ai3d_raw_marker('pipeline_shape_slat_norm_layernorm_before_manual_cast')
        if x.is_cuda and x_dtype in (torch.float16, torch.bfloat16):
            x_norm = x
        else:
            x_norm = manual_cast(x, torch.float32)
        _ai3d_raw_marker('pipeline_shape_slat_norm_layernorm_after_manual_cast')

        if x_norm is x:
            weight = self.weight
            bias = self.bias
            if weight is not None and weight.dtype != x_dtype:
                weight = weight.to(dtype=x_dtype)
            if bias is not None and bias.dtype != x_dtype:
                bias = bias.to(dtype=x_dtype)
            o = F.layer_norm(x_norm, self.normalized_shape, weight, bias, self.eps)
        else:
            o = super().forward(x_norm)

        return o if o.dtype == x_dtype else manual_cast(o, x_dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    
