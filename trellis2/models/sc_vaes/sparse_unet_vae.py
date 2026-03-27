from typing import *
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from ...modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module
from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from ...modules.sparse.linear import chunked_elementwise, chunked_linear


_AI3D_DECODE_BOUNDARY_ONCE = set()


def _ai3d_use_5070ti_quality_path() -> bool:
    return (
        os.environ.get('AI3D_SELF_ATTN_NONFLASH_DIAG') == '1'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_Q_CHUNK') == '64'
        and os.environ.get('AI3D_SELF_ATTN_NONFLASH_KV_CHUNK') == '512'
    )


def _ai3d_decode_boundary_log_once(key: str, message: str) -> None:
    if not _ai3d_use_5070ti_quality_path():
        return
    if key in _AI3D_DECODE_BOUNDARY_ONCE:
        return
    _AI3D_DECODE_BOUNDARY_ONCE.add(key)
    try:
        os.write(2, f"[ai3d-decode] {message}\n".encode("utf-8", errors="replace"))
    except Exception:
        pass


def _ai3d_runtime_log_once(key: str, message: str) -> None:
    if key in _AI3D_DECODE_BOUNDARY_ONCE:
        return
    _AI3D_DECODE_BOUNDARY_ONCE.add(key)
    try:
        os.write(2, f"[ai3d-runtime] {message}\n".encode("utf-8", errors="replace"))
    except Exception:
        pass


def _ai3d_stable_sparse_snapshot(x: sp.SparseTensor) -> sp.SparseTensor:
    stable = x.replace(x.feats.detach().clone(), x.coords.detach().clone())
    stable.clear_spatial_cache()
    return stable


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: Literal['nearest', 'spatial2channel'] = 'nearest',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        self.use_checkpoint = use_checkpoint
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        if resample_mode == 'nearest':
            self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        elif resample_mode =='spatial2channel' and self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        if resample_mode == 'nearest':
            self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        elif resample_mode =='spatial2channel' and self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        self.updown = None
        if self.downsample:
            if resample_mode == 'nearest':
                self.updown = sp.SparseDownsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseSpatial2Channel(2)
        elif self.upsample:
            self.to_subdiv = sp.SparseLinear(channels, 8)
            if resample_mode == 'nearest':
                self.updown = sp.SparseUpsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseChannel2Spatial(2)

    def _updown(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > 0))
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        subdiv = None
        if self.upsample:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        if self.resample_mode == 'spatial2channel':
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == 'nearest':
            h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.upsample:
            return h, subdiv
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = sp.SparseDownsample(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        if self.pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseUpsample(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = sp.SparseSpatial2Channel(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseChannel2Spatial(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        else:
            return self._forward(x, subdiv)
        
    
class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = sp.SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            zero_module(nn.Linear(int(channels * mlp_ratio), channels)),
        )

    def _run_mlp(self, feats: torch.Tensor) -> torch.Tensor:
        row_chunk = int(getattr(self, "_ai3d_mlp_row_chunk", 0) or 0)
        rows = int(feats.shape[0]) if feats.ndim > 0 else 0
        channels = int(feats.shape[-1]) if feats.ndim > 0 else 0
        use_chunk = row_chunk > 0 and rows > row_chunk
        _ai3d_runtime_log_once(
            f"pipeline_decode_tex_convnext_mlp_plan_rows_{rows}_chunk_{row_chunk}_mode_{int(use_chunk)}",
            (
                "pipeline_decode_tex_convnext_mlp_plan="
                f"rows:{rows},channels:{channels},dtype:{str(feats.dtype)},row_chunk:{row_chunk},chunked:{int(use_chunk)}"
            ),
        )
        try:
            if not use_chunk:
                return self.mlp(feats)
            hidden = chunked_linear(
                self.mlp[0],
                feats,
                row_chunk=row_chunk,
                marker_prefix="pipeline_decode_tex_convnext_mlp_fc1",
                marker_style="sparse_linear",
                token_axis=0,
                buffer_owner=self,
                cache_attr="_ai3d_mlp_fc1_output_buffer",
            )
            hidden = chunked_elementwise(
                hidden,
                op=F.silu,
                row_chunk=row_chunk,
                marker_prefix="pipeline_decode_tex_convnext_mlp_silu",
                op_name="silu",
                token_axis=0,
                buffer_owner=self,
                cache_attr="_ai3d_mlp_silu_output_buffer",
            )
            return chunked_linear(
                self.mlp[2],
                hidden,
                row_chunk=row_chunk,
                marker_prefix="pipeline_decode_tex_convnext_mlp_fc2",
                marker_style="sparse_linear",
                token_axis=0,
                buffer_owner=self,
                cache_attr="_ai3d_mlp_fc2_output_buffer",
            )
        except Exception as exc:
            _ai3d_runtime_log_once(
                f"pipeline_decode_tex_convnext_mlp_failure_rows_{rows}_chunk_{row_chunk}_mode_{int(use_chunk)}",
                (
                    "pipeline_decode_tex_convnext_mlp_failure="
                    f"rows:{rows},channels:{channels},dtype:{str(feats.dtype)},row_chunk:{row_chunk},"
                    f"chunked:{int(use_chunk)},error:{str(exc)[:240]}"
                ),
            )
            raise

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        h = h.replace(self._run_mlp(h.feats))
        return h + x
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseUnetVaeEncoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])
        self.to_latent = sp.SparseLinear(model_channels[-1], 2 * latent_channels)
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        **block_args[i],
                    )
                )
                
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, sample_posterior=False, return_raw=False):
        h = self.input_layer(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z
    
    
class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.low_vram = False
        
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)
        self.from_latent = sp.SparseLinear(latent_channels, model_channels[0])
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[up_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
                    
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
            
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def _low_vram_cleanup(self) -> None:
        if not torch.cuda.is_available():
            return
        gc.collect()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def _prepare_blocks_for_low_vram(self) -> None:
        for res in self.blocks:
            res.cpu()
        self._low_vram_cleanup()

    def _run_block_low_vram(
        self,
        block: nn.Module,
        h: sp.SparseTensor,
        subdiv: Optional[sp.SparseTensor] = None,
    ):
        block.to(h.device)
        try:
            if subdiv is not None:
                subdiv_device = getattr(subdiv, "device", None)
                if subdiv_device is not None and str(subdiv_device) != str(h.device) and hasattr(subdiv, "to"):
                    subdiv = subdiv.to(h.device)
                if hasattr(subdiv, "clear_spatial_cache"):
                    subdiv.clear_spatial_cache()
            if subdiv is None:
                return block(h)
            return block(h, subdiv=subdiv)
        finally:
            block.cpu()
            self._low_vram_cleanup()

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None, return_subs: bool = False) -> sp.SparseTensor:
        assert guide_subs is None or self.pred_subdiv == False, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        if _ai3d_use_5070ti_quality_path() and (guide_subs is not None or return_subs):
            h.clear_spatial_cache()
        if guide_subs is not None:
            _ai3d_decode_boundary_log_once(
                'pipeline_decode_tex_decoder_entry',
                (
                    'pipeline_decode_tex_decoder_entry='
                    f'x_dtype:{str(x.feats.dtype)},x_shape:{tuple(int(v) for v in x.feats.shape)},'
                    f'x_coords:{int(x.coords.shape[0])},decoder_dtype:{str(self.dtype)},'
                    f'guide_sub_count:{len(guide_subs)},guide_sub_rows:{[int(sub.feats.shape[0]) for sub in guide_subs]},'
                    f'guide_sub_dtypes:{[str(sub.feats.dtype) for sub in guide_subs]},'
                    f'had_channel2spatial_cache:{int(x.get_spatial_cache("channel2spatial_2") is not None)}'
                ),
            )
        elif return_subs:
            _ai3d_decode_boundary_log_once(
                'pipeline_decode_shape_decoder_entry',
                (
                    'pipeline_decode_shape_decoder_entry='
                    f'x_dtype:{str(x.feats.dtype)},x_shape:{tuple(int(v) for v in x.feats.shape)},'
                    f'x_coords:{int(x.coords.shape[0])},decoder_dtype:{str(self.dtype)},'
                    f'had_channel2spatial_cache:{int(x.get_spatial_cache("channel2spatial_2") is not None)}'
                ),
            )
        subs_gt = []
        subs = []
        low_vram_blocks = bool(self.low_vram and not self.training)
        if low_vram_blocks:
            self._prepare_blocks_for_low_vram()
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))
                        if low_vram_blocks:
                            h, sub = self._run_block_low_vram(block, h)
                        else:
                            h, sub = block(h)
                        if return_subs and _ai3d_use_5070ti_quality_path():
                            stable_sub = _ai3d_stable_sparse_snapshot(sub)
                            _ai3d_decode_boundary_log_once(
                                f'pipeline_decode_shape_sub_snapshot_stage_{i}',
                                (
                                    f'pipeline_decode_shape_sub_snapshot_stage_{i}='
                                    f'rows:{int(stable_sub.feats.shape[0])},'
                                    f'positive_rows:{int((stable_sub.feats > 0).sum().item())},'
                                    f'dtype:{str(stable_sub.feats.dtype)}'
                                ),
                            )
                            subs.append(stable_sub)
                        else:
                            subs.append(sub)
                    else:
                        subdiv = guide_subs[i] if guide_subs is not None else None
                        if subdiv is not None:
                            current_rows = int(h.coords.shape[0])
                            subdiv_rows = int(subdiv.feats.shape[0])
                            if subdiv_rows != current_rows:
                                matched_index = -1
                                matched_subdiv = None
                                for alt_index, alt_subdiv in enumerate(guide_subs):
                                    if int(alt_subdiv.feats.shape[0]) == current_rows:
                                        matched_index = alt_index
                                        matched_subdiv = alt_subdiv
                                        break
                                _ai3d_decode_boundary_log_once(
                                    f'pipeline_decode_tex_guide_sub_mismatch_stage_{i}',
                                    (
                                        f'pipeline_decode_tex_guide_sub_mismatch_stage_{i}='
                                        f'h_rows:{current_rows},guide_rows:{subdiv_rows},'
                                        f'h_dtype:{str(h.feats.dtype)},guide_dtype:{str(subdiv.feats.dtype)},'
                                        f'guide_rows_all:{[int(sub.feats.shape[0]) for sub in guide_subs]},'
                                        f'matched_index:{matched_index}'
                                    ),
                                )
                                if matched_subdiv is not None:
                                    subdiv = matched_subdiv
                                    _ai3d_decode_boundary_log_once(
                                        f'pipeline_decode_tex_guide_sub_resolved_stage_{i}',
                                        (
                                            f'pipeline_decode_tex_guide_sub_resolved_stage_{i}='
                                            f'using_index:{matched_index},rows:{current_rows},dtype:{str(subdiv.feats.dtype)}'
                                        ),
                                    )
                        if low_vram_blocks:
                            h = self._run_block_low_vram(block, h, subdiv=subdiv)
                        else:
                            h = block(h, subdiv=subdiv)
                else:
                    if low_vram_blocks:
                        h = self._run_block_low_vram(block, h)
                    else:
                        h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        else:
            if return_subs:
                return h, subs
            else:
                return h
    
    def upsample(self, x: sp.SparseTensor, upsample_times: int) -> torch.Tensor:
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        low_vram_blocks = bool(self.low_vram and not self.training)
        if low_vram_blocks:
            self._prepare_blocks_for_low_vram()
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if low_vram_blocks:
                        h, sub = self._run_block_low_vram(block, h)
                    else:
                        h, sub = block(h)
                else:
                    if low_vram_blocks:
                        h = self._run_block_low_vram(block, h)
                    else:
                        h = block(h)
       
