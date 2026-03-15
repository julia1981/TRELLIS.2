from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

try:
    import drtk  # type: ignore
except Exception as exc:  # pragma: no cover - runtime only
    raise RuntimeError(f"drtk import failed: {exc}") from exc


@dataclass
class RasterizeCudaContext:
    device: str | torch.device = "cuda"


def _to_hw(resolution: Iterable[int]) -> Tuple[int, int]:
    values = list(resolution)
    if len(values) != 2:
        raise ValueError(f"resolution must be [H, W], got {values}")
    h = int(values[0])
    w = int(values[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"resolution must be positive, got {values}")
    return h, w


def _to_screen(pos_clip: torch.Tensor, height: int, width: int) -> torch.Tensor:
    # pos_clip: [V, 4] in NDC-like clip-space. Convert to pixel space for drtk.
    w = torch.where(pos_clip[:, 3].abs() > 1e-8, pos_clip[:, 3], torch.ones_like(pos_clip[:, 3]))
    x_ndc = pos_clip[:, 0] / w
    y_ndc = pos_clip[:, 1] / w
    z = pos_clip[:, 2] / w
    x = (x_ndc * 0.5 + 0.5) * (width - 1)
    # flip Y from OpenGL-style clip space to image raster space
    y = (1.0 - (y_ndc * 0.5 + 0.5)) * (height - 1)
    return torch.stack([x, y, z], dim=-1)


def rasterize(
    _ctx: RasterizeCudaContext,
    pos: torch.Tensor,
    tri: torch.Tensor,
    resolution: Iterable[int],
):
    # nvdiffrast-compatible subset used by o_voxel.postprocess.
    # pos: [1, V, 4], tri: [F, 3]
    if pos.dim() != 3 or pos.shape[0] != 1 or pos.shape[-1] < 3:
        raise ValueError(f"pos must be [1, V, 4], got {tuple(pos.shape)}")
    if tri.dim() != 2 or tri.shape[1] != 3:
        raise ValueError(f"tri must be [F, 3], got {tuple(tri.shape)}")
    height, width = _to_hw(resolution)
    pos_clip = pos[0]
    screen_vertices = _to_screen(pos_clip, height, width)
    face_indices = tri.to(dtype=torch.int32, device=screen_vertices.device).contiguous()
    # DRTK expects batched tensors [N, V, 3] and [N, F, 3].
    if screen_vertices.dim() == 2:
        screen_vertices = screen_vertices.unsqueeze(0)
    if face_indices.dim() == 2:
        face_indices = face_indices.unsqueeze(0)

    index_img = drtk.rasterize(screen_vertices, face_indices, height=height, width=width)
    _depth_img, bary_img = drtk.render(screen_vertices, face_indices, index_img)
    if index_img.dim() == 2:
        index_img = index_img.unsqueeze(0)
    # Normalize barycentric output to [N, H, W, 3].
    if bary_img.dim() == 3 and bary_img.shape[-1] == 3:
        bary_img = bary_img.unsqueeze(0)
    elif bary_img.dim() == 4 and bary_img.shape[1] == 3:
        bary_img = bary_img.permute(0, 2, 3, 1).contiguous()
    valid = index_img >= 0

    rast = torch.zeros((1, height, width, 4), dtype=torch.float32, device=screen_vertices.device)
    rast[0, :, :, :3] = bary_img[0].to(torch.float32)
    # nvdiffrast convention: triangle id + 1, 0 means background.
    rast[0, :, :, 3] = torch.where(
        valid[0],
        index_img[0].to(torch.float32) + 1.0,
        torch.zeros_like(index_img[0], dtype=torch.float32),
    )
    return rast, None


def interpolate(
    attr: torch.Tensor,
    rast: torch.Tensor,
    tri: torch.Tensor,
):
    # nvdiffrast-compatible subset used by o_voxel.postprocess.
    # attr: [1, V, C], rast: [1, H, W, 4], tri: [F, 3]
    if attr.dim() != 3 or attr.shape[0] != 1:
        raise ValueError(f"attr must be [1, V, C], got {tuple(attr.shape)}")
    if rast.dim() != 4 or rast.shape[0] != 1 or rast.shape[-1] < 4:
        raise ValueError(f"rast must be [1, H, W, 4], got {tuple(rast.shape)}")
    if tri.dim() != 2 or tri.shape[1] != 3:
        raise ValueError(f"tri must be [F, 3], got {tuple(tri.shape)}")

    device = rast.device
    out = torch.zeros((1, rast.shape[1], rast.shape[2], attr.shape[2]), dtype=attr.dtype, device=device)

    tri_id = (rast[0, :, :, 3] - 1.0).to(torch.long)
    valid = tri_id >= 0
    if valid.any():
        tri_dev = tri.to(device=device, dtype=torch.int32)
        face = tri_dev[tri_id[valid]]
        bary = rast[0, :, :, :3][valid].to(dtype=attr.dtype)
        vertices = attr[0]
        v0 = vertices[face[:, 0]]
        v1 = vertices[face[:, 1]]
        v2 = vertices[face[:, 2]]
        out_vals = (v0 * bary[:, 0:1]) + (v1 * bary[:, 1:2]) + (v2 * bary[:, 2:3])
        out[0, valid] = out_vals

    return out, None
