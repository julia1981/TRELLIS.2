from typing import *
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import importlib
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import cumesh


def _resolve_raster_backend():
    adapter = str(os.getenv("ML_TRELLIS2_RENDERER_ADAPTER", "") or "").strip().lower()
    if adapter == "drtk":
        return importlib.import_module("trellis2.renderers.drtk_compat")
    try:
        return importlib.import_module("nvdiffrast.torch")
    except Exception:
        return importlib.import_module("trellis2.renderers.drtk_compat")


dr = _resolve_raster_backend()


def _summarize_o3d_triangle_mesh(mesh_input, stage: str) -> dict:
    summary = {"stage": stage}
    try:
        if hasattr(mesh_input, "read"):
            vertices, faces = mesh_input.read()
        elif hasattr(mesh_input, "vertices") and hasattr(mesh_input, "faces"):
            vertices, faces = mesh_input.vertices, mesh_input.faces
        else:
            summary["reason"] = "unsupported_mesh_type"
            summary["meshType"] = type(mesh_input).__name__
            return summary

        if torch.is_tensor(vertices):
            vertices_np = vertices.detach().cpu().numpy()
        else:
            vertices_np = np.asarray(vertices)
        if torch.is_tensor(faces):
            faces_np = faces.detach().cpu().numpy()
        else:
            faces_np = np.asarray(faces)

        vertices_np = np.asarray(vertices_np, dtype=np.float64)
        faces_np = np.asarray(faces_np, dtype=np.int32)
        summary["vertices"] = int(vertices_np.shape[0]) if vertices_np.ndim == 2 else 0
        summary["faces"] = int(faces_np.shape[0]) if faces_np.ndim == 2 else 0
        if summary["vertices"] <= 0 or summary["faces"] <= 0:
            summary["reason"] = "empty_mesh"
            return summary

        trimesh_o3d = importlib.import_module("open3d")
        o3d_mesh = trimesh_o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = trimesh_o3d.utility.Vector3dVector(vertices_np)
        o3d_mesh.triangles = trimesh_o3d.utility.Vector3iVector(faces_np)
        cluster_indices, _, _ = o3d_mesh.cluster_connected_triangles()
        cluster_indices = np.asarray(cluster_indices, dtype=np.int64)
        if cluster_indices.size > 0:
            counts = np.bincount(cluster_indices)
            summary["components"] = int(counts.shape[0])
            summary["largestComponentFaces"] = int(counts.max())
        else:
            summary["components"] = 0
            summary["largestComponentFaces"] = 0
        return summary
    except Exception as exc:
        summary["error"] = str(exc)
        return summary


def _format_remesh_stage_block(summary: dict) -> str:
    stage = str(summary.get("stage") or "unknown")
    if "vertices" not in summary or "faces" not in summary:
        reason = str(summary.get("reason") or summary.get("error") or "unavailable")
        return f"Die neuen Remesh-Stufen:\n- `{stage}`\n  - `{reason}`"

    vertices = int(summary.get("vertices") or 0)
    faces = int(summary.get("faces") or 0)
    components = int(summary.get("components") or 0)
    largest_component_faces = int(summary.get("largestComponentFaces") or 0)
    return (
        "Die neuen Remesh-Stufen:\n"
        f"- `{stage}`\n"
        f"  - `{vertices:,}` Vertices\n"
        f"  - `{faces:,}` Faces\n"
        f"  - `{components:,}` Komponenten\n"
        f"  - größte Komponente `{largest_component_faces:,}` Faces"
    )


def _resolve_postremesh_component_threshold() -> float:
    raw = str(os.getenv("TRELLIS2_GLB_POSTREMESH_COMPONENT_THRESHOLD", "5e-5") or "").strip()
    lowered = raw.lower()
    if lowered in {"off", "disable", "disabled", "none"}:
        return 0.0
    try:
        return max(0.0, float(raw))
    except Exception:
        return 5e-5


def _debug_dump_stage_mesh(mesh_input, stage: str) -> None:
    dump_dir_raw = str(os.getenv("TRELLIS2_GLB_DEBUG_STAGE_DUMP_DIR", "") or "").strip()
    if not dump_dir_raw:
        return
    try:
        os.makedirs(dump_dir_raw, exist_ok=True)
        if hasattr(mesh_input, "read"):
            vertices, faces = mesh_input.read()
        elif hasattr(mesh_input, "vertices") and hasattr(mesh_input, "faces"):
            vertices, faces = mesh_input.vertices, mesh_input.faces
        else:
            return
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        else:
            vertices = np.asarray(vertices)
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        else:
            faces = np.asarray(faces)
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        safe_stage = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stage)
        out_path = os.path.join(dump_dir_raw, f"{safe_stage}.ply")
        mesh.export(out_path)
        print("[run-trellis-job] stage mesh dump written", {"stage": stage, "path": out_path})
    except Exception as exc:
        print("[run-trellis-job] warning: failed to dump stage mesh", {"stage": stage, "error": str(exc)})


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sprase tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()
    
    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    
    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    if use_tqdm:
        pbar.update(1)
        
    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = cumesh.cuBVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")
    
    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        # Step 1: Aggressive simplification (3x target)
        mesh.simplify(decimation_target * 3, verbose=verbose)
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        _stage_summary = _summarize_o3d_triangle_mesh(mesh, "after_remesh")
        print("[run-trellis-job] trellis2 remesh stage", _stage_summary)
        print(_format_remesh_stage_block(_stage_summary))
            
        # Step 3: Final simplification to target count
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        _stage_summary = _summarize_o3d_triangle_mesh(mesh, "after_simplify")
        print("[run-trellis-job] trellis2 remesh stage", _stage_summary)
        print(_format_remesh_stage_block(_stage_summary))
        
        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        _stage_summary = _summarize_o3d_triangle_mesh(mesh, "after_cleanup")
        print("[run-trellis-job] trellis2 remesh stage", _stage_summary)
        print(_format_remesh_stage_block(_stage_summary))
        _debug_dump_stage_mesh(mesh, "after_cleanup")
            
        # Step 5: Unify face orientations
        mesh.unify_face_orientations()
    
    # --- Branch 2: Remeshing Pipeline ---
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = verbose,
            bvh = bvh,
        ))
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Simplify and clean the remeshed result (similar logic to above)
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Remeshing can leave duplicated faces, non-manifold edges and
        # inconsistent winding. On Blackwell we also see many tiny shard
        # components survive into UV unwrapping, which inflates vertex count
        # without materially changing the visible surface. Apply a modest
        # post-remesh component filter before UVs are generated.
        postremesh_component_threshold = _resolve_postremesh_component_threshold()
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        if postremesh_component_threshold > 0:
            mesh.remove_small_connected_components(postremesh_component_threshold)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
        if verbose:
            print(f"After remesh cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        _stage_summary = _summarize_o3d_triangle_mesh(mesh, "after_cleanup")
        print("[run-trellis-job] trellis2 remesh stage", _stage_summary)
        print(_format_remesh_stage_block(_stage_summary))
        _debug_dump_stage_mesh(mesh, "after_cleanup")
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    
    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")
    
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    _stage_summary = _summarize_o3d_triangle_mesh(type("_DebugMesh", (), {"vertices": out_vertices, "faces": out_faces})(), "after_uv_unwrap")
    print("[run-trellis-job] trellis2 remesh stage", _stage_summary)
    print(_format_remesh_stage_block(_stage_summary))
    _debug_dump_stage_mesh(type("_DebugMesh", (), {"vertices": out_vertices, "faces": out_faces})(), "after_uv_unwrap")
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)
        
    # Setup differentiable rasterizer context
    ctx = dr.RasterizeCudaContext()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
    
    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)
    
    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0
    
    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    
    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    
    # Trilinear sampling from the attribute volume (Color, Material props)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs_sampled = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )
    attrs[mask] = attrs_sampled.to(dtype=attrs.dtype)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    mask = mask.cpu().numpy()
    
    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha_mode = 'OPAQUE'
    
    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    
    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
    
    # --- Coordinate System Conversion & Final Object ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    
    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
    
    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")
    
    return textured_mesh
