#!/usr/bin/env python3
"""
usd_to_pcd.py
-------------
Convert a USD scene to a point cloud by directly sampling triangle faces.

Each triangle is sampled proportionally to its area using random barycentric
coordinates, so dense geometry gets more points and no surface is missed.
No sensor position, no ray-casting, no occlusion — just surface samples.

Optionally, a mock "intensity" channel can be derived from mesh color
(displayColor primvar, or bound material diffuseColor as a fallback) so the
resulting cloud looks more like real LiDAR data when visualized/consumed by
tools that expect an intensity field. This is NOT physically based — it's
just grayscale luminance of the mesh color, remapped to a chosen range.

Dependencies
------------
  pip install usd-core numpy
  pip install pillow    # optional — enables texture-average color for
                         # --intensity when a material has no flat color

Usage
-----
  python usd_to_pcd.py <input.usd[a|z]> [options]

Examples
--------
  python usd_to_pcd.py scene.usda                         # ~500k pts, PLY
  python usd_to_pcd.py scene.usda -n 1000000 -o map.pcd   # 1M pts, PCD
  python usd_to_pcd.py scene.usda --normals               # include normals
  python usd_to_pcd.py scene.usda --density 100           # 100 pts per unit²
  python usd_to_pcd.py scene.usda --intensity              # mock intensity from color
  python usd_to_pcd.py scene.usda --intensity --intensity-noise 5  # + jitter
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# USD extraction
# ---------------------------------------------------------------------------

def _world_matrix(prim) -> np.ndarray:
    """
    USD uses ROW-MAJOR transforms: world_pt = [x,y,z,1] @ M
    Translation is in the last ROW (M[3, :3]), NOT the last column.
    So the correct application is:  pts_world = pts_h @ M   (NOT @ M.T)
    """
    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(xform).reshape(4, 4)


_DEFAULT_COLOR = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Shader input (base) names that hold a flat/constant surface color, across
# UsdPreviewSurface and the common MDL (OmniPBR-style) convention.
_CONST_COLOR_INPUTS = ("diffuseColor", "diffuse_color_constant", "albedo_color",
                        "base_color", "baseColor", "diffuse_tint")

# Shader input (base) names that hold a texture asset (either a direct
# asset-path input, as MDL shaders use, or a texture-reader "file" input
# reached via a connection, as UsdPreviewSurface graphs use).
_TEXTURE_COLOR_INPUTS = ("diffuse_texture", "albedo_texture", "base_color_texture")
_TEXTURE_COLOR_CONNECTED = ("diffuseColor", "baseColor", "albedo")


def _texture_avg_color(asset_path, cache: dict):
    """Average RGB of a texture file, downsampled for speed. None if unreadable."""
    if asset_path is None or Image is None:
        return None
    resolved = asset_path.resolvedPath or asset_path.path
    if not resolved or resolved in cache:
        return cache.get(resolved)
    color = None
    try:
        if os.path.isfile(resolved):
            img = Image.open(resolved).convert("RGB")
            img.thumbnail((32, 32))
            color = np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0
    except Exception:
        color = None
    cache[resolved] = color
    return color


def _shader_color(shader: UsdShade.Shader, cache: dict):
    """Best-effort color for one shader prim: texture average, else constant input."""
    # MDL-style: a texture is a direct asset-path input on the shader itself.
    for name in _TEXTURE_COLOR_INPUTS:
        inp = shader.GetInput(name)
        if inp and inp.Get() is not None:
            c = _texture_avg_color(inp.Get(), cache)
            if c is not None:
                return c

    # UsdPreviewSurface-style: color input connected to a texture-reader node.
    for name in _TEXTURE_COLOR_CONNECTED:
        inp = shader.GetInput(name)
        if not inp:
            continue
        src = inp.GetConnectedSource()
        if src:
            reader = UsdShade.Shader(src[0].GetPrim())
            file_inp = reader.GetInput("file")
            if file_inp and file_inp.Get() is not None:
                c = _texture_avg_color(file_inp.Get(), cache)
                if c is not None:
                    return c

    # Flat constant color input (used when no texture is authored).
    for name in _CONST_COLOR_INPUTS:
        inp = shader.GetInput(name)
        if inp and inp.Get() is not None:
            return np.array(inp.Get(), dtype=np.float32)[:3]

    return None


def _bound_material_color(prim, cache: dict) -> np.ndarray:
    """Fallback color: sampled from the bound material's shader network."""
    try:
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not material:
            return _DEFAULT_COLOR
        for desc in Usd.PrimRange(material.GetPrim()):
            shader = UsdShade.Shader(desc)
            if not shader:
                continue
            c = _shader_color(shader, cache)
            if c is not None:
                return c
    except Exception:
        pass
    return _DEFAULT_COLOR


def _mesh_color_source(prim, mesh, texture_cache: dict):
    """
    Return (mode, values) describing per-mesh color data:
      mode="constant"    values=(1,3)  same color for every vertex
      mode="uniform"     values=(F,3)  one color per face
      mode="vertex"      values=(P,3)  one color per point, indexed like `pts`
      mode="faceVarying" values=(N,3)  one color per face-vertex, indexed like `indices`
    Falls back to the bound material's color (constant input, or texture
    average as a last resort), then flat gray.
    """
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    if primvar and primvar.HasValue():
        vals = primvar.Get()
        if vals is not None and len(vals) > 0:
            vals = np.array(vals, dtype=np.float32)
            interp = primvar.GetInterpolation()
            if interp == "uniform":
                return "uniform", vals
            if interp in ("vertex", "varying"):
                return "vertex", vals
            if interp == "faceVarying":
                return "faceVarying", vals
            return "constant", vals[:1]

    return "constant", _bound_material_color(prim, texture_cache).reshape(1, 3)


def extract_triangles(usd_path: str, with_color: bool = False):
    """Return world-space triangles, face normals, and (optionally) per-vertex color."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Cannot open: {usd_path}")

    all_v0, all_v1, all_v2 = [], [], []
    all_c0, all_c1, all_c2 = [], [], []
    texture_cache: dict = {}

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh    = UsdGeom.Mesh(prim)
        pts_a   = mesh.GetPointsAttr()
        cnt_a   = mesh.GetFaceVertexCountsAttr()
        idx_a   = mesh.GetFaceVertexIndicesAttr()
        if not (pts_a and cnt_a and idx_a):
            continue

        pts     = np.array(pts_a.Get(),  dtype=np.float32)
        counts  = np.array(cnt_a.Get(),  dtype=np.int32)
        indices = np.array(idx_a.Get(),  dtype=np.int32)
        if pts is None or len(pts) == 0:
            continue

        # Apply world transform — USD row-major: pts_world = [x,y,z,1] @ M
        M    = _world_matrix(prim)
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts  = (np.hstack([pts, ones]) @ M)[:, :3]

        if with_color:
            mode, cvals = _mesh_color_source(prim, mesh, texture_cache)

        # Triangulate faces (fan triangulation)
        i = 0
        face_idx = 0
        for count in counts:
            face = indices[i:i + count]
            for k in range(1, count - 1):
                all_v0.append(pts[face[0]])
                all_v1.append(pts[face[k]])
                all_v2.append(pts[face[k + 1]])

                if with_color:
                    if mode == "constant":
                        all_c0.append(cvals[0]); all_c1.append(cvals[0]); all_c2.append(cvals[0])
                    elif mode == "uniform":
                        c = cvals[min(face_idx, len(cvals) - 1)]
                        all_c0.append(c); all_c1.append(c); all_c2.append(c)
                    elif mode == "vertex":
                        all_c0.append(cvals[face[0]])
                        all_c1.append(cvals[face[k]])
                        all_c2.append(cvals[face[k + 1]])
                    elif mode == "faceVarying":
                        all_c0.append(cvals[i])
                        all_c1.append(cvals[i + k])
                        all_c2.append(cvals[i + k + 1])
            i += count
            face_idx += 1

    if not all_v0:
        raise RuntimeError("No mesh geometry found in USD stage.")

    v0 = np.array(all_v0, dtype=np.float32)
    v1 = np.array(all_v1, dtype=np.float32)
    v2 = np.array(all_v2, dtype=np.float32)

    # Face normals and areas
    e1      = v1 - v0
    e2      = v2 - v0
    crosses = np.cross(e1, e2)
    areas   = np.linalg.norm(crosses, axis=1) / 2.0
    normals = crosses / (np.linalg.norm(crosses, axis=1, keepdims=True) + 1e-12)

    print(f"[USD]  {len(v0):,} triangles | surface area ≈ {areas.sum():.2f} units²")

    colors = None
    if with_color:
        colors = (np.array(all_c0, dtype=np.float32),
                  np.array(all_c1, dtype=np.float32),
                  np.array(all_c2, dtype=np.float32))

    return v0, v1, v2, normals.astype(np.float32), areas.astype(np.float32), colors


# ---------------------------------------------------------------------------
# Face sampling
# ---------------------------------------------------------------------------

def sample_points(v0, v1, v2, normals, areas,
                  n_points: int = None, density: float = None,
                  seed: int = 42, colors=None,
                  intensity_scale: float = 255.0, intensity_noise: float = 0.0):
    total_area = areas.sum()
    if density is not None:
        n_points = max(1, int(total_area * density))
        print(f"[Sample] density={density}/unit² → {n_points:,} points")
    else:
        print(f"[Sample] {n_points:,} points requested")

    rng     = np.random.default_rng(seed)
    probs   = areas / total_area
    tri_idx = rng.choice(len(v0), size=n_points, p=probs)

    # Uniform barycentric sampling (Osada et al.)
    r1      = rng.random(n_points).astype(np.float32)
    r2      = rng.random(n_points).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    pts = (u[:, None] * v0[tri_idx] +
           v[:, None] * v1[tri_idx] +
           w[:, None] * v2[tri_idx])
    nrm = normals[tri_idx]

    intensity = None
    if colors is not None:
        c0, c1, c2 = colors
        rgb = (u[:, None] * c0[tri_idx] +
               v[:, None] * c1[tri_idx] +
               w[:, None] * c2[tri_idx])
        # Rec. 601 luma — mock "intensity" from mesh color, not a real return signal.
        luma = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        if intensity_noise > 0:
            luma = luma + rng.normal(0.0, intensity_noise / max(intensity_scale, 1e-6), n_points)
        intensity = np.clip(luma, 0.0, 1.0).astype(np.float32) * intensity_scale

    return pts.astype(np.float32), nrm.astype(np.float32), \
        (intensity.astype(np.float32) if intensity is not None else None)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_ply(path, points, normals=None, intensity=None):
    has_n  = normals is not None
    has_i  = intensity is not None
    props  = "property float x\nproperty float y\nproperty float z\n"
    if has_n:
        props += "property float nx\nproperty float ny\nproperty float nz\n"
    if has_i:
        props += "property float intensity\n"
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n{props}end_header\n")
    parts = [points]
    if has_n:
        parts.append(normals)
    if has_i:
        parts.append(intensity[:, None])
    data = np.hstack(parts).astype(np.float32)
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(data.tobytes())
    print(f"[PLY]  {path}  ({len(points):,} pts)")


def write_pcd(path, points, normals=None, intensity=None):
    has_n = normals is not None
    has_i = intensity is not None
    n     = len(points)

    fields = ["x", "y", "z"]
    sizes  = ["4", "4", "4"]
    types  = ["F", "F", "F"]
    counts = ["1", "1", "1"]
    parts  = [points]
    if has_n:
        fields += ["normal_x", "normal_y", "normal_z"]
        sizes  += ["4", "4", "4"]
        types  += ["F", "F", "F"]
        counts += ["1", "1", "1"]
        parts.append(normals)
    if has_i:
        fields.append("intensity")
        sizes.append("4")
        types.append("F")
        counts.append("1")
        parts.append(intensity[:, None])

    data = np.hstack(parts).astype(np.float32)
    meta = (f"SIZE {' '.join(sizes)}\nTYPE {' '.join(types)}\nCOUNT {' '.join(counts)}")
    header = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS {' '.join(fields)}\n{meta}\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              f"POINTS {n}\nDATA binary\n")
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(data.tobytes())
    print(f"[PCD]  {path}  ({n:,} pts)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Sample USD mesh faces → point cloud (.ply / .pcd)")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None,
                   help="Output path (.ply or .pcd). Default: <input>_sampled.ply")
    p.add_argument("-n", "--n-points", type=int, default=500_000,
                   help="Number of points to sample (default: 500000). "
                        "Ignored if --density is set.")
    p.add_argument("--density", type=float, default=None,
                   help="Points per unit² of surface area (overrides -n)")
    p.add_argument("--normals", action="store_true",
                   help="Include surface normals in output")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--intensity", action="store_true",
                   help="Include a mock intensity field derived from mesh color "
                        "(displayColor primvar, falling back to bound material "
                        "diffuseColor). Not a physically simulated return signal.")
    p.add_argument("--intensity-scale", type=float, default=255.0,
                   help="Max intensity value, i.e. output range is [0, scale] "
                        "(default: 255)")
    p.add_argument("--intensity-noise", type=float, default=0.0,
                   help="Stddev of Gaussian jitter added to intensity, in the "
                        "same units as --intensity-scale (default: 0, no noise)")
    args = p.parse_args()

    out = args.output or str(
        Path(args.input).parent / (Path(args.input).stem + "_sampled.ply"))
    fmt = Path(out).suffix.lower()
    if fmt not in ('.ply', '.pcd'):
        print("Output must be .ply or .pcd"); sys.exit(1)

    t0 = time.time()
    v0, v1, v2, normals, areas, colors = extract_triangles(
        args.input, with_color=args.intensity)
    pts, nrm, intensity = sample_points(v0, v1, v2, normals, areas,
                             n_points=args.n_points,
                             density=args.density,
                             seed=args.seed,
                             colors=colors,
                             intensity_scale=args.intensity_scale,
                             intensity_noise=args.intensity_noise)
    nrm_out = nrm if args.normals else None
    (write_ply if fmt == '.ply' else write_pcd)(out, pts, nrm_out, intensity)
    print(f"[Done] {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
