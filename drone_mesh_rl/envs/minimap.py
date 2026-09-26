"""
minimap.py — Camera-following HUD panel of the single-patch USGS world.

The map texture is built once from the DEM ground coloring plus the one
district, camp, and hut markers. Each frame the panel is placed in the
lower-left of the spectator view; only the player arrow moves.
"""

from __future__ import annotations

import math
import os
import struct
import zlib

import numpy as np
import pybullet as p

from .terrain import _write_png_rgb

_MAP_NAME = "minimap.png"
_HUD_DIST = 2.4
_PANEL_FRAC = 0.44
_FOV_DEG = 60.0
_ARROW_RGB = [1.0, 0.95, 0.15, 1.0]
_PARK = [0.0, 0.0, -1.0e6]
_DISTRICT_RGB = np.array([255, 230, 55], dtype=np.uint8)
_DISTRICT_DOT = np.array([255, 255, 255], dtype=np.uint8)
_CAMP_RGB = np.array([255, 145, 40], dtype=np.uint8)
_HUT_RGB = np.array([240, 70, 200], dtype=np.uint8)
_BORDER = np.array([18, 18, 20], dtype=np.uint8)


def _read_png_rgb(path):
    """Read an 8-bit RGB PNG written by _write_png_rgb (filter 0)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    pos = 8
    width = height = None
    idat = b""
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height = struct.unpack(">II", chunk[:8])
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    raw = zlib.decompress(idat)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    stride = 1 + width * 3
    for r in range(height):
        filt = raw[r * stride]
        row = raw[r * stride + 1 : (r + 1) * stride]
        if filt != 0:
            raise ValueError(f"unsupported PNG filter {filt}")
        rgb[r] = np.frombuffer(row, dtype=np.uint8).reshape(width, 3)
    return rgb


def _world_to_uv(x, y, size_x, size_y):
    u = (float(x) + 0.5 * size_x) / max(size_x, 1e-9)
    v = (float(y) + 0.5 * size_y) / max(size_y, 1e-9)
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


def _uv_to_rc(u, v, width, height):
    col = int(np.clip(round(u * (width - 1)), 0, width - 1))
    row = int(np.clip(round((1.0 - v) * (height - 1)), 0, height - 1))
    return row, col


def _fill_disk(rgb, row, col, radius, color):
    h, w = rgb.shape[:2]
    r0 = max(0, row - radius)
    r1 = min(h, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(w, col + radius + 1)
    rr = np.arange(r0, r1)[:, None]
    cc = np.arange(c0, c1)[None, :]
    mask = (rr - row) ** 2 + (cc - col) ** 2 <= radius * radius
    rgb[r0:r1, c0:c1][mask] = color


def _stroke_rect(rgb, r0, c0, r1, c1, color, t=2):
    h, w = rgb.shape[:2]
    r0, r1 = max(0, min(r0, r1)), min(h - 1, max(r0, r1))
    c0, c1 = max(0, min(c0, c1)), min(w - 1, max(c0, c1))
    rgb[r0 : min(h, r0 + t), c0 : c1 + 1] = color
    rgb[max(0, r1 - t + 1) : r1 + 1, c0 : c1 + 1] = color
    rgb[r0 : r1 + 1, c0 : min(w, c0 + t)] = color
    rgb[r0 : r1 + 1, max(0, c1 - t + 1) : c1 + 1] = color


def build_minimap_image(terrain, districts, scenery, filename=_MAP_NAME):
    """
    Full-world north-up map, generated once.

    Reuses the DEM ground texture (green / rock / snow), undoes the
    heightfield column-mirror so west is left, and flips rows so north is up.
    """
    src = terrain.get_ground_texture_path()
    rgb = _read_png_rgb(src).copy()
    # Heightfield PNG: row 0 = south (−Y), column 0 = east (+X).
    rgb = np.ascontiguousarray(rgb[:, ::-1])
    rgb = np.ascontiguousarray(rgb[::-1])
    h, w = rgb.shape[:2]
    sx, sy = float(terrain.size_x), float(terrain.size_y)

    half = 0.5 * float(districts[0].size)
    for d in districts:
        cx, cy = float(d.center[0]), float(d.center[1])
        u0, v0 = _world_to_uv(cx - half, cy - half, sx, sy)
        u1, v1 = _world_to_uv(cx + half, cy + half, sx, sy)
        r0, c0 = _uv_to_rc(u0, v1, w, h)
        r1, c1 = _uv_to_rc(u1, v0, w, h)
        patch = rgb[r0 : r1 + 1, c0 : c1 + 1]
        if patch.size:
            rgb[r0 : r1 + 1, c0 : c1 + 1] = np.clip(
                patch.astype(np.int16) + 28, 0, 255
            ).astype(np.uint8)
        _stroke_rect(rgb, r0, c0, r1, c1, _DISTRICT_RGB, t=2)
        rr, cc = _uv_to_rc(*_world_to_uv(cx, cy, sx, sy), w, h)
        _fill_disk(rgb, rr, cc, 4, _DISTRICT_DOT)

    camps = (scenery or {}).get("camps") or []
    huts = (scenery or {}).get("huts") or []
    for cx, cy in camps:
        rr, cc = _uv_to_rc(*_world_to_uv(cx, cy, sx, sy), w, h)
        _fill_disk(rgb, rr, cc, 3, _CAMP_RGB)
    for hx, hy in huts:
        rr, cc = _uv_to_rc(*_world_to_uv(hx, hy, sx, sy), w, h)
        _fill_disk(rgb, rr, cc, 3, _HUT_RGB)

    rgb[:3, :] = _BORDER
    rgb[-3:, :] = _BORDER
    rgb[:, :3] = _BORDER
    rgb[:, -3:] = _BORDER

    path = os.path.join(os.path.dirname(__file__), filename)
    _write_png_rgb(path, rgb)
    print(
        f"[MINIMAP] texture={os.path.basename(path)} {w}x{h}  "
        f"districts={len(districts)} camps={len(camps)} huts={len(huts)}  "
        f"(north up, built once)"
    )
    return path, rgb


def _mat_to_quat(R):
    """Rotation matrix (columns = axes) → PyBullet xyzw quaternion."""
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return [qx, qy, qz, qw]


def _panel_axes(forward, right):
    cam_up = np.cross(right, forward)
    un = np.linalg.norm(cam_up)
    if un < 1e-8:
        cam_up = np.array([0.0, 0.0, 1.0])
    else:
        cam_up = cam_up / un
    # +Z toward the camera so the textured +Z face is visible.
    R = np.column_stack([right, cam_up, -forward])
    return cam_up, _mat_to_quat(R)


def _frustum_half_extents(client, dist=_HUD_DIST):
    """Visible half-width / half-height at HUD distance from the live camera."""
    aspect = 16.0 / 9.0
    fov_deg = _FOV_DEG
    try:
        info = p.getDebugVisualizerCamera(physicsClientId=client)
        w, h = float(info[0]), float(info[1])
        if w > 1.0 and h > 1.0:
            aspect = w / h
        proj = info[3]
        if proj is not None and len(proj) >= 16:
            # Column-major OpenGL projection → row-major.
            P = np.asarray(proj, dtype=np.float64).reshape(4, 4).T
            p00, p11 = float(P[0, 0]), float(P[1, 1])
            if abs(p00) > 1e-8 and abs(p11) > 1e-8:
                half_w = dist / p00
                half_h = dist / p11
                if half_w > 1e-6 and half_h > 1e-6:
                    fov_deg = 2.0 * math.degrees(math.atan(1.0 / p11))
                    return half_w, half_h, half_w / half_h, fov_deg
    except Exception:
        pass
    half_h = dist * math.tan(math.radians(fov_deg) * 0.5)
    half_w = half_h * aspect
    return half_w, half_h, aspect, fov_deg


def _panel_size(terrain, half_w, half_h):
    view_w, view_h = 2.0 * half_w, 2.0 * half_h
    short = min(view_w, view_h)
    sx, sy = float(terrain.size_x), float(terrain.size_y)
    ph = _PANEL_FRAC * short
    pw = ph * (sx / max(sy, 1e-9))
    if pw > 0.50 * view_w:
        pw = 0.50 * view_w
        ph = pw * (sy / max(sx, 1e-9))
    return pw, ph


def _flush_corner_offsets(half_w, half_h, pw, ph):
    """Zero-padding lower-left: panel left = -half_w, bottom = -half_h."""
    off_right = -half_w + 0.5 * pw
    off_up = -half_h + 0.5 * ph
    return off_right, off_up


def _panel_ndc(half_w, half_h, pw, ph, off_right, off_up):
    left = (off_right - 0.5 * pw) / max(half_w, 1e-9)
    bottom = (off_up - 0.5 * ph) / max(half_h, 1e-9)
    right = (off_right + 0.5 * pw) / max(half_w, 1e-9)
    top = (off_up + 0.5 * ph) / max(half_h, 1e-9)
    return left, bottom, right, top


def _quad_mesh(hx, hy):
    z = 0.0
    verts = [[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z]]
    indices = [0, 1, 2, 0, 2, 3]
    # V=0 at panel −Y (south / image bottom), V=1 at +Y (north / image top).
    uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    normals = [[0.0, 0.0, 1.0]] * 4
    return verts, indices, uvs, normals


def _arrow_mesh(scale):
    """Triangle in XY, tip along +Y (north on the panel)."""
    s = float(scale)
    verts = [[0.0, 0.70 * s, 0.0], [-0.42 * s, -0.52 * s, 0.0], [0.42 * s, -0.52 * s, 0.0]]
    normals = [[0.0, 0.0, 1.0]] * 3
    return verts, [0, 1, 2], normals


def spawn_minimap_hud(client, terrain, districts, scenery):
    """Create the static map panel and a tiny arrow. GUI only."""
    path, base = build_minimap_image(terrain, districts, scenery)
    try:
        tex_id = p.loadTexture(path, physicsClientId=client)
    except Exception:
        tex_id = -1

    half_w, half_h, aspect, fov_deg = _frustum_half_extents(client)
    pw, ph = _panel_size(terrain, half_w, half_h)
    off_r, off_u = _flush_corner_offsets(half_w, half_h, pw, ph)
    ndc_l, ndc_b, _, _ = _panel_ndc(half_w, half_h, pw, ph, off_r, off_u)
    verts, indices, uvs, normals = _quad_mesh(0.5 * pw, 0.5 * ph)
    vis = p.createVisualShape(
        p.GEOM_MESH,
        vertices=verts,
        indices=indices,
        uvs=uvs,
        normals=normals,
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
        specularColor=[0.0, 0.0, 0.0],
        flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
        physicsClientId=client,
    )
    # Park far below the world until the first camera-space update.
    panel = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=_PARK,
        physicsClientId=client,
    )
    if tex_id >= 0:
        p.changeVisualShape(
            panel,
            -1,
            textureUniqueId=tex_id,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
            physicsClientId=client,
        )

    a_verts, a_idx, a_n = _arrow_mesh(0.085 * ph)
    a_vis = p.createVisualShape(
        p.GEOM_MESH,
        vertices=a_verts,
        indices=a_idx,
        normals=a_n,
        rgbaColor=_ARROW_RGB,
        specularColor=[0.0, 0.0, 0.0],
        flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
        physicsClientId=client,
    )
    arrow = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=a_vis,
        basePosition=_PARK,
        physicsClientId=client,
    )
    hud = {
        "panel": panel,
        "arrow": arrow,
        "pw": pw,
        "ph": ph,
        "off_right": off_r,
        "off_up": off_u,
        "size_x": float(terrain.size_x),
        "size_y": float(terrain.size_y),
        "tex_id": tex_id,
        "base_rgba": base,
        "map_path": path,
    }
    print(
        f"[MINIMAP] HUD panel {pw:.2f}x{ph:.2f} m at {_HUD_DIST:.2f} m  "
        f"fov={fov_deg:.1f} aspect={aspect:.3f}  "
        f"NDC left={ndc_l:.3f} bottom={ndc_b:.3f} "
        f"(flush lower-left, want -1,-1)"
    )
    return hud


def update_minimap_hud(client, hud, eye, yaw, pitch, camera_basis):
    """Re-place the panel flush to the lower-left of the current view; move the arrow."""
    eye = np.asarray(eye, dtype=np.float64)
    forward, right, _ = camera_basis(yaw, pitch)
    cam_up, q_panel = _panel_axes(forward, right)
    half_w, half_h, aspect, fov_deg = _frustum_half_extents(client)
    off_right, off_up = _flush_corner_offsets(half_w, half_h, hud["pw"], hud["ph"])
    hud["off_right"] = off_right
    hud["off_up"] = off_up
    if not hud.get("_flush_logged"):
        ndc_l, ndc_b, _, _ = _panel_ndc(
            half_w, half_h, hud["pw"], hud["ph"], off_right, off_up
        )
        print(
            f"[MINIMAP] live frustum {2.0 * half_w:.2f}x{2.0 * half_h:.2f} m  "
            f"fov={fov_deg:.1f} aspect={aspect:.3f}  "
            f"NDC left={ndc_l:.3f} bottom={ndc_b:.3f}"
        )
        hud["_flush_logged"] = True
    center = (
        eye
        + forward * _HUD_DIST
        + right * off_right
        + cam_up * off_up
    )
    p.resetBasePositionAndOrientation(
        hud["panel"], center.tolist(), q_panel, physicsClientId=client
    )

    u, v = _world_to_uv(eye[0], eye[1], hud["size_x"], hud["size_y"])
    local_x = (u - 0.5) * hud["pw"]
    local_y = (v - 0.5) * hud["ph"]
    arrow_pos = center + right * local_x + cam_up * local_y - forward * 0.004

    yaw_r = math.radians(yaw)
    cz, sz = math.cos(yaw_r), math.sin(yaw_r)
    # Rotate the panel axes about its normal so +Y follows camera yaw (0 = north).
    east = cz * right + sz * cam_up
    north = -sz * right + cz * cam_up
    R_arrow = np.column_stack([east, north, -forward])
    p.resetBasePositionAndOrientation(
        hud["arrow"], arrow_pos.tolist(), _mat_to_quat(R_arrow), physicsClientId=client
    )
