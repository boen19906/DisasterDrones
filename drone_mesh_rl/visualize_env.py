"""
visualize_env.py — Lightweight 3D spectator over real USGS terrain.

Default: loads the single elevation GeoTIFF in drone_mesh_rl/data/ as the
PyBullet heightfield, then tiles that patch 2x2 in memory (terrain body
only, no town, roads, or props). --procedural restores the two-downtown
metro. Camera-only loop (no drones).

Usage:
  cd drone_mesh_rl && python3 visualize_env.py
  cd drone_mesh_rl && python3 visualize_env.py --headless   # load/spawn check only
  cd drone_mesh_rl && python3 visualize_env.py --procedural # old metro scene
"""


import math
import sys
import time
import argparse
import numpy as np
import pybullet as p

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from envs.terrain import Terrain, find_dem_file, spawn_terrain_in_pybullet
from envs.town import (
    MetroLayout,
    connect_pybullet,
    spawn_town_in_pybullet,
    frame_town_camera,
)

# World / downtown defaults
_WORLD_SIZE = 3250.0
_DOWNTOWN_SIZE = 250.0
_HEIGHT_RES = 8.0
# Opposite sides of center with a wide green gap (~1350 m between facing edges)
_CITY_A = (-800.0, 0.0)
_CITY_B = (800.0, 0.0)

# Spectator fly defaults (meters / degrees per key-poll at target FPS)
# Raised so the multi-km green gap is crossable without changing key bindings.
_SPEC_MOVE_SPEED = 18.0
_SPEC_FAST_MULT = 3.0  # Left Ctrl sprint only (Left Shift is descend)
_SPEC_TURN_SPEED = 2.5
_SPEC_FLY_DIST = 1.0  # short boom so look-around feels FPS-like
_PITCH_MIN, _PITCH_MAX = -89.0, 89.0


def parse_args():
    parser = argparse.ArgumentParser(description="USGS terrain / metro spectator viewer")
    parser.add_argument(
        "--env_size",
        type=float,
        default=_DOWNTOWN_SIZE,
        help="Each downtown footprint size in meters (world is fixed ~3250 m)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Display FPS target")
    parser.add_argument("--seed", type=int, default=42, help="World seed (--procedural)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip GUI: connect p.DIRECT, print timing/body counts, then exit",
    )
    parser.add_argument(
        "--dem",
        type=str,
        default=None,
        help="Elevation GeoTIFF (default: the single .tif in drone_mesh_rl/data/)",
    )
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="Load the procedural two-downtown metro instead of the DEM",
    )
    return parser.parse_args()



def camera_basis_from_yaw_pitch(yaw_deg, pitch_deg):
    """Return (forward, right, world_up) matching PyBullet's Z-up debug camera.

    Verified against pybullet.computeViewMatrixFromYawPitchRoll and Bullet
    SimpleCamera (upAxis=2): the eye orbits the target as
      eye = target + dist * (sin(yaw)*cos(pitch), -cos(yaw)*cos(pitch), -sin(pitch))
    so forward = normalize(target - eye) is
      (-sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)).

    Yaw: decreasing yaw rotates look toward viewer-right.
    Pitch: increasing pitch tilts look upward (forward.z increases).
    Right: cross(forward, world_up) — same as Bullet's look-at side vector.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    forward = np.array([-sy * cp, cy * cp, sp], dtype=np.float64)
    fn = np.linalg.norm(forward)
    if fn > 1e-12:
        forward /= fn

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        # Near gimbal lock: horizontal right from yaw alone
        right = np.array([cy, sy, 0.0], dtype=np.float64)
        rn = np.linalg.norm(right)
    right /= rn
    return forward, right, world_up


def eye_from_camera(yaw_deg, pitch_deg, dist, target):
    """Eye position for PyBullet resetDebugVisualizerCamera parameters."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    target = np.asarray(target, dtype=np.float64)
    offset = dist * np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            -math.cos(yaw) * math.cos(pitch),
            -math.sin(pitch),
        ],
        dtype=np.float64,
    )
    return target + offset


def apply_spectator_camera(client, eye, yaw, pitch, dist):
    """Place the camera from eye + yaw/pitch using the shared basis."""
    eye = np.asarray(eye, dtype=np.float64)
    forward, _, _ = camera_basis_from_yaw_pitch(yaw, pitch)
    target = eye + forward * dist
    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target.tolist(),
        physicsClientId=client,
    )
    return target


def read_debug_camera(client):
    """Return yaw, pitch, dist, target, eye from the active PyBullet debug camera."""
    info = p.getDebugVisualizerCamera(physicsClientId=client)
    yaw = float(info[8])
    pitch = float(info[9])
    dist = float(info[10])
    target = np.array(info[11], dtype=np.float64)
    eye = eye_from_camera(yaw, pitch, dist, target)
    return yaw, pitch, dist, target, eye


def enter_spectator_fly(client, yaw, pitch, dist, target):
    """Collapse the orbit boom to a short fly distance while preserving the eye pose."""
    eye = eye_from_camera(yaw, pitch, dist, target)
    new_dist = _SPEC_FLY_DIST
    apply_spectator_camera(client, eye, yaw, pitch, new_dist)
    return yaw, pitch, new_dist, eye


def _key_down(keys, code):
    return bool(keys.get(code, 0) & p.KEY_IS_DOWN)


def _key_triggered(keys, code):
    return bool(keys.get(code, 0) & p.KEY_WAS_TRIGGERED)


def update_spectator_camera(client, keys, yaw, pitch, dist, eye, move_speed=None):
    """Minecraft-style spectator: WASD look-relative, arrows look, Space/Shift vertical."""
    speed = _SPEC_MOVE_SPEED if move_speed is None else float(move_speed)
    # Sprint: Left Ctrl only — Left Shift is descend, not faster
    if _key_down(keys, p.B3G_CONTROL):
        speed *= _SPEC_FAST_MULT

    move_fwd = 0.0
    move_right = 0.0
    move_up = 0.0
    if _key_down(keys, ord("w")) or _key_down(keys, ord("W")):
        move_fwd += speed
    if _key_down(keys, ord("s")) or _key_down(keys, ord("S")):
        move_fwd -= speed
    if _key_down(keys, ord("d")) or _key_down(keys, ord("D")):
        move_right += speed
    if _key_down(keys, ord("a")) or _key_down(keys, ord("A")):
        move_right -= speed
    if _key_down(keys, p.B3G_SPACE) or _key_down(keys, ord(" ")):
        move_up += speed
    if _key_down(keys, p.B3G_SHIFT):
        move_up -= speed

    # Signs match camera_basis: -yaw = look right, +pitch = look up
    dyaw = 0.0
    dpitch = 0.0
    if _key_down(keys, p.B3G_RIGHT_ARROW):
        dyaw -= _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_LEFT_ARROW):
        dyaw += _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_UP_ARROW):
        dpitch += _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_DOWN_ARROW):
        dpitch -= _SPEC_TURN_SPEED

    if dyaw != 0.0 or dpitch != 0.0:
        yaw = (yaw + dyaw) % 360.0
        pitch = float(np.clip(pitch + dpitch, _PITCH_MIN, _PITCH_MAX))

    if move_fwd != 0.0 or move_right != 0.0 or move_up != 0.0:
        forward, right, world_up = camera_basis_from_yaw_pitch(yaw, pitch)
        eye = np.asarray(eye, dtype=np.float64).copy()
        # Translate eye (and thus target) without changing yaw/pitch
        eye += forward * move_fwd + right * move_right + world_up * move_up

    apply_spectator_camera(client, eye, yaw, pitch, dist)
    return yaw, pitch, dist, eye


def build_world(client, env_size, seed):
    """Generate two downtowns + terrain and spawn them once in PyBullet."""
    downtown = float(env_size)
    world_size = _WORLD_SIZE
    city_centers = [_CITY_A, _CITY_B]

    t0 = time.perf_counter()
    metro = MetroLayout(
        downtown_size=downtown,
        seed=seed,
        altitude_cap=40.0,
        city_a_center=_CITY_A,
        city_b_center=_CITY_B,
    )
    t_metro = time.perf_counter() - t0

    t0 = time.perf_counter()
    terrain = Terrain(
        size_x=world_size,
        size_y=world_size,
        resolution=_HEIGHT_RES,
        seed=seed,
        city_size=downtown,
        city_centers=city_centers,
        blend_width=40.0,
        color_blend_width=55.0,
        meadow_amp=0.30,
        grass_cell=120.0,
        grass_amp=2.2,
        region_cell=220.0,
        hill_amp_min=5.0,
        hill_amp_max=11.0,
        n_hill_clusters=(5, 10),
        hill_sigma_min=90.0,
        hill_sigma_max=160.0,
        hill_fade=55.0,
        max_slope=0.12,
        obstacle_boxes=metro.boxes,
    )
    t_terrain = time.perf_counter() - t0

    t0 = time.perf_counter()
    metro.apply_ground_heights(terrain)
    t_apply = time.perf_counter() - t0

    t0 = time.perf_counter()
    spawn_town_in_pybullet(client, metro, terrain, downtown)
    t_spawn = time.perf_counter() - t0

    n_roads = len(metro.roads)
    n_dash_bodies = 0  # dashes are baked into the ground texture
    n_bodies = p.getNumBodies(physicsClientId=client)
    hf_bytes = int(terrain.grid_x) * int(terrain.grid_y) * 4
    print(
        f"[TIMING] MetroLayout={t_metro:.3f}s  Terrain={t_terrain:.3f}s  "
        f"apply_ground_heights={t_apply:.3f}s  spawn={t_spawn:.3f}s"
    )
    print(
        f"[BODIES] roads={n_roads}  dash_bodies={n_dash_bodies}  "
        f"total={n_bodies}  "
        f"height_grid={terrain.grid_x}x{terrain.grid_y}  "
        f"hf_bytes={hf_bytes}  "
        f"(buildings={len(metro.boxes)}, details={len(metro.detail_parts)}, "
        f"texture_dashes={len(metro.road_marks)})"
    )
    return metro, terrain, world_size


def build_dem_world(client, dem_path):
    """Load the elevation GeoTIFF and spawn it as the only body (no town)."""
    t0 = time.perf_counter()
    terrain = Terrain.from_dem(dem_path)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    spawn_terrain_in_pybullet(client, terrain)
    t_spawn = time.perf_counter() - t0

    info = terrain.dem_info
    rows, cols = info["raster_shape"]
    px_x, px_y = info["native_pixel"]
    ncx, ncy = info["native_cell_m"]
    unit = "deg" if info["crs_is_geographic"] else "CRS units"
    hf_bytes = int(terrain.grid_x) * int(terrain.grid_y) * 4
    zmin = terrain.z_offset
    zmax = terrain.z_offset + float(terrain.heightmap.max())
    ne_lon, ne_lat = info["transform"] * (cols, 0)
    print(f"[DEM] source={info['name']}  CRS={info['crs']}  load={t_load:.3f}s  spawn={t_spawn:.3f}s")
    print(
        f"[DEM] raster shape={rows}x{cols} (rows x cols)  native pixel="
        f"{px_x:.6g} x {px_y:.6g} {unit} (~{ncx:.2f} m E-W x {ncy:.2f} m N-S)"
    )
    print(f"[DEM] nodata filled={info['nodata_filled']} (nodata value {info['nodata_value']})")
    tile_y, tile_x = getattr(terrain, "dem_tile", (1, 1))
    print(
        f"[DEM] grid={terrain.grid_x}x{terrain.grid_y} (x cols x y rows, "
        f"downsample x{info['downsample']}, tile {tile_x}x{tile_y})  "
        f"cell={terrain.resolution_x:.2f} x {terrain.resolution_y:.2f} m  "
        f"world={terrain.size_x:.1f} x {terrain.size_y:.1f} m"
    )
    print(
        f"[DEM] elevation min={zmin:.2f} m  max={zmax:.2f} m  relief={zmax - zmin:.2f} m  "
        f"(world z = elevation - {zmin:.2f})"
    )
    print(
        f"[DEM] heightfield bytes={hf_bytes} (< 1_000_000)  "
        f"bodies={p.getNumBodies(physicsClientId=client)}"
    )
    print(
        f"[DEM] orientation: world (+X,+Y) corner = raster row 0, col {cols - 1} "
        f"(NE corner, ~{ne_lon:.5f}, {ne_lat:.5f}); north = +Y, east = +X"
    )
    return terrain


def dem_move_speed(terrain):
    """Meters per key poll: long side in ~13 s at 30 FPS, ~4 s with Left Ctrl."""
    return max(1.0, max(terrain.size_x, terrain.size_y) / 400.0)


def _debug_far_plane(client, default=1000.0):
    """Near/far from the active debug projection; GUI far is typically ~1000 m."""
    try:
        proj = p.getDebugVisualizerCamera(physicsClientId=client)[3]
    except Exception:
        proj = None
    if not proj or proj[0] <= 0:
        return float(default)
    P = np.array(proj, dtype=np.float64).reshape(4, 4).T
    far = P[2, 3] / (P[2, 2] + 1.0)
    if not np.isfinite(far) or far <= 0:
        return float(default)
    return float(far)


def frame_dem_camera(client, terrain):
    """Downward overview of the tiled DEM, kept inside the GUI far plane.

    A 2x2 USGS world is ~3 km across, but the OpenGL GUI far clip is fixed
    near 1000 m. Fitting every corner (the old approach) pushed the eye
    past that clip at pitch -89, so the heightfield vanished. Aim at the
    origin from the south, pitch about -40, and keep the look-at point
    well inside the far plane so the land is visible on launch.
    """
    far = _debug_far_plane(client)
    top = float(terrain.heightmap.max())
    target = np.array([0.0, 0.0, max(8.0, 0.4 * top)], dtype=np.float64)
    yaw = 0.0
    pitch = -40.0
    long_axis = max(float(terrain.size_x), float(terrain.size_y))
    # Ideal orbit would be ~1.5x the long axis; clamp so the target stays
    # in front of the far clip (otherwise the map is not drawn).
    dist = min(1.5 * long_axis, 0.72 * far)
    dist = max(dist, 250.0)

    eye = eye_from_camera(yaw, pitch, dist, target)
    if eye[2] <= top + 15.0:
        lift = (top + 25.0) - float(eye[2])
        target = target.copy()
        target[2] += lift
        eye = eye_from_camera(yaw, pitch, dist, target)

    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target.tolist(),
        physicsClientId=client,
    )
    print(
        f"[CAMERA] overview yaw={yaw:g} pitch={pitch:g} dist={dist:.0f} m  "
        f"eye=({eye[0]:.0f}, {eye[1]:.0f}, {eye[2]:.0f})  "
        f"target=({target[0]:.0f}, {target[1]:.0f}, {target[2]:.0f})  "
        f"far={far:.0f} m"
    )
    assert eye[2] > top, "spectator must start above the highest terrain"
    assert dist < far, "overview target must sit in front of the GUI far plane"
    return yaw, pitch, dist, target


def clear_world(client):
    """Remove all bodies so the town can be rebuilt on reset."""
    n = p.getNumBodies(physicsClientId=client)
    body_ids = [p.getBodyUniqueId(i, physicsClientId=client) for i in range(n)]
    for body_id in body_ids:
        try:
            p.removeBody(body_id, physicsClientId=client)
        except Exception:
            pass


def main():
    args = parse_args()
    # macOS has no DISPLAY (X11); only --headless opts out of the GUI.
    gui = not args.headless
    print("=" * 65)
    if args.procedural:
        print(" [3D METRO VIEWER] Two-downtown spectator")
    else:
        print(" [3D TERRAIN VIEWER] USGS elevation spectator (2x2 tiled, no city)")
    if not gui:
        print(" --headless: timing/body-count check (p.DIRECT).")
    else:
        print(" Click the 3D viewport first so keys reach PyBullet.")
        print(" Keys:")
        print("   W/A/S/D  = fly forward/left/back/right (look-relative)")
        print("   Arrows   = look (right/left/up/down)")
        print("   Space    = up | Left Shift = down | Left Ctrl = sprint")
        print("   C        = toggle spectator | P = pause | R = reset overview")
        print("   Q / ESC  = quit")
    print("=" * 65)

    client = connect_pybullet(gui=gui, shadows=False)
    if gui:
        # Belt-and-suspenders: force our GUI flags again right after connect.
        p.configureDebugVisualizer(
            p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=client
        )
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client)

    use_dem = not args.procedural
    dem_path = None
    move_speed = _SPEC_MOVE_SPEED
    if use_dem:
        dem_path = args.dem or find_dem_file()
        terrain = build_dem_world(client, dem_path)
        move_speed = dem_move_speed(terrain)
        print(
            f"[SPECTATOR] move speed {move_speed:.2f} m/poll "
            f"(x{_SPEC_FAST_MULT:g} with Left Ctrl) at {args.fps:g} FPS"
        )
    else:
        town, terrain, world_size = build_world(client, args.env_size, args.seed)
        n_buildings = len(town.boxes)
        print(
            f"[METRO] Loaded {n_buildings} building/rubble/overpass boxes "
            f"(downtown {args.env_size:.0f}m x2, world {world_size:.0f}m, "
            f"grid {terrain.grid_x}x{terrain.grid_y})."
        )
        if getattr(town, "gas_station", None):
            g = town.gas_station
            print(
                f"[METRO] Gas station beside connector near "
                f"({g['lot'][0]:.0f}, {g['lot'][1]:.0f}); "
                f"{len(town.road_marks)} center dashes in ground texture."
            )

    if not gui:
        print("[VISUALIZER] Headless check done (GUI not opened).")
        try:
            p.disconnect(physicsClientId=client)
        except Exception:
            pass
        return

    def frame_overview():
        """Reset the overview camera; returns (yaw, pitch, dist, target).

        The GUI applies resetDebugVisualizerCamera asynchronously, so reading
        the camera straight back can return the previous pose.
        """
        if use_dem:
            return frame_dem_camera(client, terrain)
        # Overview near the west downtown
        frame_town_camera(client, args.env_size * 2.5)
        cam = (
            45.0,
            -35.0,
            max(220.0, args.env_size * 1.2),
            np.array([_CITY_A[0], _CITY_A[1], 8.0]),
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=cam[2],
            cameraYaw=cam[0],
            cameraPitch=cam[1],
            cameraTargetPosition=cam[3].tolist(),
            physicsClientId=client,
        )
        return cam

    spectator = True
    yaw, pitch, dist, target = frame_overview()
    yaw, pitch, dist, eye = enter_spectator_fly(client, yaw, pitch, dist, target)
    print("[SPECTATOR] ON — click the 3D viewport, then fly with WASD / Space+Shift / arrows.")

    dt_target = 1.0 / args.fps
    paused = False
    seed = args.seed

    try:
        while True:
            loop_start = time.time()
            keys = p.getKeyboardEvents(physicsClientId=client)

            if ord("q") in keys or ord("Q") in keys or 27 in keys:
                print("[EXIT] User requested exit.")
                break
            if _key_triggered(keys, ord("p")) or _key_triggered(keys, ord("P")):
                paused = not paused
                print(f"[{'PAUSED' if paused else 'RESUMED'}]")
            if _key_triggered(keys, ord("r")) or _key_triggered(keys, ord("R")):
                clear_world(client)
                if use_dem:
                    print("[RESET] Reloading DEM and reframing...")
                    terrain = build_dem_world(client, dem_path)
                else:
                    print("[RESET] Rebuilding metro...")
                    seed = int(seed) + 1
                    town, terrain, world_size = build_world(client, args.env_size, seed)
                    print(f"[METRO] Loaded {len(town.boxes)} boxes (seed={seed}).")
                yaw, pitch, dist, target = frame_overview()
                spectator = True
                yaw, pitch, dist, eye = enter_spectator_fly(
                    client, yaw, pitch, dist, target
                )
                continue
            if _key_triggered(keys, ord("c")) or _key_triggered(keys, ord("C")):
                spectator = not spectator
                if spectator:
                    yaw, pitch, dist, target, eye = read_debug_camera(client)
                    yaw, pitch, dist, eye = enter_spectator_fly(
                        client, yaw, pitch, dist, target
                    )
                    print("[SPECTATOR] ON")
                else:
                    print("[SPECTATOR] OFF — mouse orbit available")

            if not paused and spectator:
                yaw, pitch, dist, eye = update_spectator_camera(
                    client, keys, yaw, pitch, dist, eye, move_speed=move_speed
                )

            # Idle: no physics stepping, no drone/network updates
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, dt_target - elapsed)
            time.sleep(sleep_time)

    except (KeyboardInterrupt, p.error):
        pass
    finally:
        try:
            p.disconnect(physicsClientId=client)
        except Exception:
            pass
        print("[VISUALIZER] Closed successfully.")


if __name__ == "__main__":
    main()
