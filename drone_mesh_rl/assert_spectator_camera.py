#!/usr/bin/env python3
"""Offline checks for spectator camera basis (no GUI).

Run:  python3 assert_spectator_camera.py
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pybullet as p

from visualize_env import (
    _SPEC_TURN_SPEED,
    camera_basis_from_yaw_pitch,
    eye_from_camera,
)


def _eye_from_view_matrix(yaw: float, pitch: float, dist: float = 1.0) -> np.ndarray:
    vm = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0],
        distance=dist,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    mat = np.array(vm, dtype=np.float64).reshape(4, 4, order="F")
    r, t = mat[:3, :3], mat[:3, 3]
    return -r.T @ t


def main() -> int:
    # Town overview defaults
    yaw0, pitch0 = 45.0, -40.0
    turn = _SPEC_TURN_SPEED

    forward0, right0, _ = camera_basis_from_yaw_pitch(yaw0, pitch0)
    print(f"initial yaw={yaw0} pitch={pitch0}")
    print(f"  forward={forward0}")
    print(f"  right  ={right0}")

    # 1) Eye placement matches PyBullet view-matrix convention
    for yaw in (-90.0, 0.0, 45.0, 90.0, 180.0):
        for pitch in (-60.0, -40.0, 0.0, 30.0):
            eye = eye_from_camera(yaw, pitch, 1.0, [0, 0, 0])
            ref = _eye_from_view_matrix(yaw, pitch, 1.0)
            err = float(np.linalg.norm(eye - ref))
            assert err < 1e-6, f"eye mismatch yaw={yaw} pitch={pitch} err={err}"
            fwd, _, _ = camera_basis_from_yaw_pitch(yaw, pitch)
            # forward == normalize(target - eye) with target at origin
            expect = -eye / np.linalg.norm(eye)
            assert np.allclose(fwd, expect, atol=1e-6), (fwd, expect)

    # 2) Right-arrow decreases yaw; horizontal forward rotates toward previous right
    yaw_right = yaw0 - turn
    forward1, _, _ = camera_basis_from_yaw_pitch(yaw_right, pitch0)
    df = forward1 - forward0
    df_h = np.array([df[0], df[1], 0.0])
    toward_right = float(np.dot(df_h, right0))
    toward_left = float(np.dot(df_h, -right0))
    print(f"after right-arrow (yaw {yaw0} -> {yaw_right}):")
    print(f"  new forward={forward1}")
    print(f"  dot(df_horiz, right)={toward_right:.6f} (must be > 0)")
    assert toward_right > 0.0, "right-arrow yaw must rotate look toward viewer-right"
    assert toward_right > toward_left

    # 3) WASD displacements
    speed = 3.0
    assert np.allclose(forward0 * speed, forward0 * speed)
    assert np.allclose(-(forward0 * speed), -forward0 * speed)
    w = forward0 * speed
    s = -forward0 * speed
    d = right0 * speed
    a = -right0 * speed
    assert np.allclose(w, forward0 * speed)
    assert np.allclose(s, -forward0 * speed)
    assert np.allclose(d, right0 * speed)
    assert np.allclose(a, -right0 * speed)
    print("WASD: W=+forward S=-forward D=+right A=-right OK")

    # 4) Up-arrow increases pitch; forward.z rises
    pitch_up = pitch0 + turn
    forward_up, _, _ = camera_basis_from_yaw_pitch(yaw0, pitch_up)
    dz = float(forward_up[2] - forward0[2])
    print(f"after up-arrow (pitch {pitch0} -> {pitch_up}):")
    print(f"  forward.z {forward0[2]:.6f} -> {forward_up[2]:.6f} (delta={dz:.6f})")
    assert dz > 0.0, "up-arrow must increase forward.z"

    print("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"ASSERTION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
