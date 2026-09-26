"""Render a MuJoCo rollout with a translucent kinematic ghost of the Isaac reference.

Solid robot  = converted MuJoCo env (closed-loop policy playback, or open-loop action replay).
Ghost robot  = kinematic replay of the Isaac dump trajectory (root pose + joint angles).

Usage::

    uv run python scripts/render_ghost_compare.py \\
        BUNDLE DUMP POLICY OUT.mp4 [--mode closed|open] [--steps N]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from lab2mj.bundle import check_dump_joint_order, read_manifest
from lab2mj.env import MjEnv
from lab2mj.validate import dump_fixed_commands, reset_env_from_dump

SKYBOX = (
    '<asset><texture type="skybox" builtin="gradient" rgb1="0.45 0.58 0.78" '
    'rgb2="0.92 0.95 1.0" width="512" height="512"/></asset>'
)


def bundle_with_skybox(bundle: str, tmp_root: str) -> str:
    """Copy the bundle and inject a gradient skybox into scene.xml (cosmetic only)."""
    dst = Path(tmp_root) / "bundle"
    shutil.copytree(bundle, dst)
    scene = dst / "scene.xml"
    text = scene.read_text()
    text = text.replace("<worldbody>", SKYBOX + "<worldbody>", 1)
    scene.write_text(text)
    return str(dst)


GHOST_RGBA_TINT = np.array([0.35, 0.55, 1.0])  # blue-ish
GHOST_ALPHA = 0.35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("dump")
    ap.add_argument("policy")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=("closed", "open"), default="closed")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument(
        "--playback_fps",
        type=int,
        default=None,
        help="Output frame rate; default is the bundle's policy rate (lower gives slow motion).",
    )
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    dump = np.load(args.dump)
    n_steps = min(args.steps, dump["action_raw"].shape[0])

    with tempfile.TemporaryDirectory(prefix="ghost_render_") as tmp_root:
        render(args, dump, n_steps, bundle_with_skybox(args.bundle, tmp_root))


def render(args: argparse.Namespace, dump: np.lib.npyio.NpzFile, n_steps: int, bundle: str) -> None:
    # Pin every command term the way validate does: velocity terms to the CLI command,
    # pose terms to the recorded world-frame goal (strict mode would otherwise pin them
    # to zeros and a closed-loop nav rollout would walk to the origin by construction).
    manifest = read_manifest(bundle)
    check_dump_joint_order(manifest, dump)
    fixed_commands = dump_fixed_commands(manifest, dump)
    env = MjEnv(bundle, policy_path=args.policy, strict=True, command=fixed_commands)
    obs = reset_env_from_dump(env, dump)
    fps = 1.0 / env.policy_dt
    playback_fps = fps if args.playback_fps is None else float(args.playback_fps)

    # Ghost: same compiled model, translucent blue, kinematic only.
    ghost_model = mujoco.MjModel.from_xml_path(f"{bundle}/scene.xml")
    for g in range(ghost_model.ngeom):
        rgba = ghost_model.geom_rgba[g]
        rgba[:3] = 0.5 * rgba[:3] + 0.5 * GHOST_RGBA_TINT
        rgba[3] = GHOST_ALPHA if rgba[3] > 0 else 0.0
    # Hide the ghost's ground plane (keep only the robot).
    for g in range(ghost_model.ngeom):
        if ghost_model.geom_bodyid[g] == 0:
            ghost_model.geom_rgba[g, 3] = 0.0
    ghost_data = mujoco.MjData(ghost_model)

    # Bundles converted before the directional-light fix carry one point light near the
    # world origin, leaving anything meters away (all rough-terrain env origins, and any
    # robot mid-rollout) nearly black. Convert the scene lights to tilted directional
    # sun light and raise the headlight so existing bundles render lit.
    for i in range(env.model.nlight):
        env.model.light_type[i] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        env.model.light_dir[i] = np.array([0.25, 0.15, -0.95]) / np.linalg.norm([0.25, 0.15, -0.95])
    env.model.vis.headlight.ambient[:] = 0.35
    env.model.vis.headlight.diffuse[:] = 0.55
    env.model.vis.headlight.specular[:] = 0.2
    # Default 1024 shadow map stretched over a large scene renders blocky shadows.
    env.model.vis.quality.shadowsize = 8192

    env.model.vis.global_.offwidth = args.width
    env.model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 120.0, -12.0, 2.8

    rm = env.robot_map
    qa = rm.root_qpos_adr

    # Hoist the dump arrays: NpzFile.__getitem__ re-decompresses the whole member
    # on every access, ~7 accesses per frame inside the render loop otherwise.
    root_pos_w, root_quat_w, joint_pos = dump["root_pos_w"], dump["root_quat_w"], dump["joint_pos"]
    action_raw = dump["action_raw"]
    command_str = np.array2string(dump["command"], precision=1)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            f"{playback_fps:g}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            args.out,
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None

    label = "policy playback (closed loop)" if args.mode == "closed" else "action replay (open loop)"
    max_dq_seen = 0.0
    try:
        for t in range(n_steps):
            action = env.policy_action(obs) if args.mode == "closed" else action_raw[t]
            obs, _, _, _ = env.step(action)

            # Ghost kinematic pose from the dump at step t.
            ghost_data.qpos[qa : qa + 3] = root_pos_w[t]
            ghost_data.qpos[qa + 3 : qa + 7] = root_quat_w[t]
            ghost_data.qpos[rm.qpos_adr] = joint_pos[t]
            mujoco.mj_kinematics(ghost_model, ghost_data)

            # Divergence metrics (Isaac order both sides).
            dq = np.abs(env.data.qpos[rm.qpos_adr] - joint_pos[t])
            droot = env.data.qpos[qa : qa + 3] - root_pos_w[t]
            max_dq_seen = max(max_dq_seen, float(dq.max()))

            # Camera tracks the midpoint of both roots.
            mid = 0.5 * (env.data.qpos[qa : qa + 3] + root_pos_w[t])
            cam.lookat[:] = [mid[0], mid[1], mid[2] * 0.9]

            renderer.update_scene(env.data, camera=cam, scene_option=vopt)
            mujoco.mjv_addGeoms(ghost_model, ghost_data, vopt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
            frame = renderer.render()

            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            lines = [
                f"solid = MuJoCo ({label})   ghost = Isaac reference",
                f"t = {t * env.policy_dt:5.2f} s   cmd = {command_str}",
                f"max |dq| now {dq.max():6.4f} rad   (peak {max_dq_seen:6.4f})",
                f"root offset  dx {droot[0]:+6.3f}  dy {droot[1]:+6.3f}  dz {droot[2]:+6.3f} m",
            ]
            y = 12
            for line in lines:
                draw.text((14, y + 1), line, fill=(0, 0, 0))
                draw.text((13, y), line, fill=(255, 255, 255))
                y += 18
            ffmpeg.stdin.write(np.asarray(img).tobytes())

        ffmpeg.stdin.close()
        ffmpeg.wait()
    finally:
        renderer.close()
        if ffmpeg.poll() is None:  # exception path: stop ffmpeg instead of leaking it
            ffmpeg.kill()
            ffmpeg.wait()
    print(f"wrote {args.out}  ({n_steps} frames @ {playback_fps:g} fps, peak |dq| {max_dq_seen:.4f} rad)")


if __name__ == "__main__":
    main()
