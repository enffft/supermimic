"""Play a checkpoint in the robot/task environment saved with that checkpoint.

Examples:
  python scripts/play.py --ckpt logs/g1/ckpt_latest.pt
  python scripts/play.py --ckpt logs/go2/ckpt_latest.pt --vx 0.3 --vy 0 --yaw-rate 0
  python scripts/play.py --ckpt logs/m1/ckpt_latest.pt --clips a_wheel_000000 a_wheel_000002
  python scripts/play.py --ckpt logs/m1/ckpt_latest.pt --num-envs 6

Motion playback uses restart semantics: a timeout or physical failure resets the
physics state and policy history to the same selected clip's first frame.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.spec import get_robot_spec


M1_PLAY_CLIPS = (
    "a_wheel_000000",
    "a_wheel_000002",
    "b_arm_000006",
    "c_assisted_000029",
    "c_assisted_000039",
    "c_assisted_000149",
)


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal) to the viewer's policy(obs) protocol."""

    def __init__(self, policy: Policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


class _AllEnvViserViewer(ViserPlayViewer):
    """Use mjlab's viewer and scene, with all requested comparison envs visible."""

    def setup(self) -> None:
        super().setup()
        self._scene.show_all_envs = True
        self._scene.request_update()


def _as_path(value) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _load_checkpoint(path: Path, device: str) -> tuple[Policy, dict, dict]:
    # Keep the full training checkpoint (WM and optimizer states) on CPU; only the
    # Actor weights needed by the viewer are copied to the inference device.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: checkpoint has no saved training config")
    policy_cfg = config.get("policy")
    if not isinstance(policy_cfg, dict):
        raise ValueError(f"{path}: checkpoint has no saved Policy config")
    policy_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in policy_cfg.items()}
    policy = Policy(PolicyCfg(**policy_cfg)).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, config, ckpt


def _saved_motion_inputs(
    config: dict,
    robot_name: str,
    npz_dir: Path | None,
    clips: list[str] | None,
) -> tuple[list[str], bool]:
    data_cfg = config.get("data", {})
    if data_cfg.get("library"):
        library = _as_path(data_cfg["library"])
        if not library.is_file():
            raise FileNotFoundError(f"checkpoint Motion Library does not exist: {library}")
        if npz_dir is not None:
            raise ValueError("--npz-dir cannot override a packed checkpoint Motion Library")
        return [str(library)], True

    saved_names = data_cfg.get("clips")
    names = clips if clips is not None else saved_names
    if clips is not None and not clips:
        raise ValueError("--clips must contain at least one clip name")

    if npz_dir is not None:
        directory = npz_dir
    else:
        directory = _as_path(data_cfg.get("npz_dir", "data/npz/lafan1"))
        if robot_name == "m1_piper" and directory == Path("data/npz/lafan1"):
            # M1 has a separate validated bank and never consumes LAFAN1 files.
            directory = Path("data/npz/m1_piper")

    if robot_name == "m1_piper" and names is None:
        names = list(M1_PLAY_CLIPS)
    manifest = data_cfg.get("manifest")
    manifest_path = _as_path(manifest) if manifest else None
    files = [str(p) for p in discover_npz(directory, names=names, manifest=manifest_path)]
    if not files:
        raise FileNotFoundError(f"no saved Motion Data found under {directory} for clips={names}")
    return files, False


def _asset_path(robot_name: str) -> Path:
    spec = get_robot_spec(robot_name)
    if spec.model_path is not None:
        return spec.model_path
    from supermimic.data.motion_convert import g1_xml_path

    return g1_xml_path()


def main(
    ckpt: Path,
    npz_dir: Path | None = None,
    clips: list[str] | None = None,
    num_envs: int | None = None,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool | None = None,
    loop_mode: Literal["restart"] = "restart",
    vx: float | None = None,
    vy: float | None = None,
    yaw_rate: float | None = None,
    velocity_arrow: bool = True,
    actual_velocity_arrow: bool = False,
) -> None:
    policy, config, checkpoint_data = _load_checkpoint(ckpt, device)
    env_cfg = config.get("env", {})
    # Robot and task come only from the saved checkpoint. There are intentionally
    # no robot/task CLI overrides that could load weights into the wrong model.
    robot_name = get_robot_spec(str(env_cfg.get("robot_name", "g1"))).name
    task_name = str(env_cfg.get("task_name", "motion_tracking"))
    if robot_name == "go2":
        saved_actuator = checkpoint_data.get("actuator_model_version", config.get("actuator_model_version"))
        current_actuator = get_robot_spec(robot_name).actuator_model_version
        if saved_actuator != current_actuator:
            print(
                "[WARN] Go2 Actor was saved with a different or unspecified actuator model "
                f"(checkpoint={saved_actuator!r}, current={current_actuator!r}); "
                "this Play run is an evaluation on changed physics, not a compatible full resume."
            )
    if task_name not in ("motion_tracking", "velocity_command", "world_point_reach"):
        raise ValueError(f"unsupported checkpoint task {task_name!r}")
    if task_name == "velocity_command" and robot_name != "go2":
        raise ValueError(f"checkpoint mismatch: {robot_name} cannot use velocity_command")
    if robot_name == "go2" and task_name != "velocity_command":
        raise ValueError("Go2 checkpoint must use velocity_command")
    if task_name == "world_point_reach" and robot_name != "m1_piper":
        raise ValueError("world_point_reach checkpoint must use M1-Piper")

    data_files: list[str] = []
    compact_library = False
    selected_names: list[str] | None = None
    if task_name == "motion_tracking":
        data_files, compact_library = _saved_motion_inputs(config, robot_name, npz_dir, clips)
        selected_names = [Path(p).stem for p in data_files] if not compact_library else None
        if compact_library:
            selected_names = clips if clips is not None else config.get("data", {}).get("clips")
        if num_envs is None:
            if robot_name == "m1_piper" or clips is not None:
                num_envs = len(selected_names) if selected_names else 1
            else:
                num_envs = 1
    else:
        if clips is not None or npz_dir is not None:
            raise ValueError("--clips and --npz-dir apply only to motion_tracking checkpoints")
        if any(value is not None for value in (vx, vy, yaw_rate)):
            fixed_command = (float(vx or 0.0), float(vy or 0.0), float(yaw_rate or 0.0))
        else:
            fixed_command = None
        if num_envs is None:
            num_envs = 1

    if num_envs is None or num_envs < 1:
        raise ValueError("num_envs must be positive")
    if loop_mode != "restart":
        raise ValueError("only loop_mode='restart' is supported")

    if ghost is None:
        ghost = task_name in ("motion_tracking", "world_point_reach")
    history_length = int(env_cfg.get("history_length", policy.cfg.history))
    future_frames = int(env_cfg.get("future_frames", 10))
    anchor_obs = bool(env_cfg.get("anchor_obs", robot_name == "m1_piper" and task_name == "motion_tracking"))
    anchor_orientation_obs = bool(env_cfg.get("anchor_orientation_obs", False))
    play_env_args = SuperMimicEnvCfgArgs(
        npz_files=data_files,
        robot_name=robot_name,
        task_name=task_name,
        num_envs=num_envs,
        future_frames=future_frames,
        history_length=history_length,
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        anchor_orientation_obs=anchor_orientation_obs,
        compact_library=compact_library,
        velocity_gaussian_noise_std=0.0,
        velocity_ou_env_fraction=0.0,
        velocity_ou_noise_std=0.0,
        velocity_curriculum=bool(env_cfg.get("velocity_curriculum", True)),
        velocity_curriculum_initial_stage=int(env_cfg.get("velocity_curriculum_initial_stage", 0)),
        velocity_curriculum_fixed_stage=env_cfg.get("velocity_curriculum_fixed_stage"),
        velocity_stage_a_vx_range=tuple(env_cfg.get("velocity_stage_a_vx_range", (0.15, 0.55))),
        velocity_stage_b_vx_range=tuple(env_cfg.get("velocity_stage_b_vx_range", (-0.2, 0.7))),
        velocity_stage_b_vy_range=tuple(env_cfg.get("velocity_stage_b_vy_range", (-0.15, 0.15))),
        velocity_stage_b_yaw_range=tuple(env_cfg.get("velocity_stage_b_yaw_range", (-0.4, 0.4))),
        velocity_stage_c_vx_range=tuple(env_cfg.get("velocity_stage_c_vx_range", (-0.4, 1.0))),
        velocity_stage_c_vy_range=tuple(env_cfg.get("velocity_stage_c_vy_range", (-0.3, 0.3))),
        velocity_stage_c_yaw_range=tuple(env_cfg.get("velocity_stage_c_yaw_range", (-0.8, 0.8))),
        velocity_stage_a_static_fraction=float(env_cfg.get("velocity_stage_a_static_fraction", 0.15)),
        velocity_stage_b_keep_a_fraction=float(env_cfg.get("velocity_stage_b_keep_a_fraction", 0.35)),
        velocity_stage_c_keep_a_fraction=float(env_cfg.get("velocity_stage_c_keep_a_fraction", 0.25)),
        velocity_stage_c_keep_b_fraction=float(env_cfg.get("velocity_stage_c_keep_b_fraction", 0.25)),
        velocity_resample_seconds=float(env_cfg.get("velocity_resample_seconds", 10.0)),
        velocity_fixed_command=fixed_command if task_name == "velocity_command" else None,
        velocity_debug_vis=bool(velocity_arrow) if task_name == "velocity_command" else False,
        velocity_show_actual_velocity=actual_velocity_arrow,
        velocity_fall_height=float(env_cfg.get("velocity_fall_height", 0.17)),
        velocity_fall_tilt_deg=float(env_cfg.get("velocity_fall_tilt_deg", 70.0)),
        m1_fall_height=float(env_cfg.get("m1_fall_height", 0.20)),
        m1_success_distance=float(env_cfg.get("m1_success_distance", 0.08)),
        m1_near_target_fraction=float(env_cfg.get("m1_near_target_fraction", 0.60)),
        m1_near_x=tuple(env_cfg.get("m1_near_x", (0.05, 0.30))),
        m1_near_y=tuple(env_cfg.get("m1_near_y", (-0.20, 0.20))),
        m1_near_z=tuple(env_cfg.get("m1_near_z", (-0.15, 0.15))),
        m1_far_x=tuple(env_cfg.get("m1_far_x", (1.00, 1.40))),
        m1_far_y=tuple(env_cfg.get("m1_far_y", (-0.40, 0.40))),
        m1_far_z=tuple(env_cfg.get("m1_far_z", (-0.10, 0.25))),
        debug_vis=bool(ghost) if task_name in ("motion_tracking", "world_point_reach") else False,
    )

    raw_env = make_env(play_env_args, device=device)
    env = RslRlVecEnvWrapper(raw_env)
    try:
        if task_name == "motion_tracking":
            command = raw_env.command_manager.get_term("motion")
            library = command.library
            available = [clip.name for clip in library.clips]
            wanted = selected_names if selected_names is not None else available
            missing = sorted(set(wanted) - set(available))
            if missing:
                raise ValueError(f"requested Motion clips absent from checkpoint library: {missing}")
            if not wanted:
                raise ValueError("the checkpoint Motion Library has no clips")
            clip_index = {name: i for i, name in enumerate(available)}
            chosen = torch.tensor(
                [clip_index[name] for name in wanted], device=raw_env.device, dtype=torch.long
            )
            # Fixed forced_frame is consulted by every manager reset, so clip end
            # and early falls restart this same reference and reset all histories.
            env_ids = torch.arange(num_envs, device=raw_env.device)
            command.forced_frame.copy_(library.clip_start[chosen[env_ids % chosen.numel()]])
            print(f"[INFO] motion clips per env: {[wanted[i % len(wanted)] for i in range(num_envs)]}")
        obs, _ = env.reset()
        if task_name == "velocity_command" and fixed_command is not None:
            # Reset hooks sample the configured fixed command; assert the policy,
            # reward manager, and command visualizer all read the same tensor.
            expected = raw_env.command_manager.get_term("velocity").command.new_tensor(fixed_command)
            got = raw_env.command_manager.get_term("velocity").command
            if not torch.allclose(got, expected.expand_as(got)):
                raise RuntimeError(f"fixed velocity command was not installed across reset: {got}")
        if task_name == "world_point_reach":
            command = raw_env.command_manager.get_term("world_point")
            print(f"[INFO] M1 fixed env-local world target[0]={command.target_position_world[0].tolist()} (target marker and TCP error enabled)")

        if policy.cfg.num_joints != get_robot_spec(robot_name).num_joints:
            raise ValueError(
                f"checkpoint Actor has {policy.cfg.num_joints} actions but {robot_name} expects "
                f"{get_robot_spec(robot_name).num_joints}"
            )
        goal_dim = int(policy.cfg.goal_dim)
        print(
            f"[INFO] checkpoint={ckpt} robot={robot_name} task={task_name} "
            f"actions={policy.cfg.num_joints} obs={history_length * (3 * policy.cfg.num_joints + 6)} "
            f"goal={goal_dim} envs={num_envs} viewer={viewer}"
        )
        print(f"[INFO] model asset={_asset_path(robot_name)}")
        if task_name == "velocity_command":
            command = raw_env.command_manager.get_term("velocity")
            print(f"[INFO] velocity command={command.command[0].tolist()} (body frame), loop=episode reset")

        pol = _DictPolicy(policy)
        if viewer == "auto":
            viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
        print(f"[INFO] viewer={viewer}; action/observation/reset noise, DR and replay recording are disabled")
        if viewer == "native":
            play_viewer = NativeMujocoViewer(env, pol)
            play_viewer._show_all_envs = num_envs > 1
            play_viewer.run()
        else:
            _AllEnvViserViewer(env, pol).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
