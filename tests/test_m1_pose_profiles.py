from __future__ import annotations

import pytest


def test_prior_joint_reset_matches_v33_high_l_and_preserves_m_carry_profile():
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.robots.spec import (
        M1_CARRY_DEFAULT, M1_JOINTS, M1_V33_L_HIGH_CARRY_DEFAULT, get_robot_spec,
    )

    for task_type in ("L", "M"):
        cfg = make_env_cfg(
            SuperMimicEnvCfgArgs(
                npz_files=[],
                robot_name="m1_piper",
                task_name="m1_prior_joint",
                num_envs=1,
                obs_noise=False,
                reset_noise=False,
                action_noise_std=0.0,
                m1_joint_task_type=task_type,
            )
        )
        entity = cfg.scene.entities["robot"]
        profile = "dataset_v3_3_l_high_carry" if task_type == "L" else "dataset_carry"
        expected = M1_V33_L_HIGH_CARRY_DEFAULT if task_type == "L" else M1_CARRY_DEFAULT
        assert entity.init_state.pos[2] == pytest.approx(
            get_robot_spec("m1_piper", m1_pose_profile=profile, m1_action_contract="v3").default_base_height
        )
        assert tuple(entity.init_state.joint_pos[n] for n in M1_JOINTS) == pytest.approx(
            expected
        )
        assert entity.init_state.joint_vel == {".*": 0.0}
        assert cfg.actions["mixed"].position_target_bounds


def test_v33_l_profile_uses_source_stance_and_default_is_inside_every_position_bound():
    from supermimic.robots.spec import M1_JOINTS, get_robot_spec

    spec = get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    )
    assert spec.default_base_height == pytest.approx(0.47)
    assert spec.default_pose_profile == "m1_pose_dataset_v3_3_l_high_carry_v1"
    assert spec.default_joint_pos[M1_JOINTS.index("FAR_HIP_JOINT")] == pytest.approx(0.8812830158963126)
    assert spec.default_joint_pos[M1_JOINTS.index("FAR_KNEE_JOINT")] == pytest.approx(-1.6184887638693073)
    for name, default, (lower, upper) in zip(
        spec.joint_names, spec.default_joint_pos, spec.position_target_bounds,
    ):
        if name.endswith("FOOT_JOINT"):
            continue
        assert lower <= default <= upper, name


def test_legacy_manipulation_profile_remains_explicit_and_v2_compatible():
    from supermimic.robots.spec import M1_CARRY_DEFAULT, M1_DEFAULT, get_robot_spec

    legacy = get_robot_spec("m1_piper")
    carry = get_robot_spec("m1_piper", m1_pose_profile="dataset_carry", m1_action_contract="v3")
    assert legacy.default_joint_pos == M1_DEFAULT
    assert legacy.default_pose_profile == "m1_pose_legacy_manipulation_v1"
    assert legacy.action_contract_version == "m1_mixed_action_v2"
    assert carry.default_joint_pos == M1_CARRY_DEFAULT
    assert carry.default_pose_profile == "m1_pose_dataset_carry_v1"
    assert carry.action_contract_version == "m1_mixed_action_v3"
    assert legacy.default_joint_pos[:16] == carry.default_joint_pos[:16]
    assert legacy.default_base_height == carry.default_base_height
    assert all(
        legacy.default_joint_pos[i] == carry.default_joint_pos[i] == 0.0 for i in (3, 7, 11, 15)
    )


def test_m1_prior_joint_rejects_legacy_pose_under_v3_bounds():
    from supermimic.robots.spec import get_robot_spec

    with pytest.raises(ValueError, match="outside v3 target bounds"):
        get_robot_spec("m1_piper", m1_pose_profile="legacy_manipulation", m1_action_contract="v3")


def test_g1_go2_action_contracts_are_unchanged():
    from supermimic.robots.spec import get_robot_spec

    g1 = get_robot_spec("g1")
    go2 = get_robot_spec("go2")
    assert g1.action_scales == (1.0,) * 29
    assert go2.action_scales == (0.25,) * 12
    assert not g1.position_target_bounds and not go2.position_target_bounds
