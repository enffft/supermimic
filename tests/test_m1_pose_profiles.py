from __future__ import annotations

import pytest


def test_prior_joint_reset_uses_dataset_carry_for_l_and_m():
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.robots.spec import M1_CARRY_DEFAULT, M1_JOINTS, get_robot_spec

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
        assert entity.init_state.pos[2] == pytest.approx(
            get_robot_spec("m1_piper").default_base_height
        )
        assert tuple(entity.init_state.joint_pos[n] for n in M1_JOINTS) == pytest.approx(
            M1_CARRY_DEFAULT
        )
        assert entity.init_state.joint_vel == {".*": 0.0}
        assert cfg.actions["mixed"].position_target_bounds


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
