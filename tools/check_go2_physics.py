"""Compare exported physical arrays with the source compiled MuJoCo 3.16 model."""

import json
from pathlib import Path

import mujoco
import numpy as np

from unisupermimic.tasks.velocity.go2.contract import DEFAULT_Q, JOINTS, MODEL

ROOT = Path(__file__).resolve().parents[1]


def main():
    reference = np.load(ROOT / "migration/source_reference/compiled_physics.npz")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    comparisons = []
    for key in reference.files:
        source = reference[key]
        target = np.asarray(getattr(model, key))
        assert source.shape == target.shape, (key, source.shape, target.shape)
        error = float(np.max(np.abs(source - target))) if source.size else 0.0
        if key.startswith("geom_"):
            physical = (reference["geom_contype"] != 0) | (reference["geom_conaffinity"] != 0)
            checked_source, checked_target = source[physical], target[physical]
        else:
            checked_source, checked_target = source, target
        physical_error = (
            float(np.max(np.abs(checked_source - checked_target))) if checked_source.size else 0.0
        )
        np.testing.assert_allclose(
            checked_source, checked_target, atol=1e-6, rtol=1e-5, err_msg=key
        )
        comparisons.append(
            {
                "field": key,
                "all_geoms_max_abs_error": error,
                "physical_max_abs_error": physical_error,
                "shape": list(source.shape),
            }
        )
    # Explicitly exercise every named actuator rather than trusting list order.
    mapping = []
    for name in JOINTS:
        j = model.joint(name)
        a = model.actuator(name)
        assert model.actuator_trnid[a.id, 0] == j.id
        assert model.actuator_gainprm[a.id, 0] == 40.0
        assert model.actuator_biasprm[a.id, 2] == -1.0
        mapping.append(
            {
                "joint": name,
                "qpos_index": int(j.qposadr[0]),
                "dof_index": int(j.dofadr[0]),
                "actuator_index": a.id,
            }
        )
    np.testing.assert_allclose(model.key_qpos[0, 7:], DEFAULT_Q, atol=1e-6)
    # Pure physics early response: +/- normalized action on each joint.
    perturbations = []
    for name in JOINTS:
        measured = []
        for sign in (-1, 1):
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            a = model.actuator(name).id
            q = model.joint(name).qposadr[0]
            data.ctrl[a] += sign * 0.25 * 0.1
            mujoco.mj_step(model, data, nstep=4)
            measured.append(float(data.qpos[q]))
        assert measured[1] > measured[0], (name, measured)
        perturbations.append(
            {"joint": name, "q_positive_minus_negative": measured[1] - measured[0]}
        )
    result = {
        "source_mujoco": "3.16.0",
        "target_mujoco": mujoco.__version__,
        "physical_arrays": comparisons,
        "mapping": mapping,
        "action_perturbations": perturbations,
        "control_dt": 0.02,
        "sim_dt": model.opt.timestep,
        "iterations": model.opt.iterations,
        "ls_iterations": model.opt.ls_iterations,
        "limitation": "Physical parameter comparison and local response only; engine version changes preclude a trajectory-equivalence claim.",
    }
    out = ROOT / "results/physics_equivalence.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "PASS",
        len(comparisons),
        "physical arrays; 12 named joint/actuator mappings; 12 signed perturbations",
    )


if __name__ == "__main__":
    main()
