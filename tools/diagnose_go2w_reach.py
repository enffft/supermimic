"""Fixed-base kinematic workspace under the actual target clipping law.

This is a joint-target envelope, not proof of dynamic stair feasibility.
"""

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from unisupermimic.tasks.velocity.go2w.contract import (
    DEFAULT_Q,
    LEG_JOINTS,
    LEGS,
    MODEL,
    soft_limits,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    q_indices = np.array([model.joint(name).qposadr[0] for name in LEG_JOINTS])
    bodies = [model.body(f"robot/{leg}_wheel_link").id for leg in LEGS]
    mujoco.mj_kinematics(model, data)
    nominal = data.xpos[bodies].copy()
    limits = soft_limits().numpy()
    result = []
    for scale in (.25, .5, .75):
        for leg, name in enumerate(LEGS):
            part = slice(leg * 3, leg * 3 + 3)
            samples = []
            for action in itertools.product(np.linspace(-1, 1, 13), repeat=3):
                data.qpos[q_indices] = DEFAULT_Q
                data.qpos[q_indices[part]] = np.clip(np.array(DEFAULT_Q)[part] + scale * np.array(action),
                                                    limits[part, 0], limits[part, 1])
                mujoco.mj_kinematics(model, data)
                samples.append(data.xpos[bodies[leg]].copy())
            points = np.array(samples)
            delta = points - nominal[leg]
            best = int(np.argmax(delta[:, 2]))
            result.append(dict(leg=name, target_scale_rad=scale,
                               max_center_lift_m=float(delta[:, 2].max()),
                               xyz_displacement_at_max_lift_m=delta[best].tolist(),
                               center_y_span_m=float(np.ptp(points[:, 1])),
                               center_x_span_m=float(np.ptp(points[:, 0]))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": str(MODEL), "grid_per_joint": 13, "rows": result,
        "limits": "Actual soft joint limits, action [-1,1]; fixed neutral base and other legs; FK only",
        "interpretation": "Lift relative to nominal wheel center, not obstacle clearance or a learned/dynamically feasible gait. Only .25 is the current controller; other scales are offline candidates.",
    }, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
