"""Explicit physical-evaluation admission; no optimizer/time-based promotion.

S levels are indexed by stairs direction x terrain level. Training terrain
mastery and flat speed stage are independent. Evidence is stored in checkpoints.
"""


def stairs_speed_admission(evaluation, diagnostics, direction, terrain_level, speed_level):
    prefix = f"stairs_{'up' if direction == 0 else 'down'}_L{terrain_level}_S{speed_level}_"
    physical = {k: v for k, v in evaluation.items() if k.startswith(prefix)}
    physics_ok = len(physical) == 7 and all(
        v["fall"] == 0
        and v["survival_s"] >= 3.98
        and v["xy_rmse_m_s"] < 0.12
        and v["yaw_rmse_rad_s"] < 0.18
        and v.get("outside_tile_fraction", 0) <= 0.01
        for v in physical.values()
    )
    gradients = [
        g
        for g in diagnostics["gradients"]
        if g["scenario"].startswith(prefix)
        and g["quantity"] == "joint_velocity"
        and g["epsilon"] == 0.03
        and g["sign_agreement"] is not None
    ]
    prediction = [
        p
        for p in diagnostics["prediction"]
        if p["scenario"].startswith(prefix) and p["horizon"] == 32
    ]
    # Preserve the previous >=80% per-probe qd direction rule. No global legacy
    # acceptance flag is changed. A failed pair remains at its admitted range.
    gradient_ok = len(gradients) == 7 and all(
        g["sign_agreement"] >= 0.80 and g["normalized_error"] < 1.0 for g in gradients
    )
    prediction_ok = len(prediction) >= 5 and all(
        p["position_m"] < 0.08
        and p["velocity_m_s"] < 0.15
        and p["joint_position_rad"] < 0.12
        and p["attitude_rad"] < 0.15
        for p in prediction
    )
    return dict(
        admitted=physics_ok and gradient_ok and prediction_ok,
        physics_ok=physics_ok,
        gradient_ok=gradient_ok,
        prediction_ok=prediction_ok,
        minimum_qd_sign=min((g["sign_agreement"] for g in gradients), default=None),
        physical_scenarios=len(physical),
        gradient_probes=len(gradients),
    )


def imagination_scope_admission(diagnostics_by_speed, stairs_levels):
    """Keep all real WM windows; restrict only imagined optimization starts.

    Unchanged flat/rough/slope domains retain their prior admission. Every
    nonflat stair pair needs >=80% qd direction agreement on all seven command
    probes and bounded heldout H32 error. Physical task proficiency controls
    subsequent speed promotion, not the ability to learn a not-yet-mastered task.
    """
    import torch

    nlevels = len(stairs_levels[0])
    mask = torch.ones(6, nlevels, dtype=torch.bool)
    details = {}
    for direction in range(2):
        for level in range(1, nlevels):
            speed = int(stairs_levels[direction][level])
            diag = diagnostics_by_speed[speed]
            prefix = f"stairs_{'up' if direction == 0 else 'down'}_L{level}_S{speed}_"
            gradients = [
                g
                for g in diag["gradients"]
                if g["scenario"].startswith(prefix)
                and g["quantity"] == "joint_velocity"
                and g["epsilon"] == 0.03
                and g["sign_agreement"] is not None
            ]
            prediction = [
                p
                for p in diag["prediction"]
                if p["scenario"].startswith(prefix) and p["horizon"] == 32
            ]
            reliable = len(gradients) == 7 and all(
                g["sign_agreement"] >= 0.80 and g["normalized_error"] < 1 for g in gradients
            )
            accurate = len(prediction) >= 5 and all(
                p["position_m"] < 0.08
                and p["velocity_m_s"] < 0.15
                and p["joint_position_rad"] < 0.12
                and p["attitude_rad"] < 0.15
                for p in prediction
            )
            mask[direction + 4, level] = reliable and accurate
            details[prefix] = dict(
                admitted=bool(reliable and accurate),
                gradient_ok=reliable,
                prediction_ok=accurate,
                minimum_qd_sign=min((g["sign_agreement"] for g in gradients), default=None),
            )
    return mask, details
