"""Retained skills remain guarded; improvements and exploratory failures are distinct."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from train_go2_stairs_omni import continuation_regressions


def result(vx=0.4, fall=0.0):
    return dict(actual_vx_m_s=vx, actual_vy_m_s=0, actual_wz_rad_s=0, fall=fall, crossing_passed=0)


def test_mean_motion_improvement_does_not_trigger_bias_guard():
    assert (
        continuation_regressions(
            {"stairs_up_L7_cmd0": result(0.35)}, {"stairs_up_L7_cmd0": result(0.03)}
        )
        == []
    )


def test_retained_falls_stop_but_exploratory_falls_are_separate():
    baseline = {"stairs_down_L4_cmd0": result(), "stairs_down_L6_cmd0": result()}
    current = {k: result(fall=0.6) for k in baseline}
    all_regressions = continuation_regressions(current, baseline)
    retained = continuation_regressions(current, baseline, retained_max_level=4)
    assert len(all_regressions) == 2 and len(retained) == 1
    assert retained[0]["scenario"] == "stairs_down_L4_cmd0"
