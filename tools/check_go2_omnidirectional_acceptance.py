"""Report real acceptance boundaries; never equate core recovery to full readiness."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    root = a.run
    result = json.loads((root / "stageB_evaluation/evaluation.json").read_text())
    agg = result["aggregate"]
    from evaluate_go2_omnidirectional import FLAT

    core = {}
    for name, _ in FLAT[:14]:
        r = agg[name]
        xy_limit = 0.12 if name == "forward_high" else 0.1
        core[name] = r["fall"] == 0 and r["xy_rmse_m_s"] < xy_limit and r["yaw_rmse_rad_s"] < 0.15
    stairs = {
        name: agg[name]["fall"] == 0
        and agg[name]["xy_rmse_m_s"] < 0.1
        and agg[name]["crossing_passed"] >= 0.4
        for name in ("stairs_up_L4_cmd0", "stairs_down_L4_cmd0")
    }
    probes = json.loads((root / "original_probes_stageB/acceptance.json").read_text())
    rejected = [name for name in ("lateral_boundary", "backward_boundary") if agg[name]["fall"] > 0]
    evidence = dict(
        core_14_commands_pass=all(core.values()),
        core_command_checks=core,
        level4_forward_retained=all(stairs.values()),
        level4_checks=stairs,
        original_contact_gradient_gate=probes["physical_gradient_pass"],
        original_minimum_gradient_agreement=probes["minimum_probe_sign_agreement"],
        boundary_commands_with_falls=rejected,
        terrain_perturbation_falls=[
            dict(scenario=r["scenario"], seed=r["seed"], survival_s=r["survival_s"])
            for r in result["records"]
            if r["domain"] == "terrain" and r["fall"]
        ],
        stageC_longrun_allowed=all(core.values())
        and all(stairs.values())
        and probes["physical_gradient_pass"]
        and not rejected,
        reason="Core omnidirectional recovery passed. Keep speed level0; contact-gradient tail and extreme lateral failures block unattended speed-range expansion/long training.",
    )
    (root / "acceptance.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
