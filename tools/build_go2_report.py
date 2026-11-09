"""Fill the Chinese technical report with measured validation artifacts."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/go2_unilab_migration_report.md"


def read(name, file):
    return json.loads((ROOT / "results" / name / file).read_text())


def table(headers, rows):
    def cell(value):
        return (
            ("是" if value else "否") if isinstance(value, bool) else str(value).replace("|", r"\|")
        )

    return "\n".join(
        [
            "| " + " | ".join(cell(value) for value in headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(cell(value) for value in row) + " |" for row in rows],
        ]
    )


def main():
    text = REPORT.read_text()
    blocks = {}
    rows = []
    for name in ["physics_final_mujoco", "physics_final_mjwarp"]:
        data = read(name, "physical.json")
        for c in data["cases"]:
            rows.append(
                [
                    data["backend"],
                    str(c["command"]),
                    c["envs"],
                    f"{c['mean_velocity_xy'][0]:.5f}",
                    f"{c['mean_velocity_xy'][1]:.5f}",
                    f"{c['mean_yaw_rate']:.5f}",
                    f"{c['velocity_rmse']:.5f}",
                    c["done_count"],
                    f"{c['saturation_fraction'] * 100:.2f}%",
                    f"{c['reward_proxy_max_abs_gap']:.3g}",
                ]
            )
    blocks["PHYSICS"] = table(
        [
            "后端",
            "指令vx/vy/wz",
            "环境",
            "实测vx",
            "实测vy",
            "实测wz",
            "xy RMSE",
            "done",
            "饱和",
            "Reward proxy最大差",
        ],
        rows,
    )
    runs = {name: read("profile_serial_" + name, "training.json") for name in ["mujoco", "mjwarp"]}
    rows = []
    for name, data in runs.items():
        s = data["summary"]
        rows.append(
            [
                name,
                s["completed"],
                f"{s['wm_loss']:.5f}",
                f"{s['pi_loss']:.5f}",
                f"{s['v_loss']:.5f}",
                f"{s['pi_gradnorm']:.5f}",
                f"{data['actor_output']['abs_max']:.5f}",
                data["actor_output"]["saturation_fraction"],
            ]
        )
    blocks["TRAINING"] = table(
        [
            "后端",
            "完成",
            "WM loss",
            "Actor loss",
            "Critic loss",
            "Actor grad norm",
            "Actor |a|max",
            "饱和比例",
        ],
        rows,
    )
    rows = []
    for name, data in runs.items():
        for item in data["wm_predictions"]:
            rows.append([name, item["horizon"], item["valid_samples"], f"{item['state_rmse']:.6f}"])
    blocks["TRAINING"] += "\n\n" + table(
        ["后端", "WM horizon", "有效样本", "mixed-state RMSE"], rows
    )
    rows = []
    for joint in range(12):
        rows.append(
            [
                joint,
                *[
                    f"{runs[name]['wm_action_perturbations'][joint]['positive_response_fraction'] * 100:.2f}%"
                    for name in runs
                ],
            ]
        )
    blocks["TRAINING"] += (
        "\n\nWM单关节±.1扰动，64个Replay起点，正向 q 响应占比（不足100%即仍有反向样本）：\n\n"
        + table(["关节index（上述命名顺序）", "MuJoCo数据拟合WM", "mjwarp数据拟合WM"], rows)
    )
    rows = []
    for section in ["simulation", "wm_fit", "actor_bptt", "critic_update"]:
        for name, data in runs.items():
            timing = data["synchronized_timings"][section]
            rows.append(
                [
                    section,
                    name,
                    timing["count"],
                    f"{timing['mean_ms']:.3f}",
                    f"{timing['median_ms']:.3f}",
                    f"{timing['min_ms']:.3f}",
                    f"{timing['max_ms']:.3f}",
                ]
            )
    blocks["TIMINGS"] = table(
        ["阶段", "后端", "样本数", "均值ms", "中位ms", "最小ms", "最大ms"], rows
    )
    rows = []
    for name, data in runs.items():
        resources = read("profile_serial_" + name, "resource_usage.json")
        rows.append(
            [
                name,
                f"{data['peak_torch_allocated_bytes'] / 2**30:.3f} GiB",
                f"{data['peak_torch_reserved_bytes'] / 2**30:.3f} GiB",
                f"{resources['peak_process_gpu_mib']:.0f} MiB",
                resources["samples"],
                f"{resources['peak_host_rss_kib'] / 2**20:.3f} GiB",
            ]
        )
    blocks["RESOURCES"] = table(
        [
            "后端",
            "Torch allocated峰",
            "Torch reserved峰",
            "整进程GPU采样峰",
            "采样数",
            "host RSS峰",
        ],
        rows,
    )
    rows = []
    for profile in ["rough", "slope", "stairs", "mixed"]:
        d = read("terrain_" + profile + "_force", "terrain.json")
        c = d["contacts"]
        rows.append(
            [
                profile,
                "16 / 100+50",
                c["terrain_contacts"],
                c["elevated_contacts"],
                f"{c['maximum_force_norm']:.3f}",
                f"{d['heading']} / {d['quarter_turns']}",
                d["training_performed"],
            ]
        )
    blocks["TERRAIN"] = table(
        [
            "profile",
            "环境/步数",
            "terrain接触累计",
            "z>.005接触累计",
            "最大力范数N",
            "heading/quarter_turns",
            "训练",
        ],
        rows,
    )
    for name, content in blocks.items():
        start = f"<!-- GENERATED_{name} -->"
        end = f"<!-- /GENERATED_{name} -->"
        value = start + "\n\n" + content + "\n\n" + end
        pattern = re.escape(start) + r"[\s\S]*?" + re.escape(end)
        text = re.sub(pattern, lambda _: value, text) if end in text else text.replace(start, value)
    REPORT.write_text(text)
    print(REPORT)


if __name__ == "__main__":
    main()
