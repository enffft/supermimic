"""Validate the material package without importing a simulator or starting a viewer.

Uses saved evidence, streaming SHA256, Pillow and shell command printing only.
Run with the existing Conda supermimic interpreter.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

from PIL import Image

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
DATA = OUT / "data"
checks = []


def check(name, condition, detail=""):
    checks.append({"name": name, "passed": bool(condition), "detail": detail})


def reject_constant(value):
    raise ValueError(f"Non-finite JSON literal: {value}")


def load(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, timeout=15).strip()


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True


manifest = load(DATA / "experiment_manifest.json")
check("Git branch preserved", run("git", "branch", "--show-current") == manifest["branch"])
check("Git HEAD preserved", run("git", "rev-parse", "HEAD") == manifest["head"])
check("Original worktree status preserved", run("git", "status", "--short") == manifest["status_at_extraction"])

json_count = 0
for path in DATA.rglob("*.json"):
    if path.name == "deliverable_validation.json":
        continue
    try:
        value = load(path)
        check(f"Finite JSON: {path.relative_to(OUT)}", finite_tree(value))
    except (ValueError, OSError) as exc:
        check(f"JSON: {path.relative_to(OUT)}", False, str(exc))
    json_count += 1

for item in manifest["sources"]:
    original = ROOT / item["path"]
    check(f"Original SHA: {item['path']}", original.is_file() and sha(original) == item["sha256"])
    if item.get("snapshot"):
        snapshot = OUT / item["snapshot"]
        check(f"Snapshot SHA: {item['snapshot']}", snapshot.is_file() and sha(snapshot) == item["sha256"])

for name, item in manifest["checkpoints"].items():
    path = ROOT / item["path"]
    check(f"Checkpoint unchanged: {name}", path.is_file() and sha(path) == item["sha256"])

registry = load(DATA / "figure_registry.json")
check("17 figure entries", len(registry) == 17)
for name, item in registry.items():
    for source in item["inputs"] + [item["script"]]:
        check(f"Figure dependency: {name}/{source}", (OUT / source).is_file())
    png, svg = [OUT / "figures" / (name + suffix) for suffix in [".png", ".svg"]]
    with Image.open(png) as picture:
        check(f"PNG dimensions: {name}", picture.size == (3840, 2160))
        picture.verify()
    tree = ET.parse(svg)
    tags = [element.tag.rsplit("}", 1)[-1] for element in tree.iter()]
    check(f"SVG path font: {name}", "path" in tags and "text" not in tags)
    check(f"Nonempty figure: {name}", png.stat().st_size > 20_000 and svg.stat().st_size > 20_000)

link_count = 0
for path in OUT.glob("*.md"):
    text = path.read_text()
    check(f"No template markers: {path.name}", "<!--" not in text and "{{" not in text)
    check(f"No malformed math delimiters: {path.name}", r"\∣" not in text)
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        target = match.group(1).strip().strip("<>")
        if urlparse(target).scheme or target.startswith("#"):
            continue
        local = unquote(target.split("#", 1)[0])
        check(f"Markdown link: {path.name}/{target}", (path.parent / local).exists())
        link_count += 1

cache = {}
numeric_count = 0
with (DATA / "numeric_evidence.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        source = row["source_snapshot"]
        if source not in cache:
            cache[source] = load(OUT / source)
        value = cache[source]
        for key in row["json_pointer"].split("/")[1:]:
            key = key.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        expected = json.loads(row["value"])
        if value != expected:
            check(f"Numeric pointer {source}:{row['json_pointer']}", False)
        numeric_count += 1
check("Numeric evidence pointers verified", numeric_count > 60_000, str(numeric_count))

for robot, envs in [("go2", 3072), ("m1", 2048)]:
    ppo = load(DATA / f"{robot}_ppo_training_progress.json")
    check(f"{robot} PPO transition count", ppo["real_transitions"] == ppo["real_env_steps_per_env"] * envs)
    if robot == "go2":
        sm = load(DATA / "go2_supermimic_training_progress.json")
        steps = sm["completed_env_steps_per_env"]
    else:
        sm = load(DATA / "m1_supermimic_completion.json")
        steps = sm["iterations"]
    check(f"{robot} SuperMimic transition count", sm["real_transitions"] == steps * envs)
    cfg = load(DATA / f"{robot}_supermimic_config.json")
    policy = cfg["policy"]
    expected_input = 213 if robot == "go2" else 728
    check(f"{robot} Actor input", policy["history"] * (3 * policy["num_joints"] + 6) + policy["goal_dim"] == expected_input)
    check(f"{robot} Direct goal path", policy["goal_encoding"] == "direct")
    check(f"{robot} H32", cfg["policy_trainer"]["horizon"] == 32)

for algorithm in ["ppo", "supermimic"]:
    m = load(DATA / f"m1_{algorithm}_manipulation.json")
    metrics, goals = m["metrics"], m["goals"]
    completed = [goal for goal in goals if not goal.get("censored", False)]
    successful = [goal for goal in completed if goal["success"]]
    check(f"M1 {algorithm} ended goal denominator", len(completed) == metrics["M_completed_goals"])
    check(f"M1 {algorithm} goal success numerator", len(successful) == metrics["M_successful_goals"])
    check(f"M1 {algorithm} success rate", math.isclose(len(successful) / len(completed), metrics["M_goal_success_rate"]))
    check(f"M1 {algorithm} fall denominator", len(m["episodes"]) == metrics["M_completed_episodes"])
    check(f"M1 {algorithm} fall rate", math.isclose(sum(ep["fall"] for ep in m["episodes"]) / len(m["episodes"]), metrics["M_fall_rate"]))
    check(f"M1 {algorithm} v2 target and seed", m["task_contract"]["target"]["radius_m"] == 3 and m["seed"] == 123 and m["duration_s"] == 60)
go2 = load(DATA / "go2_results.json")
check("Go2 nine paired commands", len(go2["go2_ppo"]) == len(go2["go2_supermimic"]) == 9 and [r["command_body_xy_yaw"] for r in go2["go2_ppo"]] == [r["command_body_xy_yaw"] for r in go2["go2_supermimic"]])
check("Go2 all 64 ended episodes", all(row["episodes"] == 64 for rows in go2.values() for row in rows))

verification = load(DATA / "evidence_verification.json")
check("Export CPU equality evidence", all(v["policy_tensors_exact_equal"] and v["random_input_max_action_error"] == 0 for v in verification["exports"].values()))
check("Historical ten source hashes", len(verification["historical_source_hash_matches"]) == 10 and all(v["matches"] for v in verification["historical_source_hash_matches"].values()))

wrapper = OUT / "scripts/play_demo.sh"
subprocess.run(["bash", "-n", str(wrapper)], check=True)
cases = [f"go2_{a}_{c}" for a in ["ppo", "sm"] for c in ["forward", "yaw", "side"]]
cases += ["go2_old_trot"]
cases += [f"m1_{a}_{c}" for a in ["ppo", "sm"] for c in ["L", "M", "far", "low", "random"]]
help_text = (DATA / "play_help.txt").read_text()
for case in cases:
    printed = run("bash", str(wrapper), case, "--print")
    tokens = shlex.split(printed)
    flags = [token for token in tokens if token.startswith("--")]
    check(f"Play interface: {case}", all(flag in help_text for flag in flags))
    check(f"Play Conda interpreter: {case}", "/home/yifeiy/miniconda3/envs/supermimic/bin/python" in tokens)
    checkpoint = tokens[tokens.index("--ckpt") + 1]
    check(f"Play checkpoint exists: {case}", (ROOT / checkpoint).is_file())

for path in (OUT / "scripts").glob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
check("Analysis script syntax", True, "ast.parse only; no imported copied sources")

external = load(OUT / "external_sources/index.json")
check("Six public repositories", len(external) == 6)
for repo in external:
    directory = OUT / "external_sources" / repo["repo"].replace("/", "__")
    for item in repo["files"]:
        check(f"Public source SHA: {repo['repo']}/{item['path']}", sha(directory / item["path"]) == item["sha256"])

summary = {
    "passed": all(item["passed"] for item in checks),
    "check_count": len(checks), "json_count": json_count,
    "figure_pairs": len(registry), "numeric_pointers": numeric_count,
    "markdown_local_links": link_count, "play_cases": len(cases),
    "method": "CPU file checks, saved evidence and command printing; no new simulator/viewer/training",
    "visual_review": "17-figure contact sheet and G02/M04/M05/A01 individual review; CJK visible, no obvious clipping",
    "limits": "Checks consistency with saved evidence, not historical source completeness, causal mechanisms or new runtime behavior.",
    "checks": checks,
}
(DATA / "deliverable_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, ensure_ascii=False, indent=2))
for item in checks:
    if not item["passed"]:
        print("FAIL:", item["name"], item["detail"])
raise SystemExit(0 if summary["passed"] else 1)
