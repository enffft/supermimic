#!/usr/bin/env bash
set -euo pipefail

# Gated 10-minute L-only follow-up. This intentionally refuses to start unless
# the supplied finite-difference audit is complete, positive, and tied to a V5
# bounded-residual checkpoint. Prior remains frozen for the whole run.
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ACTOR_WM_GATE_EVIDENCE.json [tracking|task]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gate_evidence="$(realpath "$1")"
loss_variant="${2:-task}"
if [[ "$loss_variant" != "tracking" && "$loss_variant" != "task" ]]; then
  echo "loss variant must be tracking or task" >&2
  exit 2
fi

cd "$repo_root"
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python - "$gate_evidence" <<'PY'
import json, pathlib, sys, torch
p = pathlib.Path(sys.argv[1])
evidence = json.loads(p.read_text())
if evidence.get("audit_type") != "m1_wm_mujoco_finite_difference_v1":
    raise SystemExit("refusing Actor BPTT: unsupported WM/MuJoCo audit type")
if evidence.get("action_response_pass") is not True or evidence.get("task_gradient_direction_pass") is not True:
    raise SystemExit("refusing Actor BPTT: real action response/task-gradient gate did not pass")
if int(evidence.get("heldout_state_count", 0)) < 128:
    raise SystemExit("refusing Actor BPTT: fewer than 128 held-out physical states")
checkpoint_path = pathlib.Path(evidence.get("checkpoint", ""))
if not checkpoint_path.is_file():
    raise SystemExit("refusing Actor BPTT: audited V5 checkpoint is unavailable")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint.get("format") != "supermimic_m1_action22_reference_v2" or checkpoint.get("format_version") != 5:
    raise SystemExit("refusing Actor BPTT: audited checkpoint is not bounded-residual V5")
if checkpoint.get("world_model_sha256") != evidence.get("world_model_sha256"):
    raise SystemExit("refusing Actor BPTT: audited WM SHA256 does not match checkpoint")
print("WM/MuJoCo gate evidence is structurally eligible; the trainer will recheck all live contracts.")
PY

run_name="m1_action22_l_gated_1024_10m_${loss_variant}_20261104"
log_root="logs/m1_prior_actor"
if [[ -e "$log_root/$run_name" ]]; then
  run_name="${run_name}_$(date +%H%M%S)"
fi

loss_args=()
if [[ "$loss_variant" == "tracking" ]]; then
  loss_args+=(--m1-joint-training.loss.actor-task-weight 0.0)
fi

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
/home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name "$run_name" \
  --log-root "$log_root" \
  --seed 20261104 \
  --device cuda:0 \
  --env.robot-name m1_piper \
  --env.task-name m1_prior_joint \
  --env.num-envs 1024 \
  --env.history-length 5 \
  --env.m1-pose-profile dataset_v3_3_l_high_carry \
  --env.m1-action-contract v3 \
  --env.no-obs-noise \
  --env.no-reset-noise \
  --env.action-noise-std 0.0 \
  --env.no-domain-randomization \
  --m1-joint-training.enabled \
  --m1-joint-training.task-type L \
  --m1-joint-training.state-source privileged \
  --m1-joint-training.prior-checkpoint /home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt \
  --m1-joint-training.expected-prior-sha256 05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c \
  --m1-joint-training.prior-mode prior_mean \
  --m1-joint-training.prior-input-compatibility legacy_v1 \
  --m1-joint-training.prior-updates-per-event 0 \
  --m1-joint-training.actor-updates-per-event 1 \
  --m1-joint-training.freeze-prior \
  --m1-joint-training.actor-update-mode wm_task_gated \
  --m1-joint-training.actor-wm-gate-evidence-path "$gate_evidence" \
  --m1-joint-training.actor-leg-delta-max-normalized 0.05 \
  --m1-joint-training.actor-arm-delta-max-normalized 0.05 \
  --m1-joint-training.actor-wheel-delta-max-normalized 0.10 \
  --m1-joint-training.exploration-enabled \
  --m1-joint-training.exploration-env-fraction 0.25 \
  --m1-joint-training.exploration-leg-std 0.02 \
  --m1-joint-training.exploration-arm-std 0.015 \
  --m1-joint-training.exploration-wheel-std 0.05 \
  --m1-joint-training.batch-size 32 \
  --m1-joint-training.micro-batch-size 8 \
  --schedule.m1-wm-only-iterations 200 \
  --schedule.max-hours 0.167 \
  --schedule.log-interval 25 \
  --schedule.checkpoint-interval 100 \
  --schedule.checkpoint-interval-minutes 2 \
  "${loss_args[@]}"
