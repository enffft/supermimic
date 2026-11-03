#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRIOR=/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt
PRIOR_SHA256=05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c
LOG_ROOT=logs/m1_prior_actor
BASE_RUN_NAME=m1_action22_v1_LM50_2048_1h_20261103
PREFLIGHT="${M1_PREFLIGHT:-0}"
if [[ "$PREFLIGHT" == "1" ]]; then
  BASE_RUN_NAME="${BASE_RUN_NAME}_preflight"
  WARMUP_ITERATIONS=12
  WM_ONLY_ITERATIONS=1
  MAX_HOURS=0.1
  CHECKPOINT_MINUTES=0
  STOP_AT_ITERATION=50
else
  WARMUP_ITERATIONS=40
  WM_ONLY_ITERATIONS=200
  MAX_HOURS=1.0
  CHECKPOINT_MINUTES=15
  STOP_AT_ITERATION=0
fi
RUN_NAME="$BASE_RUN_NAME"
suffix=1
while [[ -e "$LOG_ROOT/$RUN_NAME" ]]; do
  RUN_NAME="${BASE_RUN_NAME}_$(printf '%02d' "$suffix")"
  suffix=$((suffix + 1))
done

if [[ ! -f "$PRIOR" ]]; then
  echo "V3.3 Motion Prior checkpoint not found: $PRIOR" >&2
  exit 2
fi
actual_sha="$(sha256sum "$PRIOR" | awk '{print $1}')"
if [[ "$actual_sha" != "$PRIOR_SHA256" ]]; then
  echo "V3.3 Motion Prior SHA256 mismatch: expected=$PRIOR_SHA256 actual=$actual_sha" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "run_name=$RUN_NAME"
echo "log_dir=$LOG_ROOT/$RUN_NAME"
echo "prior=$PRIOR"
echo "prior_sha256=$actual_sha"
echo "preflight=$PREFLIGHT"
mkdir -p "$LOG_ROOT/$RUN_NAME"

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 5 > "$LOG_ROOT/$RUN_NAME/gpu_monitor.csv" &
GPU_MONITOR_PID=$!
(
  while true; do
    date --iso-8601=seconds
    free -b
    sleep 30
  done
) > "$LOG_ROOT/$RUN_NAME/host_memory_monitor.log" &
MEMORY_MONITOR_PID=$!
cleanup_monitors() {
  kill "$GPU_MONITOR_PID" "$MEMORY_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" "$MEMORY_MONITOR_PID" 2>/dev/null || true
}
trap cleanup_monitors EXIT

STOP_ARGS=()
if [[ "$STOP_AT_ITERATION" != "0" ]]; then
  STOP_ARGS=(--schedule.stop-at-iteration "$STOP_AT_ITERATION")
fi

/home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name "$RUN_NAME" \
  --log-root "$LOG_ROOT" \
  --seed 20261103 \
  --device cuda:0 \
  --env.robot-name m1_piper \
  --env.task-name m1_prior_joint \
  --env.num-envs 2048 \
  --env.history-length 5 \
  --env.episode-length-s 10 \
  --env.action-noise-std 0 \
  --env.no-obs-noise \
  --env.no-reset-noise \
  --env.no-domain-randomization \
  --env.replay-capacity 32 \
  --env.m1-pose-profile dataset_v3_3_l_high_carry \
  --env.m1-action-contract v3 \
  --env.m1-joint-task-type balanced \
  --env.m1-joint-l-task-fraction 0.5 \
  --env.m1-joint-balanced-fixed-quota \
  --env.m1-joint-vx-range -0.09 0.145 \
  --env.m1-joint-vx-stop-fraction 0.15 \
  --env.m1-joint-vx-reverse-fraction 0.35 \
  --env.m1-joint-vy-range -0.05 0.055 \
  --env.m1-joint-wz-range -0.17 0.18 \
  --env.m1-joint-target-x-range 0.12 0.36 \
  --env.m1-joint-target-y-range -0.20 0.20 \
  --env.m1-joint-target-z-range -0.15 0.15 \
  --m1-joint-training.enabled \
  --m1-joint-training.task-type balanced \
  --m1-joint-training.l-task-fraction 0.5 \
  --m1-joint-training.training-mode staged_action22_v1 \
  --m1-joint-training.prior-checkpoint "$PRIOR" \
  --m1-joint-training.expected-prior-sha256 "$PRIOR_SHA256" \
  --m1-joint-training.prior-mode prior_mean \
  --m1-joint-training.prior-input-compatibility legacy_v1 \
  --m1-joint-training.freeze-prior \
  --m1-joint-training.prior-updates-per-event 0 \
  --m1-joint-training.actor-updates-per-event 1 \
  --m1-joint-training.actor-pretrain-updates 384 \
  --m1-joint-training.balanced-replay-sampling \
  --m1-joint-training.batch-size 32 \
  --m1-joint-training.micro-batch-size 8 \
  --m1-joint-training.exploration-enabled \
  --m1-joint-training.exploration-env-fraction 0.25 \
  --m1-joint-training.exploration-leg-std 0.02 \
  --m1-joint-training.exploration-arm-std 0.015 \
  --m1-joint-training.exploration-wheel-std 0.05 \
  --m1-joint-training.loss.termination-failure-weight 1.0 \
  --schedule.warmup-iterations "$WARMUP_ITERATIONS" \
  --schedule.m1-wm-only-iterations "$WM_ONLY_ITERATIONS" \
  --schedule.max-hours "$MAX_HOURS" \
  --schedule.log-interval 50 \
  --schedule.checkpoint-interval 10000 \
  --schedule.checkpoint-interval-minutes "$CHECKPOINT_MINUTES" \
  --wm-trainer.batches-per-update 1 \
  --wm-trainer.batch-size 32 \
  --wm-trainer.micro-batch-size 8 \
  "${STOP_ARGS[@]}" 2>&1 | tee "$LOG_ROOT/$RUN_NAME/console.log"

if [[ "$PREFLIGHT" == "1" ]]; then
  echo "Preflight completed; full evaluation is intentionally not run."
  exit 0
fi

CHECKPOINT="$LOG_ROOT/$RUN_NAME/ckpt_latest.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training exited without a final checkpoint: $CHECKPOINT" >&2
  exit 3
fi

EVAL_ROOT="$LOG_ROOT/$RUN_NAME/evaluation"
mkdir -p "$EVAL_ROOT"
for task in L M; do
  for controller in actor prior_direct; do
    output="$EVAL_ROOT/${task}_${controller}.json"
    echo "Evaluating task=$task controller=$controller checkpoint=$CHECKPOINT"
    /home/yifeiy/miniconda3/envs/supermimic/bin/python \
      scripts/evaluate_m1_prior_joint.py \
      --checkpoint "$CHECKPOINT" \
      --task-type "$task" \
      --controller "$controller" \
      --num-envs 64 \
      --steps 500 \
      --seed 20261103 \
      --device cuda:0 \
      --out "$output" 2>&1 | tee "$EVAL_ROOT/${task}_${controller}.log"
  done
done
