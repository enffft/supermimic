#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$repo_root/logs/m1_prior_actor/m1_v33_L_2048_2h_actor_only_resume72m"
if [[ -e "$run_dir" ]]; then
  echo "Run directory already exists; refusing to overwrite: $run_dir" >&2
  exit 1
fi
prior_path="/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt"
prior_sha256="05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c"
resume_path="$repo_root/logs/m1_prior_actor/m1_v33_L_2048_2h_resume90m_rngfix/ckpt_0003000.pt"
resume_sha256="968feac5ccd01681a8c7133daa9b6095d3b934da4dcd7ade7810a696a90aae71"

actual_prior_sha256="$(/usr/bin/sha256sum "$prior_path" | /usr/bin/awk '{print $1}')"
if [[ "$actual_prior_sha256" != "$prior_sha256" ]]; then
  echo "V3.3 prior SHA-256 mismatch: expected $prior_sha256, got $actual_prior_sha256" >&2
  exit 1
fi
actual_resume_sha256="$(/usr/bin/sha256sum "$resume_path" | /usr/bin/awk '{print $1}')"
if [[ "$actual_resume_sha256" != "$resume_sha256" ]]; then
  echo "it3000 Checkpoint SHA-256 mismatch: expected $resume_sha256, got $actual_resume_sha256" >&2
  exit 1
fi

cd "$repo_root"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec env CUDA_VISIBLE_DEVICES=0 \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name m1_v33_L_2048_2h_actor_only_resume72m \
  --log-root logs/m1_prior_actor \
  --seed 20261103 \
  --device cuda:0 \
  --env.robot-name m1_piper \
  --env.task-name m1_prior_joint \
  --env.num-envs 2048 \
  --env.history-length 5 \
  --env.m1-pose-profile dataset_v3_3_l_high_carry \
  --env.m1-action-contract v3 \
  --env.m1-joint-task-type L \
  --env.m1-joint-vx-range -0.09 0.145 \
  --env.m1-joint-vy-range 0 0 \
  --env.m1-joint-wz-range 0 0 \
  --env.no-obs-noise \
  --env.no-reset-noise \
  --env.no-domain-randomization \
  --env.action-noise-std 0 \
  --m1-joint-training.enabled \
  --m1-joint-training.resume "$resume_path" \
  --m1-joint-training.prior-checkpoint "$prior_path" \
  --m1-joint-training.expected-prior-sha256 "$prior_sha256" \
  --m1-joint-training.prior-input-compatibility legacy_v1 \
  --m1-joint-training.task-type L \
  --m1-joint-training.state-source privileged \
  --m1-joint-training.prior-mode prior_mean \
  --m1-joint-training.training-mode controlled_joint_experiment \
  --m1-joint-training.no-freeze-prior \
  --m1-joint-training.batch-size 32 \
  --m1-joint-training.micro-batch-size 8 \
  --m1-joint-training.prior-updates-per-event 0 \
  --m1-joint-training.actor-updates-per-event 1 \
  --m1-joint-training.controlled-joint-min-wm-events 200 \
  --m1-joint-training.controlled-joint-actor-pretrain-updates 384 \
  --m1-joint-training.controlled-joint-min-real-samples 100 \
  --schedule.total-iterations 320000 \
  --schedule.m1-wm-only-iterations 200 \
  --schedule.log-interval 25 \
  --schedule.checkpoint-interval 500 \
  --schedule.max-hours 1.2
