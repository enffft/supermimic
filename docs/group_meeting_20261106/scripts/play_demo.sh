#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python_bin=/home/yifeiy/miniconda3/envs/supermimic/bin/python
case_name="${1:-help}"
print_only="${2:-}"
cmd=(env PYTHONPATH=src OMP_NUM_THREADS=1 "$python_bin" scripts/play.py)
go2_common=(--num-envs 1 --viewer native --velocity-arrow --actual-velocity-arrow)
m1_common=(--num-envs 1 --viewer native)
m1_tag=v2_lm50_2048_3p5h_seed2026
case "$case_name" in
  go2_ppo_forward|go2_sm_forward|go2_ppo_yaw|go2_sm_yaw|go2_ppo_side|go2_sm_side|go2_old_trot)
    if [[ "$case_name" == go2_old_trot ]]; then
      checkpoint=logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt
    elif [[ "$case_name" == go2_ppo_* ]]; then
      checkpoint=/home/yifeiy/rl_sar/policy/go2/ppo_velocity/policy_inference.pt
    else
      checkpoint=/home/yifeiy/rl_sar/policy/go2/supermimic_velocity/policy_inference.pt
    fi
    cmd+=(--ckpt "$checkpoint" "${go2_common[@]}")
    if [[ "$case_name" == *_yaw ]]; then cmd+=(--vx 0 --vy 0 --yaw-rate 0.4)
    elif [[ "$case_name" == *_side ]]; then cmd+=(--vx 0 --vy 0.3 --yaw-rate 0)
    else cmd+=(--vx 0.3 --vy 0 --yaw-rate 0); fi
    ;;
  m1_ppo_L|m1_sm_L|m1_ppo_M|m1_sm_M|m1_ppo_far|m1_sm_far|m1_ppo_low|m1_sm_low|m1_ppo_random|m1_sm_random)
    if [[ "$case_name" == m1_ppo_* ]]; then checkpoint="logs/m1_ppo/$m1_tag/policy_inference.pt"
    else checkpoint="logs/m1_reward/$m1_tag/ckpt_latest.pt"; fi
    cmd+=(--ckpt "$checkpoint" "${m1_common[@]}")
    if [[ "$case_name" == *_L ]]; then cmd+=(--m1-task L --vx 0.12 --vy 0 --yaw-rate 0)
    else
      cmd+=(--m1-task M)
      if [[ "$case_name" == *_far ]]; then cmd+=(--m1-target-world 2.5 0 0.8)
      elif [[ "$case_name" == *_low ]]; then cmd+=(--m1-target-world 1.5 0 0.15)
      elif [[ "$case_name" != *_random ]]; then cmd+=(--m1-target-world 1.5 0 0.8); fi
    fi
    ;;
  help|--help|-h)
    printf '%s\n' 'Usage: bash docs/group_meeting_20261106/scripts/play_demo.sh CASE [--print]' \
      'Go2: go2_{ppo,sm}_{forward,yaw,side}, go2_old_trot' \
      'M1: m1_{ppo,sm}_{L,M,far,low,random}' \
      'Explicit CASE starts one viewer. --print only prints the exact command. No training.'
    exit 0 ;;
  *) printf 'Unknown case: %s\n' "$case_name" >&2;exit 2 ;;
esac
[[ -f "$checkpoint" ]] || { printf 'Checkpoint missing: %s\n' "$checkpoint" >&2;exit 3; }
if [[ "$print_only" == --print ]]; then printf '%q ' "${cmd[@]}";printf '\n';exit 0;fi
[[ -z "$print_only" ]] || { printf 'Unknown option: %s\n' "$print_only" >&2;exit 2; }
exec "${cmd[@]}"
