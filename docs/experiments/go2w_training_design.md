# Go2W unified curriculum: initial audit (2026-11-16)

Status: **stopped at the physical H16 gradient gate**. WM-only initialization,
optional 7D Actor mode and differentiable Critic scan are implemented and tested.
Ten-level curriculum, new rewards/history bank and formal training are incomplete.
Initial-state descriptions below deliberately preserve the pre-change audit.

## Source and recovery

- Original repository: `/home/yifeiy/ws_go2_unilab/Uni-SuperMimic`.
- Original HEAD and fetched `user/experiment/go2-velocity-unilab`:
  `bc4d172d5c93239b0e50e32e3481cd83601dff76`.
- `origin` is Motphys/Uni-SuperMimic; the requested origin branch does not exist.
  `user` is enffft/supermimic. Both remotes were fetched successfully.
- Original dirty tree preserved without stashing, resetting or cleaning. Patch,
  complete modified/untracked file archive, original status and isolation manifest:
  `../runs/go2w_unified_audit_20261116/`.
- Isolated worktree: `../Uni-SuperMimic-unified`, branch
  `experiment/go2w-unified-curriculum`, preserved local baseline commit
  `6784ac8ed6f0f6f377063fa6b32633529b5f678f`.
- Actual WM source is **the omni two-hour snapshot**, based on the continual
  snapshot plus task/config overlays. Neither public HEAD nor the current dirty
  tree alone describes that run. Per-file comparisons are in
  `docs/validation/go2w_unified_curriculum/initial_audit.json`.

## Existing interfaces and dynamics

- State: 41 floats, world position, wxyz quaternion, world linear/angular
  velocity, 12 leg positions, 12 leg velocities and 4 wheel velocities. Wheel
  angles excluded. Logical order FL/FR/RL/RR, hip/thigh/calf, then four wheels;
  physical simulator order is resolved by joint names.
- Actor: 5 x 50 proprioception; 3 velocity goals; 253 inputs, 16 outputs,
  fixed term scales and direct single MLP. No height scan or target in Actor.
  Required 7D goal / 257D input is not yet implemented at initial audit.
- WM: structured 134 -> 2048 -> 2048 -> 2048 -> 40. Inputs are 41 invariant
  features + 16 actions + 77 scan, empty nominal physics context. Every predicted
  pose queries collision-aligned static terrain. Increment and output statistics
  have distinct semantics. Control dt .02, simulator dt .005.
- Critic: invariant state + context + velocity goal. Terrain conditioning is
  currently a rejected placeholder in shared learner; it must be implemented.
- Full range controller already exists in dirty local task, absent from public
  HEAD. `a=-1/0/+1` maps to soft lower/default/upper position. Range fraction
  .95, default `(0.1,.8,-1.5,-.1,.8,-1.5,.1,1,-1.5,-.1,1,-1.5)`.
  Leg PD 40/1; wheel velocity scale 15, velocity gain .5, kp=0.
  Torque limits `(23.7,23.7,45.43)*4 + (15,)*4`. One law `mixed_torque`
  is called at simulator substeps and by analytic imagined torque.
- XML SHA: `7425da3e392362dd44999c6835c571e76c3f1e2bd552c8fd29dfb00f2df2fc38`;
  collision wheel SHA: `51c95e80dcd41c0cf1a9575d4d7c2c89855848f7fa2c4bf28f2f10c2e3d639e2`.
  Wheel axes are local +Y; world rolling sign/radius require physical calibration.

## Current curriculum, rewards and replay

- Six fixed domains: flat/rough/slope up/down/stairs up/down, eight levels,
  .03*level stairs. Persistent coverage reserves 512/4096 environments but is
  fixed to the existing domain atlas. No distinct single step or target metadata.
- Mastery is per environment with episode streaks, not requested per-skill
  64–128 episode windows. Dynamic ten-level statistics are not implemented.
- Real reward: manager reward -> MDP transition reward -> task `reward_core`.
  Imagined reward: learner -> same MDP transition reward -> same task core,
  with recorded command domain. Endpoint torque proxy differs from measured
  last-substep torque; this existing discrepancy needs explicit parity handling.
- Existing reward has vx/vy/yaw Gaussian tracking, optional tracking error cost,
  stop, support plane alignment, height and motor/action regularization, dt=.02.
  Wheel rolling consistency, traversal potential and once-only success absent.
- Replay: device-resident time x env ring, s/s1/a/a_prev/xi/r/f/ep_step/done,
  optional goal and terrain/command labels. 8-step samples mask transitions
  after first terminal; stored tensors can contain later episode rows with zero
  loss mask. Actor history pads at reset. Strict no-crossing history bank absent.
- Replay target (7D goals, existing metadata only) is **2,281,701,376 bytes**
  (2.125 GiB), excluding new targets/bank/sampling copies. Per-field dtype/shape/
  bytes recorded in initial audit JSON. Current implementation hardcodes eight
  levels in replay, diagnostics, sampler and terrain. No history bank exists.

## Candidate WM and initialization

- Read-only source: `../artifacts/world_models/go2w_conditioned.pt`.
- **File** SHA256:
  `3c3c4d1c5e70bb12c8cc68274fda39003e465c6ec1fbe745c0729f9f7d028610`.
- **Tensor** SHA256 (the value quoted by the historical report):
  `ed04040096a0b0254fb58ef18ffa65ef7bf2c1f07e15fa7a2e6df36c7428c490`.
- All tensors finite. Input/output/increment normalizer counts all 61,677,568.
  Tensor hash exactly equals original omni run `training/last.pt`, iteration 4765.
  The older continual run publish manifest refers to a different, earlier model.
- Existing `load_world_model` restores Adam and update count; experiment init
  also loads Actor. Neither implements this task's fresh-optimizer WM-only init.
  A separate strict path is required; existing resume semantics must stay intact.
- No source asset writes, policy bootstrap or old Replay import are permitted.

## Execution and resources

- Interpreter `/home/yifeiy/miniconda3/envs/go2_unilab/bin/python`, Torch
  2.11.0+cu126, MuJoCo 3.11.0, local UniLab dependency.
- Use `PYTHONPATH=src` in this worktree to prevent editable install resolving
  to original dirty source. Existing entrypoint:
  `python -m unisupermimic.scripts.train_supermimic --help` (Hydra).
  Experiment wrapper `tools/train_go2w_experiment.py --help` exists but is not
  suitable for requested formal run: restores WM optimizer, can overwrite source
  asset, lacks requested features, starts timer before loop.
- Existing tests: `python -m pytest tests/tasks/test_go2w_*.py
  tests/rl/algos/test_supermimic_*.py tests/rl/algos/test_learner_batching.py`.
  Physical tests marked slow require explicit `-m slow`.
- RTX 4070 SUPER: 12,282 MiB; initial used 2,265 MiB. Two unrelated resident
  vision processes use 1,074 and 314 MiB; do not terminate them.
  RAM 15 GiB, available 6.5 GiB; swap 12/19 GiB used; disk 151 GiB free.
  Replay checkpoint copies and CPU historical pool must account for this pressure.
- Formal targets remain 4096 env, 1024 ring, H8 WM/H16 imagination,
  batch 2048/2048, 10 WM updates, 14400 seconds of actual train loop.
  No target has been reduced and no training result is claimed by this audit.

## Implemented preflight and stopping evidence

- `checkpoint.initialize_wm_only` validates file and tensor SHA256, complete
  tensor keys/shapes/dtypes, finite/nonnegative statistics, architecture and exact
  dynamics contract before transferring. Actor/Critic, EMA, all optimizers,
  replay and counters remain fresh. Normalizer counts are inherited unchanged.
- `terrain_mode_conditioning` is opt-in so old Go2W contracts remain unchanged.
  New MDP policy spec uses 7D conditions (257D total Actor input). Real command
  sampling, Replay and imagined goal preserve the same four mode bits.
  The mode label is supplied by the task manager, not inferred by the policy.
- Shared learner now implements the opt-in terrain Critic with a fresh scan
  at each predicted pose, preserving Actor derivatives through frozen Critic
  parameters. New Critic input is 125D; WM stays 134D/40D.
- Expanded regression: 116 passed, 2 skipped; separate MuJoCo physics test:
  1 passed. These are not the complete ten mandatory task gates.
- A 32-env probe on flat terrain compares central physical finite differences
  and WM autograd for vx/vy/yaw at H1/H8/H16. Only trajectories before episode
  boundaries are accepted. H16 yaw has negative direction cosine at all three
  small perturbations (.001/.002/.005) after five initial neutral steps.
- A separate 32-env diagnostic collected 8192 new fresh-Actor transitions and
  performed 200 WM-only updates. Actor and Critic hashes did not change; all
  source tensors/statistics were inherited before the fresh Adam updates.
  Held-out neutral probes still fail, now consistently for H16 vy as well.
  This candidate is **not approved for training or publication**.
- The local direction gate uses cosine > 0 when the physical gradient norm is
  above 1e-3. This is a conservative newly explicit acceptance criterion.
  Contact finite differences are perturbation-dependent and cannot prove the
  exact derivative of a smooth plant; repeated negative results block approval
  rather than justify a claim that every imagined trajectory has wrong gradients.
- Original worktree status and original WM file SHA256 were verified unchanged.
  No 4096-env smoke or four-hour run was launched. Complete stopping report:
  `go2w_unified_curriculum_4h_report.md`.
