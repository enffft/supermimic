# M1 V3.3 Action Contract audit

## L Reset profile

The V3.3 L bank is not the earlier V2.5 / low manipulation stance. All 999 L motions start at root z 0.470 m in the `nominal_high/m1_standing_pose_v2_3/carry` profile, with front hip/knee `+0.881283/-1.618489` rad, rear `-0.881283/+1.618489` rad, and Piper carry `[0, 0.5, -0.4, 0.3, 0, 0]` rad. A named `dataset_v3_3_l_high_carry` profile carries these source values. A constructor check rejects any action envelope that excludes a position joint's default angle.

M continues to use `dataset_carry` (initial root z 0.389286 m); L's Reset change is task-specific.

## Asymmetric mapping retained

Contract v3 keeps the existing per-joint asymmetric semantic:

```text
action -1 → lower target
action  0 → saved default pose
action +1 → upper target
```

Each target is first clamped to its serialized position envelope, then guarded by the simulator URDF soft limit. Wheel actions remain angular-velocity targets `clip(action, -1, 1) * wheel_scale_rad_s` (5 rad/s at full action). Action inversion used by both fixed evaluation and reference-control audit is centralized in `inverse_m1_setpoint_action`.

## V3.3 position target envelopes

The intervals below are absolute radians, computed from V3.3 mixed-bank P1/P99 plus 0.05 rad and intersected with URDF 90% soft limits and Piper SDK bounds. Wheel slots are velocity-controlled and have no position interval. The full bank contains 923,892 frames.

| Joint | V3 target interval (rad) | L default (rad) |
|---|---:|---:|
| FAR_ABAD | [-0.36324, 0.43849] | 0 |
| FAR_HIP | [0.61401, 1.46816] | 0.88128 |
| FAR_KNEE | [-2.52090, -1.19558] | -1.61849 |
| FBL_ABAD | [-0.40312, 0.39113] | 0 |
| FBL_HIP | [0.60863, 1.47425] | 0.88128 |
| FBL_KNEE | [-2.52090, -1.14031] | -1.61849 |
| RAR_ABAD | [-0.38559, 0.41193] | 0 |
| RAR_HIP | [-1.52413, -0.64906] | -0.88128 |
| RAR_KNEE | [1.14922, 2.52090] | 1.61849 |
| RBL_ABAD | [-0.43197, 0.37001] | 0 |
| RBL_HIP | [-1.53056, -0.65860] | -0.88128 |
| RBL_KNEE | [1.17965, 2.52090] | 1.61849 |
| Piper q1 | [-0.46264, 0.51819] | 0 |
| Piper q2 | [0.29709, 2.78807] | 0.5 |
| Piper q3 | [-1.72607, -0.32987] | -0.4 |
| Piper q4 | [-0.39860, 0.41906] | 0.3 |
| Piper q5 | [-0.28104, 0.21721] | 0 |
| Piper q6 | [-1.05, 0.05] | 0 |

The wider q6 interval reflects V3.3's real q6 variation (future mean approximately -0.183 rad, std approximately 0.379 rad); it remains limited by the SDK/URDF intersection and is not expanded to the entire SDK range. The front and rear knee envelopes include the new L default, avoiding out-of-range default targets.

## Raw Motion Bank envelope statistics

These counts classify raw reference joint frames, not generated CVAE predictions. A frame may be within physical limits but outside the training action envelope. “Not dynamically validated” is true for the source bank in all three task groups.

| Task | Outside action envelope | URDF violation | SDK violation | Velocity violation | Dynamics validation |
|---|---:|---:|---:|---:|---|
| L | 59,463 / 6,060,978 (0.9811%) | 0 | 0 | 0 | not validated |
| M | 72,267 / 4,441,558 (1.6271%) | 88 (0.00198%) | 88 (0.00198%) | 2,351 (0.05293%) | not validated |
| LM | 48,697 / 9,823,088 (0.4957%) | 0 | 0 | 0 | not validated |

The V3.3 dataset manifest reports 899 L quick-screen/partial-collision cases and 100 L full-trajectory collision audits; M has 1000 full audits; LM has 448 full audits and 825 quick screens. These assurance labels are preserved in the source metadata. No actuator speed/position limits were increased to force complete reference coverage.

The v3 Save→Load test checks endpoints and the saved wheel string through a round-trip. A separate physical check ran Stop, +0.12 m/s and -0.08 m/s with actual wheel contact and MuJoCo steps; see [integration audit](M1_V33_PRIOR_INTEGRATION_AUDIT.md#physical-interface-smoke-check).
