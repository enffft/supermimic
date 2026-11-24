# Go2W flat kinematic generation report

Generated 2026-11-24T18:59:04.513809+08:00 from commit `8d823e4374701a820d60419988cb58b4371d01f0`.
Generated 30,401 audited frames in 18.31 s (1660.4 frames/s). NPZ size 13.02 MiB. Audit 9.79 s.
Gait `diagonal`. Validation level `kinematic_only`. Physics validated `false`.

| Category | Requested | Saved | Qualified | Candidate | Rejected | Generation (s) |
|---|---:|---:|---:|---:|---:|---:|
| forward_backward | 22 | 22 | 22 | 0 | 0 | 3.81 |
| yaw | 16 | 16 | 0 | 16 | 0 | 2.93 |
| lateral | 14 | 14 | 0 | 14 | 0 | 2.56 |
| forward_yaw | 16 | 16 | 0 | 16 | 0 | 2.95 |
| forward_lateral | 16 | 16 | 0 | 16 | 0 | 2.96 |
| lateral_yaw | 16 | 16 | 0 | 16 | 0 | 2.93 |
| standing | 1 | 1 | 1 | 0 | 0 | 0.17 |

Geometry coverage includes 214 nonadjacent robot pairs, 19 nonwheel ground pairs, and all four wheel meshes per sample. Same-link and direct parent-child pairs are excluded. Continuous-time collision checking is outside this audit.

## Commands requiring review

| Motion | Requested (vx, vy, wz) | Status | Findings |
|---|---|---|---|
| yaw_right_wz_080 | [0.0, 0.0, -0.8] | candidate | dynamic_diagonal_support |
| yaw_right_wz_070 | [0.0, 0.0, -0.7] | candidate | dynamic_diagonal_support |
| yaw_right_wz_060 | [0.0, 0.0, -0.6] | candidate | dynamic_diagonal_support |
| yaw_right_wz_050 | [0.0, 0.0, -0.5] | candidate | dynamic_diagonal_support |
| yaw_right_wz_040 | [0.0, 0.0, -0.4] | candidate | dynamic_diagonal_support |
| yaw_right_wz_030 | [0.0, 0.0, -0.3] | candidate | dynamic_diagonal_support |
| yaw_right_wz_020 | [0.0, 0.0, -0.2] | candidate | dynamic_diagonal_support |
| yaw_right_wz_010 | [0.0, 0.0, -0.1] | candidate | dynamic_diagonal_support |
| yaw_left_wz_010 | [0.0, 0.0, 0.1] | candidate | dynamic_diagonal_support |
| yaw_left_wz_020 | [0.0, 0.0, 0.2] | candidate | dynamic_diagonal_support |
| yaw_left_wz_030 | [0.0, 0.0, 0.3] | candidate | dynamic_diagonal_support |
| yaw_left_wz_040 | [0.0, 0.0, 0.4] | candidate | dynamic_diagonal_support |
| yaw_left_wz_050 | [0.0, 0.0, 0.5] | candidate | dynamic_diagonal_support |
| yaw_left_wz_060 | [0.0, 0.0, 0.6] | candidate | dynamic_diagonal_support |
| yaw_left_wz_070 | [0.0, 0.0, 0.7] | candidate | dynamic_diagonal_support |
| yaw_left_wz_080 | [0.0, 0.0, 0.8] | candidate | dynamic_diagonal_support |
| lateral_right_vy_035 | [0.0, -0.35, 0.0] | candidate | dynamic_diagonal_support |
| lateral_right_vy_030 | [0.0, -0.3, 0.0] | candidate | dynamic_diagonal_support |
| lateral_right_vy_025 | [0.0, -0.25, 0.0] | candidate | dynamic_diagonal_support |
| lateral_right_vy_020 | [0.0, -0.2, 0.0] | candidate | dynamic_diagonal_support |
| lateral_right_vy_015 | [0.0, -0.15, 0.0] | candidate | dynamic_diagonal_support |
| lateral_right_vy_010 | [0.0, -0.1, 0.0] | candidate | dynamic_diagonal_support |
| lateral_right_vy_005 | [0.0, -0.05, 0.0] | candidate | dynamic_diagonal_support |
| lateral_left_vy_005 | [0.0, 0.05, 0.0] | candidate | dynamic_diagonal_support |
| lateral_left_vy_010 | [0.0, 0.1, 0.0] | candidate | dynamic_diagonal_support |
| lateral_left_vy_015 | [0.0, 0.15, 0.0] | candidate | dynamic_diagonal_support |
| lateral_left_vy_020 | [0.0, 0.2, 0.0] | candidate | dynamic_diagonal_support |
| lateral_left_vy_025 | [0.0, 0.25, 0.0] | candidate | dynamic_diagonal_support |
| lateral_left_vy_030 | [0.0, 0.3, 0.0] | candidate | dynamic_diagonal_support |
| lateral_left_vy_035 | [0.0, 0.35, 0.0] | candidate | dynamic_diagonal_support |
| forward_yaw_001 | [0.1, 0, 0.2] | candidate | dynamic_diagonal_support |
| forward_yaw_002 | [0.2, 0, 0.4] | candidate | dynamic_diagonal_support |
| forward_yaw_003 | [0.3, 0, 0.6] | candidate | dynamic_diagonal_support |
| forward_yaw_004 | [0.4, 0, 0.3] | candidate | dynamic_diagonal_support |
| forward_yaw_005 | [0.5, 0, 0.2] | candidate | dynamic_diagonal_support |
| forward_yaw_006 | [0.6, 0, 0.4] | candidate | dynamic_diagonal_support |
| forward_yaw_007 | [0.7, 0, 0.3] | candidate | dynamic_diagonal_support |
| forward_yaw_008 | [0.25, 0, 0.7] | candidate | dynamic_diagonal_support |
| forward_yaw_009 | [0.1, -0.0, -0.2] | candidate | dynamic_diagonal_support |
| forward_yaw_010 | [0.2, -0.0, -0.4] | candidate | dynamic_diagonal_support |
| forward_yaw_011 | [0.3, -0.0, -0.6] | candidate | dynamic_diagonal_support |
| forward_yaw_012 | [0.4, -0.0, -0.3] | candidate | dynamic_diagonal_support |
| forward_yaw_013 | [0.5, -0.0, -0.2] | candidate | dynamic_diagonal_support |
| forward_yaw_014 | [0.6, -0.0, -0.4] | candidate | dynamic_diagonal_support |
| forward_yaw_015 | [0.7, -0.0, -0.3] | candidate | dynamic_diagonal_support |
| forward_yaw_016 | [0.25, -0.0, -0.7] | candidate | dynamic_diagonal_support |
| forward_lateral_001 | [0.1, 0.05, 0] | candidate | dynamic_diagonal_support |
| forward_lateral_002 | [0.15, 0.15, 0] | candidate | dynamic_diagonal_support |
| forward_lateral_003 | [0.2, 0.25, 0] | candidate | dynamic_diagonal_support |
| forward_lateral_004 | [0.3, 0.1, 0] | candidate | dynamic_diagonal_support |
| forward_lateral_005 | [0.4, 0.2, 0] | candidate | dynamic_diagonal_support |
| forward_lateral_006 | [0.5, 0.1, 0] | candidate | dynamic_diagonal_support |
| forward_lateral_007 | [0.6, 0.15, 0] | candidate | dynamic_diagonal_support |
| forward_lateral_008 | [0.25, 0.3, 0] | candidate | dynamic_diagonal_support |
| forward_lateral_009 | [0.1, -0.05, -0.0] | candidate | dynamic_diagonal_support |
| forward_lateral_010 | [0.15, -0.15, -0.0] | candidate | dynamic_diagonal_support |
| forward_lateral_011 | [0.2, -0.25, -0.0] | candidate | dynamic_diagonal_support |
| forward_lateral_012 | [0.3, -0.1, -0.0] | candidate | dynamic_diagonal_support |
| forward_lateral_013 | [0.4, -0.2, -0.0] | candidate | dynamic_diagonal_support |
| forward_lateral_014 | [0.5, -0.1, -0.0] | candidate | dynamic_diagonal_support |
| forward_lateral_015 | [0.6, -0.15, -0.0] | candidate | dynamic_diagonal_support |
| forward_lateral_016 | [0.25, -0.3, -0.0] | candidate | dynamic_diagonal_support |
| lateral_yaw_001 | [0, 0.05, 0.2] | candidate | dynamic_diagonal_support |
| lateral_yaw_002 | [0, 0.1, -0.3] | candidate | dynamic_diagonal_support |
| lateral_yaw_003 | [0, 0.15, 0.4] | candidate | dynamic_diagonal_support |
| lateral_yaw_004 | [0, 0.2, -0.5] | candidate | dynamic_diagonal_support |
| lateral_yaw_005 | [0, 0.25, 0.2] | candidate | dynamic_diagonal_support |
| lateral_yaw_006 | [0, 0.3, -0.3] | candidate | dynamic_diagonal_support |
| lateral_yaw_007 | [0, 0.15, 0.7] | candidate | dynamic_diagonal_support |
| lateral_yaw_008 | [0, 0.1, -0.7] | candidate | dynamic_diagonal_support |
| lateral_yaw_009 | [0.0, -0.05, -0.2] | candidate | dynamic_diagonal_support |
| lateral_yaw_010 | [0.0, -0.1, 0.3] | candidate | dynamic_diagonal_support |
| lateral_yaw_011 | [0.0, -0.15, -0.4] | candidate | dynamic_diagonal_support |
| lateral_yaw_012 | [0.0, -0.2, 0.5] | candidate | dynamic_diagonal_support |
| lateral_yaw_013 | [0.0, -0.25, -0.2] | candidate | dynamic_diagonal_support |
| lateral_yaw_014 | [0.0, -0.3, 0.3] | candidate | dynamic_diagonal_support |
| lateral_yaw_015 | [0.0, -0.15, -0.7] | candidate | dynamic_diagonal_support |
| lateral_yaw_016 | [0.0, -0.1, 0.7] | candidate | dynamic_diagonal_support |
