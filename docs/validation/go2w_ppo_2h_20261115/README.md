# Go2W PPO two-hour experiment

Portable scalar evidence, physical evaluation JSON and exact PPO bridge/runner protocol. Binary weights and full source snapshot remain in the local run recorded in provenance.

Initial Actor, physics, reward and terrain are shared with the prior SuperMimic run. PPO uses native RSL-RL on-policy optimization, not WM imagination. Persistent high-level collection environments contribute PPO gradients.

`comparison.csv`: 55 physical scenarios per method, 12 trials each; `final_extended.json`: 21 flat scenarios. Multiple reset seeds perturb the same atlas; this is one training seed.
