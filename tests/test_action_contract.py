from __future__ import annotations

import torch

from supermimic.tasks.action_contract import clamp_executed_action, position_setpoint


def test_deployment_action_map_matches_clipped_real_position_setpoint():
    policy = torch.tensor([[-1.5, -0.5, 0.25, 1.6]])
    scale = torch.tensor([0.25, 0.25, 0.25, 0.25])
    default = torch.tensor([0.1, -0.1, 0.2, -0.2])
    executed = clamp_executed_action(policy, 1.0)
    target = position_setpoint(executed, default, scale)
    torch.testing.assert_close(executed, torch.tensor([[-1.0, -0.5, 0.25, 1.0]]))
    torch.testing.assert_close(target, torch.tensor([[-0.15, -0.225, 0.2625, 0.05]]))
