"""Optional real-data source; no imitation loss and no replacement training loop."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class DriverActionContract:
    joints: tuple[str, ...]
    default_position: tuple[float, ...]
    action_scale: float
    control_dt: float
    kp: float
    kd: float
    action_clip: float
    coordinates: str = "body_velocity_command;normalized_joint_position_offset"


class CollectionDriver(Protocol):
    contract: DriverActionContract

    def __call__(self, proprio_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor: ...

    def reset(self, reset_mask: torch.Tensor) -> None: ...
