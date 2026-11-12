"""Explicit tensor layouts; legacy robots retain equal position/velocity widths."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StateLayout:
    joint_position_dim: int
    joint_velocity_dim: int
    action_dim: int
    # Velocity columns that integrate the stored joint positions (no wheel angles).
    position_velocity_indices: tuple[int, ...] = ()
    version: str = "joint_state_v1"

    def __post_init__(self):
        if min(self.joint_position_dim, self.joint_velocity_dim, self.action_dim) <= 0:
            raise ValueError("State/action widths must be positive")
        indices = self.position_velocity_indices
        if not indices:
            indices = tuple(range(self.joint_position_dim))
            object.__setattr__(self, "position_velocity_indices", indices)
        if (len(indices) != self.joint_position_dim or len(set(indices)) != len(indices)
                or any(i < 0 or i >= self.joint_velocity_dim for i in indices)):
            raise ValueError("Each stored position needs one valid velocity column")

    @classmethod
    def legacy(cls, num_joints):
        return cls(num_joints, num_joints, num_joints)

    @property
    def state_dim(self):
        return 13 + self.joint_position_dim + self.joint_velocity_dim

    @property
    def increment_dim(self):
        return self.state_dim - 1

    @property
    def invariant_dim(self):
        return self.state_dim

    @property
    def increment_sizes(self):
        return (3, 3, 3, 3, self.joint_position_dim, self.joint_velocity_dim)


@dataclass(frozen=True)
class DirectPolicySpec:
    """Term-major history, fixed sensor scales, and a separately supplied goal."""

    widths: tuple[int, ...] = (12, 12, 3, 3, 12)
    scales: tuple[float, ...] = (1.0, 0.2, 0.5, 1.0, 1.0)
    history: int = 5
    goal_dim: int = 3
    action_dim: int = 12

    def __post_init__(self):
        if len(self.widths) != len(self.scales) or any(w <= 0 for w in self.widths):
            raise ValueError("Observation terms require aligned positive widths and scales")
