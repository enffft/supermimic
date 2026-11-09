"""Read / write the full floating-base simulator state (qpos | qvel)."""

from __future__ import annotations

import torch
import mujoco

STATE_DIM = 36 + 35
NQ = 36
NV = 35


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.joint_names = tuple(env.scene["robot"].joint_names)
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        self.nq = int(self.sim.data.qpos.shape[-1])
        self.nv = int(self.sim.data.qvel.shape[-1])
        if self.nq < 7 or self.nv < 6 or self.nq - 7 != self.nv - 6:
            raise ValueError(f"expected floating base plus scalar joints, got nq={self.nq}, nv={self.nv}")
        model = self.sim.mj_model
        entity_prefix = "robot/"
        model_joint_names: set[str] = set()
        qpos_addr: dict[str, int] = {}
        dof_addr: dict[str, int] = {}
        joint_type: dict[str, int] = {}
        free_joints: list[tuple[int, int]] = []
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                free_joints.append((int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])))
                continue
            model_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            # mjlab namespaces compiled model names with the scene entity (``robot/``),
            # while the articulation, RobotSpec, and Actor use the bare URDF joint name.
            name = model_name.removeprefix(entity_prefix)
            model_joint_names.add(name)
            qpos_addr[name] = int(model.jnt_qposadr[joint_id])
            dof_addr[name] = int(model.jnt_dofadr[joint_id])
            joint_type[name] = int(model.jnt_type[joint_id])
        if model_joint_names != set(self.joint_names):
            raise ValueError(
                f"simulator joints differ from mjlab articulation; "
                f"missing={set(self.joint_names)-model_joint_names}, extra={model_joint_names-set(self.joint_names)}"
            )
        if free_joints != [(0, 0)]:
            raise ValueError(f"expected a single floating base at qpos/dof 0, got {free_joints}")
        self.joint_qpos_addr = tuple(qpos_addr[name] for name in self.joint_names)
        self.joint_dof_addr = tuple(dof_addr[name] for name in self.joint_names)
        if any(joint_type[name] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for name in self.joint_names):
            raise ValueError("StateIO supports scalar hinge/slide joints only")
        self.state_dim = self.nq + self.nv

    def read(self) -> torch.Tensor:
        """(N, nq+nv) copy of [qpos | qvel]."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        qpos = self.sim.data.qpos[:]
        qvel = self.sim.data.qvel[:]
        qpos_state = torch.cat([qpos[:, :7], qpos[:, self.joint_qpos_addr]], dim=-1)
        qvel_state = torch.cat([qvel[:, :6], qvel[:, self.joint_dof_addr]], dim=-1)
        s = torch.cat([qpos_state, qvel_state], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        qpos = torch.zeros_like(state[:, :self.nq])
        qvel = torch.zeros_like(state[:, self.nq:self.nq + self.nv])
        qpos[:, :7] = state[:, :7]
        qvel[:, :6] = state[:, self.nq:self.nq + 6]
        for joint_i, address in enumerate(self.joint_qpos_addr):
            qpos[:, address] = state[:, 7 + joint_i]
        for joint_i, address in enumerate(self.joint_dof_addr):
            qvel[:, address] = state[:, self.nq + 6 + joint_i]
        self.sim.data.qpos[env_ids] = qpos
        self.sim.data.qvel[env_ids] = qvel
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
