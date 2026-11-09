"""State representation for the world model (plan D1–D3).

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(29), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(29)]            (71, MuJoCo layout, origin-free)

Canonical features (network input, heading-invariant, 71):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(29), joint_vel(29)]

Network output ``delta`` (70) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(29)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(29)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

NJ = 29  # backwards-compatible G1 defaults; concrete specs use StateSpec.nj.
STATE_DIM = 13 + 2 * NJ
FEAT_DIM = STATE_DIM
DELTA_DIM = 12 + 2 * NJ


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    num_joints: int = NJ
    # Wheel coordinates are kept in the physical state but treated as angles in errors/features.
    wheel_indices: tuple[int, ...] = ()
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) for friction, encoder bias, com offset.
    xi_offset: tuple[float, ...] = field(default_factory=lambda: (0.75,) + (0.0,) * 29 + (0.0,) * 3)
    xi_scale: tuple[float, ...] = field(default_factory=lambda: (0.45,) + (0.01,) * 29 + (0.05,) * 3)


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        self.nj = int(cfg.num_joints)
        self.state_dim = 13 + 2 * self.nj
        self.feat_dim = self.state_dim
        self.delta_dim = 12 + 2 * self.nj
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])  # noqa: E731
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, self.nj, self.nj)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, self.nj, 3, 3, self.nj)).to(device=device, dtype=dtype)
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, self.nj, self.nj)).to(device=device, dtype=dtype)
        # Context is [friction, one encoder bias per joint, COM offset xyz].  Adapt the
        # historical 29-joint defaults without changing the G1 numerical path.
        xi_dim = self.nj + 4
        if len(cfg.xi_offset) == 33 and self.nj != 29:
            xi_offset = cfg.xi_offset[:1] + (0.0,) * self.nj + cfg.xi_offset[-3:]
            xi_scale = cfg.xi_scale[:1] + (0.01,) * self.nj + cfg.xi_scale[-3:]
        else:
            xi_offset, xi_scale = cfg.xi_offset, cfg.xi_scale
        if len(xi_offset) != xi_dim or len(xi_scale) != xi_dim:
            raise ValueError(f"xi normalization must have {xi_dim} entries for {self.nj} joints")
        self.xi_offset = torch.tensor(xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(xi_scale, device=device, dtype=dtype)
        self.wheel_indices = tuple(int(i) for i in cfg.wheel_indices)
        if any(i < 0 or i >= self.nj for i in self.wheel_indices):
            raise ValueError("wheel_indices must refer to a joint in the state")

    # ------------------------------------------------------------------ split / features

    def split(self, s: torch.Tensor):
        j = 7 + self.nj
        return s[..., 0:3], s[..., 3:7], s[..., 7:j], s[..., j:j + 3], s[..., j + 3:j + 6], s[..., j + 6:j + 6 + self.nj]

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features (…, 13+2*nj)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        if self.wheel_indices:
            wi = torch.as_tensor(self.wheel_indices, device=qj.device)
            qj = qj.clone()
            qj[..., wi] = torch.atan2(torch.sin(qj[..., wi]), torch.cos(qj[..., wi]))
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    def delta_target(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Invert ``integrate`` and return its pre-``out_scale`` real-transition target.

        This is used only to update optional RunningNorm statistics from newly
        collected simulator transitions. Keeping the fixed ``out_scale`` outside
        this target means the learned normalized delta is first denormalized back
        to the exact units consumed by ``integrate``.
        """
        j = self.nj
        p, q, qj, v, w, qd = self.split(s)
        p1, q1, qj1, v1, w1, qd1 = self.split(s_next)
        q = quat_normalize(q)
        q1 = quat_normalize(q1)
        hq = heading_quat(q)
        dt = self.dt if self.cfg.explicit_integration else 0.0

        dv_h = quat_apply_inverse(hq, v1 - v)
        dw_b = w1 - w
        dqd = qd1 - qd
        dp_h = quat_apply_inverse(hq, p1 - p - dt * v1)
        dtheta = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q), q1)) - dt * w1
        dq = qj1 - qj - dt * qd1
        if self.wheel_indices:
            wi = torch.as_tensor(self.wheel_indices, device=dq.device)
            dq = dq.clone()
            dq[..., wi] = torch.atan2(torch.sin(dq[..., wi]), torch.cos(dq[..., wi]))
        physical = torch.cat([dv_h, dw_b, dqd, dp_h, dtheta, dq], dim=-1)
        return physical / self.out_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, 70) to state ``s`` (…, 71)."""
        d = delta * self.out_scale
        j = self.nj
        dv_h, dw_b, dqd, dp_h, dth_b, dq = d[..., 0:3], d[..., 3:6], d[..., 6:6+j], d[..., 6+j:9+j], d[..., 9+j:12+j], d[..., 12+j:12+2*j]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        qj_err = qj1 - qj2
        if self.wheel_indices:
            wi = torch.as_tensor(self.wheel_indices, device=qj_err.device)
            qj_err = qj_err.clone()
            qj_err[..., wi] = torch.atan2(torch.sin(qj_err[..., wi]), torch.cos(qj_err[..., wi]))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj_err, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
