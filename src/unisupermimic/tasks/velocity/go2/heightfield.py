"""Immutable collision-aligned terrain, queried in WORLD coordinates.

Only the explicitly named static terrain geom is read. No backend internals,
robot geometry, runtime ray casting, learned encoder, or actor inputs are used.
MuJoCo splits each cell along its (0,0)--(1,1) diagonal; matching that split
is essential at stairs. Derivatives are piecewise affine, undefined at edges.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

SCAN_X = tuple(-0.42 + 0.12 * i for i in range(11))
SCAN_Y = tuple(-0.36 + 0.12 * j for j in range(7))


@dataclass(frozen=True)
class TerrainHeightField:
    heights: torch.Tensor  # y,x; immutable by convention
    center: tuple[float, float, float]
    half_size: tuple[float, float]
    yaw: float = 0.0
    fingerprint: str = "analytic"
    constant: float | None = None

    @classmethod
    def flat(cls, device="cpu", height=0.0):
        return cls(
            torch.zeros(2, 2, device=device),
            (0.0, 0.0, height),
            (1.0, 1.0),
            fingerprint=f"plane:{height}",
            constant=height,
        )

    @classmethod
    def from_mujoco(cls, model, *, geom_name="go2_terrain", device="cpu", metadata=None):
        """Copy compiled collision data through public MuJoCo APIs, once."""
        import mujoco

        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid < 0:
            raise ValueError(f"Missing explicit terrain geom {geom_name!r}")
        if model.geom_bodyid[gid] != 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_HFIELD:
            raise ValueError("Terrain must be a static world-body heightfield")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        rotation = data.geom_xmat[gid].reshape(3, 3)
        if not np.allclose(rotation[2], (0, 0, 1), atol=1e-8):
            raise NotImplementedError("Tilted heightfield geoms are not supported")
        hid = model.geom_dataid[gid]
        nr, nc, adr = (
            int(a[hid]) for a in (model.hfield_nrow, model.hfield_ncol, model.hfield_adr)
        )
        size = model.hfield_size[hid].copy()
        center = tuple(float(v) for v in data.geom_xpos[gid])
        heights = model.hfield_data[adr : adr + nr * nc].copy().reshape(nr, nc) * size[2]
        digest = hashlib.sha256()
        digest.update(heights.tobytes())
        digest.update(size.tobytes())
        digest.update(data.geom_xmat[gid].tobytes())
        digest.update(json.dumps((center, metadata), sort_keys=True, default=str).encode())
        return cls(
            torch.as_tensor(heights, device=device, dtype=torch.float32),
            center,
            (float(size[0]), float(size[1])),
            float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            digest.hexdigest(),
        )

    def to(self, device):
        from dataclasses import replace

        return replace(self, heights=self.heights.to(device))

    def _coordinates(self, xy):
        offset = xy - xy.new_tensor(self.center[:2])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        x = c * offset[..., 0] + s * offset[..., 1]
        y = -s * offset[..., 0] + c * offset[..., 1]
        return torch.stack((x, y), -1) / xy.new_tensor(self.half_size)

    def contains(self, xy):
        if self.constant is not None:
            return torch.isfinite(xy).all(-1)
        return (self._coordinates(xy).abs() <= 1).all(-1) & torch.isfinite(xy).all(-1)

    def ground_height(self, world_xy):
        if self.constant is not None:
            return world_xy[..., 0] * 0 + self.constant
        uv = (self._coordinates(world_xy) + 1) * 0.5
        # Bounded extrapolation keeps immature imagined trajectories finite;
        # contains() is separately used to terminate trajectories out of map.
        uv = uv.clamp(0, 1)
        nr, nc = self.heights.shape
        x, y = uv[..., 0] * (nc - 1), uv[..., 1] * (nr - 1)
        ix, iy = x.floor().long().clamp(0, nc - 2), y.floor().long().clamp(0, nr - 2)
        u, v = x - ix, y - iy
        h00, h10 = self.heights[iy, ix], self.heights[iy, ix + 1]
        h01, h11 = self.heights[iy + 1, ix], self.heights[iy + 1, ix + 1]
        height = torch.where(
            u >= v, h00 + u * (h10 - h00) + v * (h11 - h10), h00 + v * (h01 - h00) + u * (h11 - h01)
        )
        return height + self.center[2]

    def scan_points(self, base_position, base_rotation):
        x, y = torch.meshgrid(
            base_position.new_tensor(SCAN_X), base_position.new_tensor(SCAN_Y), indexing="ij"
        )
        offsets = torch.stack((x.flatten(), y.flatten()), -1)
        yaw = torch.atan2(base_rotation[..., 1, 0], base_rotation[..., 0, 0])
        c, s = yaw.cos()[..., None], yaw.sin()[..., None]
        wx = c * offsets[:, 0] - s * offsets[:, 1]
        wy = s * offsets[:, 0] + c * offsets[:, 1]
        return base_position[..., None, :2] + torch.stack((wx, wy), -1)

    def local_scan(self, base_position, base_rotation, *, sensor_grad=True):
        if not sensor_grad:
            base_position, base_rotation = base_position.detach(), base_rotation.detach()
        points = self.scan_points(base_position, base_rotation)
        return base_position[..., 2, None] - self.ground_height(points)

    def foot_clearance(self, foot_positions, radius=0.022):
        return foot_positions[..., 2] - self.ground_height(foot_positions[..., :2]) - radius
