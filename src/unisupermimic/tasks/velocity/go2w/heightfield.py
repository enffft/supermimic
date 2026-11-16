"""Collision-aligned static scan for heightfields plus exact vertical stair boxes."""

import hashlib
from dataclasses import fields

import torch

from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField

from .curriculum import SINGLE_HEIGHTS, STAIR_HEIGHTS


class TraversalHeightField(TerrainHeightField):
    @classmethod
    def from_field(cls, field, model):
        import mujoco

        boxes = []
        for gid in range(model.ngeom):
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith(
                "traversal_"
            ):
                if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX or model.geom_bodyid[gid] != 0:
                    raise ValueError("Traversal terrain requires static boxes")
                boxes.append((model.geom_pos[gid].tolist(), model.geom_size[gid].tolist()))
        if len(boxes) != 100:
            raise ValueError("Missing vertical stair collision geometry")
        values = {f.name: getattr(field, f.name) for f in fields(field)}
        values["fingerprint"] = hashlib.sha256(
            (field.fingerprint + repr(boxes)).encode()
        ).hexdigest()
        return cls(**values)

    def ground_height(self, world_xy):
        base = super().ground_height(world_xy)
        x, y = world_xy.unbind(-1)
        row = ((x + 40) / 8).floor().long().clamp(0, 9)
        col = ((y + 28) / 8).floor().long()
        local = x + 40 - row * 8
        is_stair = (col >= 3) & (col <= 6) & (x >= -40) & (x < 40)
        single = world_xy.new_tensor(SINGLE_HEIGHTS)[row]
        multi = world_xy.new_tensor(STAIR_HEIGHTS)[row]
        steps = torch.floor((local - 1.4) / 0.4 + 1).clamp(0, 4)
        h = torch.where(col <= 4, single * (local >= 1.4), multi * steps)
        top = torch.where(col <= 4, single, multi * 4)
        h = torch.where((col == 4) | (col == 6), top - h, h)
        return torch.where(is_stair, h, base)
