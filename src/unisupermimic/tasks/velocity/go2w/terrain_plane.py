"""Differentiable terrain trend from the existing yaw-aligned 77-point scan."""

import math
from functools import lru_cache

import torch

from unisupermimic.tasks.velocity.go2.heightfield import SCAN_X, SCAN_Y


@lru_cache(maxsize=16)
def plane_coefficients(device, dtype):
    # Cold path only. Broad Gaussian weighting spans several .4m treads, with
    # support-area points carrying more weight than the forward edge of the scan.
    x, y = torch.meshgrid(
        torch.tensor(SCAN_X, dtype=torch.float64),
        torch.tensor(SCAN_Y, dtype=torch.float64),
        indexing="ij",
    )
    x, y = x.flatten(), y.flatten()
    design = torch.stack((x, y, torch.ones_like(x)), -1)
    weight = torch.exp(-0.5 * ((x / 0.42).square() + (y / 0.28).square()))
    weighted = design.T * weight
    ridge = torch.diag(torch.tensor([1e-6, 1e-6, 0.0], dtype=torch.float64))
    fit = torch.linalg.solve(weighted @ design + ridge, weighted)
    return fit.to(device=device, dtype=dtype)


def terrain_plane_normal(terrain_scan, base_rotation, max_target_tilt_deg=35.0):
    """Fit h=-scan+constant; return bounded upward normal in world coordinates.

    All 77 samples must be finite; invalid scans fall back to world-up. The
    caller terminates out-of-atlas states separately. No target detach is used.
    """
    if terrain_scan.shape[-1] != 77:
        raise ValueError("Terrain plane requires the 11x7 scan")
    fit = plane_coefficients(terrain_scan.device, terrain_scan.dtype)
    valid = torch.isfinite(terrain_scan).all(-1, keepdim=True)
    height = -torch.nan_to_num(terrain_scan, nan=0.0, posinf=0.0, neginf=0.0)
    # Remove the base-height constant before fitting for numerical stability.
    height = height - height.mean(-1, keepdim=True)
    slope = (height @ fit.T)[..., :2]
    cap = math.tan(math.radians(max_target_tilt_deg))
    slope = slope * (cap / slope.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
    slope = torch.where(valid, slope, 0.0)
    normal = torch.cat((-slope, torch.ones_like(slope[..., :1])), -1)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    yaw = torch.atan2(base_rotation[..., 1, 0], base_rotation[..., 0, 0])
    c, s = yaw.cos(), yaw.sin()
    nx, ny, nz = normal.unbind(-1)
    return torch.stack((c * nx - s * ny, s * nx + c * ny, nz), -1)
