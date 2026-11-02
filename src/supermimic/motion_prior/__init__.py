"""Differentiable adapters for the AMP-LOCO-M1 conditional VAE prior."""

from supermimic.motion_prior.checkpoint import load_m1_motion_prior
from supermimic.motion_prior.cvae import M1MotionPrior

__all__ = ["M1MotionPrior", "load_m1_motion_prior"]
