"""
unity_camera_env.py  —  Back-compatible shim (Passive Only)

This module intentionally re-exports the minimal, strictly passive
Unity client so older code that imports `UnityCameraEnv` keeps working
without any of the previous heuristics (masking, optical flow, etc).
New code should import directly from `live_unity_env`:
    from live_unity_env import LiveUnityEnv, UnityEnvConfig
This file exists only to preserve compatibility with older scripts.
"""
from __future__ import annotations
import warnings
# Re-export the passive live environment
from live_unity_env import LiveUnityEnv as UnityCameraEnv, UnityEnvConfig  # noqa: F401

__all__ = ["UnityCameraEnv", "UnityEnvConfig"]
# Gentle heads-up when this shim is imported
warnings.warn(
    (
        "unity_camera_env.UnityCameraEnv is deprecated and now aliases "
        "live_unity_env.LiveUnityEnv (pure passive RGB). "
        "Please update imports to `from live_unity_env import LiveUnityEnv`."
    ),
    DeprecationWarning,
    stacklevel=2,
)

def create_env(
    host: str = "127.0.0.1",
    port: int = 5556,
    img_size: tuple[int, int] = (84, 84),
    max_steps: int = 500,
    action_repeat: int = 1,
) -> UnityCameraEnv:
    """
    Convenience factory mirroring the older API.
    Returns:
        UnityCameraEnv (actually LiveUnityEnv): passive RGB HWC uint8 observations.
    """
    cfg = UnityEnvConfig(
        host=host,
        port=port,
        img_width=img_size[0],
        img_height=img_size[1],
        max_steps=max_steps,
        action_repeat=action_repeat,
    )
    return UnityCameraEnv(cfg)