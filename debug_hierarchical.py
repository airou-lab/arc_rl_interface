#!/usr/bin/env python3
"""
Debug script to test hierarchical wrapper chain
"""

import numpy as np
from unity_dense_env import UnityDenseEnv
from wrappers.waypoint_tracking_wrapper import WaypointTrackingWrapper

print("Testing wrapper chain for hierarchical policy...")
print("=" * 60)

# Test 1: Base environment
print("\n1. Testing UnityDenseEnv alone...")
try:
    env = UnityDenseEnv(
        host="127.0.0.1",
        port=5556,
        img_width=84,
        img_height=84,
        max_steps=500,
        verbose=True
    )
    print("Created UnityDenseEnv")

    obs, info = env.reset()
    print(f"Reset successful! Obs keys: {obs.keys()}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"Vec shape: {obs['vec'].shape}")

    # Take a step
    obs, r, d, t, info = env.step([0.0, 0.3, 0.0])
    print(f"Step successful! Reward: {r}")

    env.close()
    print("UnityDenseEnv works!")

except Exception as e:
    print(f"UnityDenseEnv failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)

# Test 2: With WaypointTrackingWrapper
print("\n2. Testing with WaypointTrackingWrapper...")
try:
    env = UnityDenseEnv(
        host="127.0.0.1",
        port=5556,
        img_width=84,
        img_height=84,
        max_steps=500,
        verbose=True
    )
    print("Created base env")

    # Wrap with waypoint tracker
    env = WaypointTrackingWrapper(env)
    print("Added WaypointTrackingWrapper")

    # Try reset
    print("Attempting reset through wrapper...")
    obs, info = env.reset()
    print(f"Reset successful! Obs keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")

    # Check if wrapper added anything
    if hasattr(env, 'trajectory_buffer'):
        print(f"Trajectory buffer exists: {env.trajectory_buffer is not None}")

    # Take a step
    obs, r, d, t, info = env.step([0.0, 0.3, 0.0])
    print(f"Step successful! Reward: {r}")

    env.close()
    print("WaypointTrackingWrapper works!")

except Exception as e:
    print(f"WaypointTrackingWrapper failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("\nDebug complete. Check which layer failed above.")