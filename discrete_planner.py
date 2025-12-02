#!/usr/bin/env python3
from __future__ import annotations
from typing import Sequence, Optional, Tuple
import numpy as np

"""
discrete_planner.py

Heuristic vehicle path-planner over a fixed discrete action set.

- Actions live in the same continuous space as Unity:
    steer   in [-1, 1]   (negative = left, positive = right, per AdvancedDoubleTrackController)
    throttle in [0, 1]
    brake    in [0, 1]

- This module does NOT know about Unity directly. It only consumes:
    vec  : np.ndarray coming from UnityDenseEnv.obs["vec"]
           expected shape >= (5,)
           [0] speed      (m/s)
           [1] yaw_rate   (rad/s)
           [2] last_steer
           [3] last_thr
           [4] last_brk
           [5] lat_err    (optional)
           [6] hdg_err    (optional)
           [7] kappa      (optional)

    goal : tuple(cos, sin, distXZ) coming from info["goal"]
           cos,sin are heading direction of goal in car frame
           distXZ is planar distance in meters

- planner_choose_action(vec, goal) returns a single (3,) float32 action.
"""

# ---------------------------------------------------------------------------
# Discrete action table
# ---------------------------------------------------------------------------

# (steer, throttle, brake)
# You can tune this table later; for now it's a reasonable 15-action grid.
_ACTION_LIST = [
    # straight / speed control
    (0.0, 0.0, 0.0),   # idle
    (0.0, 0.4, 0.0),   # slow straight
    (0.0, 0.8, 0.0),   # fast straight
    (0.0, 0.2, 0.5),   # gentle brake / coast

    # gentle left / right
    (-0.3, 0.4, 0.0),  # gentle left, medium speed
    (-0.3, 0.8, 0.0),  # gentle left, fast
    (0.3, 0.4, 0.0),   # gentle right, medium
    (0.3, 0.8, 0.0),   # gentle right, fast

    # sharper left / right
    (-0.6, 0.3, 0.0),  # sharper left, slower
    (-0.6, 0.0, 0.3),  # sharp left, with brake
    (0.6, 0.3, 0.0),   # sharper right, slower
    (0.6, 0.0, 0.3),   # sharp right, with brake

    # hard brake straight
    (0.0, 0.0, 1.0),

    # mild lane-change-ish
    (-0.15, 0.5, 0.0),
    (0.15, 0.5, 0.0),
]

ACTION_TABLE: np.ndarray = np.asarray(_ACTION_LIST, dtype=np.float32)


def get_action_table() -> np.ndarray:
    """Return a copy of the discrete action table."""
    return ACTION_TABLE.copy()


# ---------------------------------------------------------------------------
# Planner core
# ---------------------------------------------------------------------------

def _safe_goal(goal: Optional[Sequence[float]]) -> Tuple[float, float, float]:
    """
    Normalize goal triple (cos, sin, distXZ).
    If missing, fall back to "straight ahead, far away".
    """
    if goal is None or len(goal) < 3:
        return 1.0, 0.0, 1e3

    c, s, d = goal[0], goal[1], goal[2]
    c = float(0.0 if c is None else c)
    s = float(0.0 if s is None else s)
    d = float(1e3 if d is None else d)

    # normalize cos/sin a bit to avoid weird magnitudes
    norm = np.sqrt(c * c + s * s)
    if norm > 1e-6:
        c /= norm
        s /= norm
    else:
        c, s = 1.0, 0.0

    return c, s, d


def planner_choose_action(vec: np.ndarray, goal: Optional[Sequence[float]]) -> np.ndarray:
    """
    Select the best discrete action (steer, throttle, brake) for the current state.

    Args:
        vec  : 1D ndarray, obs["vec"] from UnityDenseEnv (length >= 5, ideally 8).
        goal : (cos, sin, distXZ) triple from info["goal"], or None.

    Returns:
        np.ndarray shape (3,) float32: chosen action.
    """
    # Flatten and pad vec defensively
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    speed = float(v[0]) if v.size > 0 else 0.0
    yaw_rate = float(v[1]) if v.size > 1 else 0.0
    lat_err = float(v[5]) if v.size > 5 else 0.0

    cos, sin, dist = _safe_goal(goal)
    heading_err = float(np.arctan2(sin, cos))  # goal heading error (positive = goal to left in car frame)

    # Unity convention from AdvancedDoubleTrackController:
    #   steerCmd: negative = left, positive = right
    # So if goal is left (heading_err > 0), we want negative steer.
    target_steer_sign = -np.sign(heading_err)  # -1 (left), +1 (right), or 0 when heading_err ~ 0

    scores = []

    for (steer, thr, brk) in ACTION_TABLE:
        steer = float(steer)
        thr = float(thr)
        brk = float(brk)

        # 1) Alignment with goal heading
        #    - match steer sign with target_steer_sign when |heading_err| large
        #    - for small heading_err, prefer straight or small steering.
        steer_mag = abs(steer)
        desired_mag = min(1.0, abs(heading_err) / (np.pi / 4.0))  # saturate near 45deg
        steer_sign = np.sign(steer)

        if abs(heading_err) < 0.05:  # ~3 deg -> essentially aligned
            # prefer straight (small steering magnitude)
            align_score = 1.0 - steer_mag
        else:
            # prefer steer direction that matches target_steer_sign
            sign_match = 1.0 if (steer_sign == target_steer_sign and steer_sign != 0.0) else -0.5
            align_mag = 1.0 - abs(steer_mag - desired_mag)
            align_score = 0.7 * align_mag + 0.3 * sign_match

        # 2) Forward motion utility
        #    If goal is broadly ahead (cos>0), reward throttle.
        #    If goal is behind (cos<0), discourage throttle.
        if cos > 0.0:
            forward_score = thr * (0.8 * cos) - 0.2 * brk
        else:
            forward_score = -0.3 * thr - 0.1 * brk

        # 3) Stability penalty:
        #    - penalize large steering at high speed
        #    - penalize combining big yaw_rate, throttle, and steering.
        stability_pen = 0.0
        if abs(speed) > 1.5:
            stability_pen += (steer_mag ** 2) * (abs(speed) - 1.5)
        stability_pen += 0.15 * abs(yaw_rate) * (steer_mag + thr)

        # 4) "Keep moving" bonus when far from goal
        move_bonus = 0.0
        if dist > 1.5:
            move_bonus = 0.2 * thr

        # 5) Lateral error correction from route tracking:
        #    If lat_err > 0 (car left-of-path), we want steer right (positive).
        lane_score = 0.0
        if v.size > 5 and abs(lat_err) > 0.05:
            desired_lane_sign = -np.sign(lat_err)  # left+ -> steer right
            if steer_sign == desired_lane_sign and steer_sign != 0.0:
                lane_score += 0.3
            else:
                lane_score -= 0.1 * steer_mag

        total_score = (
            1.5 * align_score
            + forward_score
            + move_bonus
            + lane_score
            - 0.5 * stability_pen
        )
        scores.append(total_score)

    best_idx = int(np.argmax(scores))
    return ACTION_TABLE[best_idx].copy()