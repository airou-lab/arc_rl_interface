# Simplified Steering-Only Training

## Overview

This is a **dramatically simplified** training approach that addresses the core learning problem:

**The agent was trying to learn too much at once** (steering, throttle, brake) from sparse rewards.

### Key Simplifications

1. **1D Action Space**: Agent only outputs steering [-1, 1]
2. **Automatic Throttle**: Speed adjusts based on steering magnitude (slow down for turns)
3. **Dense Rewards**: Classical control law provides steering target
4. **Still Passive Visual**: Policy cannot see geometric information

## Files

- `train_steering_only.py` - Main training script
- `evaluate_steering.py` - Evaluate trained models  
- `sanity_check.py` - Quick environment test
- `live_unity_env.py` - Unity TCP environment (from your codebase)

## Usage

### 1. Sanity Check (verify connection)

```bash
python sanity_check.py --host 127.0.0.1 --port 5556
```

### 2. Training

```bash
# Basic training (200k steps)
python train_steering_only.py

# With custom settings
python train_steering_only.py \
    --timesteps 500000 \
    --lr 3e-4 \
    --n_steps 256 \
    --batch_size 64
```

### 3. Evaluation

```bash
python evaluate_steering.py models/steer_only_XXXXXX/final_model.zip \
    --n_episodes 10 --deterministic
```

## How It Works

### Reward Structure (Dense Shaping)

The agent receives dense rewards that tell it exactly what to do:

```
+ Progress reward: ~2.5 per meter traveled
+ Lane keeping: +0.2 when centered, penalty when off
+ Steering target: +0.4 for correct steering, +0.25 bonus for excellent
+ Smoothness: +0.15 for smooth steering
- Off track: -0.3 per meter deviation
- Crash: -5.0
```

### Passive Visual Compliance

The policy **only sees**:
- Camera image (128×128×3)
- turn_bias: {-1, 0, +1} navigation command
- speed, yaw_rate (proprioception)
- last_steer, last_thr, last_brk (action history)

The policy **does NOT see**:
- goal_dist (masked to 0)
- lat_err, hdg_err (masked to 0)
- kappa, ds (masked to 0)

**But rewards use lat_err/hdg_err!** This is the key:
- Dense reward signal guides learning
- Policy can't "cheat" because it never sees the values
- CNN must learn road geometry from images to predict correct steering

### Steering Target Computation

```python
target_steer = (
    0.35 * turn_bias +      # Follow navigation command
    0.40 * lat_err +         # Correct lateral deviation
    0.60 * hdg_err           # Align with road direction
)
```

This is a classical control law (PD-like). The RL agent learns to approximate this behavior from visual input only.

### Curvature-Adaptive Speed

```python
throttle = 0.42 - 0.25 * abs(steering)
```

When steering hard → slow down automatically.
When going straight → speed up.

No learning required for throttle!

## Why This Should Work

1. **Single DoF**: Only steering to learn = much easier
2. **Dense Reward**: Every step tells agent "you're doing well/poorly"
3. **Smooth Reward Landscape**: Exponential decay on steering error
4. **Classical Bootstrap**: Target steering from control theory
5. **No Exploration Needed for Speed**: Throttle is automatic

## Expected Behavior

- Early training: Random steering, lots of crashes
- ~50k steps: Starting to follow road somewhat
- ~100k steps: Reasonable lane keeping on straight roads  
- ~200k steps: Should handle mild curves
- ~500k steps: Should handle most scenarios

## Troubleshooting

### "Connection refused"
- Make sure Unity is running with RLClientSender enabled
- Check port 5556 is not blocked

### "Training stuck at 0 reward"
- Run sanity_check.py to verify connection
- Check Unity console for errors
- Ensure car spawns on road

### "Car goes in circles"
- This is expected early in training
- The dense rewards will guide it

## Next Steps (After This Works)

1. **Curvature Auxiliary Task**: Train CNN to predict road curvature
2. **Variable Speed**: Learn throttle once steering is stable
3. **Real Robot Deployment**: Use the bridge with trained model