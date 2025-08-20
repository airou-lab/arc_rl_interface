# Clean Spatiotemporal RL (Passive, Vision-Only)

This set removes **all heuristics** (masks, optical flow, path planning, action blending) and uses a **recurrent policy** to learn maneuvers from **RGB-only** sequences.

## Files

- `unity_camera_env.py` — Offline passive env (RGB only; CHW). For smoke tests only (frames aren't coupled to actions).
- `live_unity_env.py` — Live passive env over TCP. Unity must return `(jpeg, reward, done, truncated)` each step. Python sends `(steer, throttle)`.
- `train_policy_RNN.py` — RecurrentPPO + CnnLstmPolicy training. Use `--live` for real spatiotemporal training.
- `evaluate_policy_RNN.py` — Evaluate a trained recurrent model (offline or live).
- `inference_server_RNN.py` — RNN inference server that keeps LSTM state and returns actions for incoming JPEG frames.
- `plot_reward_log.py` — Plot SB3 `monitor.csv` episode rewards.
- `test_env.py` — Quick smoke test for the offline env.

## Unity ↔ Python protocol (live training)

- **Reset**: Python sends `b'R'`. Unity replies: `len|jpeg|reward|done|truncated` where:
  - `len`: 4-byte big-endian unsigned int (`!I`)
  - `jpeg`: encoded RGB frame bytes
  - tail: `!f??` → float32 reward, bool done, bool truncated
- **Step**: Python sends `!ff` → (steer, throttle). Unity replies again with the same `(len|jpeg|reward|done|truncated)`.
- **Preprocessing**: Both sides use the same crop constant `CROP_TOP_FRAC = 0.25`. Images are resized to `(W, H)`.

## Train (recommended: live spatiotemporal)

```bash
python train_policy_RNN.py --live --host 127.0.0.1 --port 5556 \\
  --img_size 84 84 --max_steps 500 --timesteps 200000
```

This uses `sb3-contrib` RecurrentPPO + `CnnLstmPolicy` and wraps envs with `VecTransposeImage` so the CNN sees HWC.

## Evaluate (recurrent)

```bash
python evaluate_policy_RNN.py --model_path models/<run>/final_model.zip --live --host 127.0.0.1 --port 5556
```

## Inference (RNN)

```bash
python inference_server_RNN.py --model_path models/<run>/final_model.zip --port 5555
```

Unity drives the loop: send `len|jpeg` → receive `(steer, throttle)`.

---

### Notes
- Keep **observations** strictly RGB (no masks/flows). Keep **actions** verbatim (no blending). Compute **rewards** only on the Unity side using simulator signals (alive, speed, smoothness, collision/goal).
- Offline env is not a true MDP. Use it only for debugging pipeline and quick tests; for learning maneuvers you need the **live** env.
