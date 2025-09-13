# Clean Spatiotemporal RL (Passive, Vision-Only)

Train and evaluate a **recurrent policy** (RecurrentPPO with CNN+LSTM) that learns autonomous maneuvers from **RGB frames only** streaming from a Unity mini‑city.  
No semantic masks, optical flow, handcrafted path planners, or action blending — purely **passive visual navigation**.

---

## Repository layout (Python side)

- **`live_unity_env.py`** — Gymnasium environment that connects to Unity via TCP and exchanges frames/actions in real time. Returns observations as **HWC (`H×W×3`) `uint8` RGB**.
- **`unity_camera_env.py`** — **Back‑compat shim** that aliases `LiveUnityEnv`. Prefer `from live_unity_env import LiveUnityEnv` in new code.
- **`action_repeat_wrapper.py`** — Optional wrapper to repeat each action for *N* steps.
- **`train_policy_RNN.py`** — Train RecurrentPPO (`CnnLstmPolicy`) on the live Unity stream.
- **`inference_server_RNN.py`** — Drive the Unity sim using a saved recurrent model.
- **`evaluate_policy_RNN.py`** — Evaluate a saved model for *N* episodes and report returns.
- **`record_run_from_model.py`** — Run a model and optionally save **MP4 + CSV** of the episode.
- **`smoke_client.py`** — Minimal TCP client to sanity‑check the Unity socket loop.
- **`test_env.py`** — Quick manual step loop with the env (no SB3).
- **`plot_reward_log.py`** — Plot rewards from SB3 `monitor.csv` or our `run.csv`.

---

## Unity ↔ Python protocol (live)

**Reset**
- Python → Unity: single byte `b'R'`
- Unity → Python: `len (u32 BE)` → `jpeg (len bytes)` → `reward (f32 BE)` → `done (u8)` → `truncated (u8)`

**Step (each control step)**
- Python → Unity: `steer (f32 BE)` + `throttle (f32 BE)`
- Unity → Python: same `(len|jpeg|reward|done|truncated)` structure

**Spaces**
- **Observation**: `(H, W, 3)` **HWC** `uint8` RGB (e.g., `84×84×3`)
- **Action**: continuous `Box([-1, 0], [1, 1])` → `(steer, throttle)`  
  `steer ∈ [−1, +1]`, `throttle ∈ [0, 1]`
- **Preprocessing**: none. JPEG → RGB decode only (no crops, masks, or flow).

> Rewards are computed **in Unity** (alive, velocity, lane/road penalties, collisions, goals). Python remains passive.

---

## Install (one‑time setup)

```bash
cd arc_rl_interface/unity_env
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install "gymnasium>=0.29" pillow numpy pandas matplotlib
pip install stable-baselines3 sb3-contrib
pip install tqdm rich "imageio[ffmpeg]"   # progress bar + optional MP4
```

Quick sanity check:

```bash
python - <<'PY'
import gymnasium, stable_baselines3 as sb3, sb3_contrib, PIL, numpy
print("OK:", gymnasium.__version__, sb3.__version__, sb3_contrib.__version__)
PY
```

---

## Unity scene wiring (checklist)

- Exactly **one** `RLClientSender` active in the scene (`t:RLClientSender` search).
- On **RLClientSender**:
  - **Port**: `5556`
  - **Capture Camera**: assign your forward‑looking camera
  - **Output Width/Height**: `84 / 84`
  - **JPEG Quality**: ~`80`
  - **Max Steps**: e.g., `500–1000`
  - *(Optional)* Lockstep enabled to stabilize frame/step timing
- Optional debug visuals *(do not affect control)*:
  - `ActionTrajectoryPreview` (cyan preview, press **T**)
  - `TelemetryHUD` (stats, press **H**)
  - `ActuatorGauges` (steer/throttle, press **G**)
  - For always‑visible lines, add `Assets/Shaders/AlwaysOnTopLine.shader` (Built‑in or URP variant) and assign it to the LineRenderer.

Press **Play** in Unity. Console should show:

```
RLClientSender: listening on 0.0.0.0:5556
```

---

## Smoke test (no RL)

With Unity **playing**:

```bash
source venv/bin/activate
python smoke_client.py
```

**Expected:** “Connected to Unity.” then repeated step prints; the car moves; Unity logs “client connected”.

---

## Train (Recurrent PPO)

```bash
PYTHONUNBUFFERED=1 python -u train_policy_RNN.py   --host 127.0.0.1 --port 5556   --img_size 84 84 --max_steps 500   --timesteps 200000   --lr 5e-4 --n_steps 256 --batch_size 128 --n_epochs 5   --ent_coef 0.02 --repeat 1 --verbose 1
```

- Checkpoints → `models/rppo_YYYYMMDD_HHMMSS/`
- Logs → `./tb/` (if you use TensorBoard)

---

## Evaluate a saved model

```bash
MODEL=models/rppo_YYYYMMDD_HHMMSS/final_model.zip
python evaluate_policy_RNN.py --model "$MODEL"   --host 127.0.0.1 --port 5556 --img_size 84 84 --max_steps 500
```

---

## Inference (drive Unity with a saved model)

```bash
MODEL=models/rppo_YYYYMMDD_HHMMSS/final_model.zip
python inference_server_RNN.py --model "$MODEL"   --host 127.0.0.1 --port 5556 --img_size 84 84 --max_steps 500
```

**Record the run (optional):**
```bash
python record_run_from_model.py --model "$MODEL"   --host 127.0.0.1 --port 5556 --img_size 84 84   --out_mp4 proof_run.mp4 --out_csv proof_run.csv --fps 30

python plot_reward_log.py proof_run.csv
```

---

## Troubleshooting

- **No connection / script exits**  
  Unity must be **in Play** and listening on the same port. Check:
  ```bash
  lsof -n -P -iTCP:5556 | grep LISTEN
  ```

- **CHW vs HWC errors**  
  Our env returns **HWC** `(H, W, 3)`. If you see `(3, H, W)`, you’re using an old env. Ensure:
  ```python
  from live_unity_env import LiveUnityEnv
  ```
  and `unity_camera_env.py` is the shim that re‑exports `LiveUnityEnv`.

- **Progress bar import error**  
  `pip install tqdm rich` (or remove `progress_bar=True` in training).

- **Unity “NetLoop exception: interrupted”**  
  Harmless when a Python client disconnects mid‑episode.

- **Overlapping overlays**  
  Disable any legacy `OnGUI` overlays; use `TelemetryHUD` and gauges.

---

## Design principles (Don’t)
- No semantic segmentation, optical flow, handcrafted path planning, or action blending.

## Design principles (Do)
- The policy observes **only RGB** and outputs actions; **Unity computes rewards** and termination.
- Keep interfaces minimal, typed, and documented for easy handoff.
---

## Moving the Unity project to a lab desktop

1. **Project Settings → Editor**
   - Version Control: **Visible Meta Files**
   - Asset Serialization: **Force Text**
2. **Close Unity** to release file locks.
3. Copy the project folder, excluding generated dirs:
   - **Keep**: `Assets/`, `Packages/`, `ProjectSettings/` (+ your custom folders)
   - **Exclude**: `Library/`, `Temp/`, `Obj/`, `Logs/`, `Builds/`
   ```bash
   rsync -av --progress      --exclude 'Library/' --exclude 'Temp/' --exclude 'Obj/' --exclude 'Logs/' --exclude 'Builds/'      /path/to/UnityProject/ /Volumes/ExternalDrive/UnityProject/
   ```
4. On the lab machine, open via Unity Hub → **Open** → select the folder.
5. Re‑check `RLClientSender` wiring and port; press **Play**.

---

## Code style & conventions (Python)

- PEP 8 naming, type hints everywhere, and module‑level docstrings.
- Image tensors are **HWC `uint8`** consistently.
- Single environment implementation (`LiveUnityEnv`) with a documented socket protocol.
- Small, single‑purpose scripts (`train_*`, `evaluate_*`, `inference_*`, `record_*`, `smoke_*`).

---

## Deprecations

- `unity_camera_env.py` previously contained extra heuristics.  
  It now **aliases** `LiveUnityEnv` and emits a `DeprecationWarning`. Prefer:
  ```python
  from live_unity_env import LiveUnityEnv, UnityEnvConfig
  ```