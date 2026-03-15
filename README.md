# CHAI — Companion Humanoid AI

A real-time companion system for the Unitree G1 humanoid robot: detects approaching humans, greets them, listens to Hindi speech, translates it to English, and autonomously clears obstacles from the path using the robot's arms.

---

## What it does

1. **VLM perception** — a background thread continuously captures camera frames and queries `Qwen2-VL-72B-Instruct` (via Nebius AI Studio) to detect humans and obstacles in the scene.
2. **State machine** — drives the robot through Idle → Greet → Listen → Translate → Guide → Obstacle-clear → Resume.
3. **Hindi translation** — spoken Hindi is transcribed via Google Speech Recognition, then translated to English with `Llama-3.3-70B-Instruct`.
4. **Obstacle clearing** — when an obstacle is detected the robot stops, announces a warning, and sweeps it aside with the appropriate arm.
5. **Sim / real toggle** — the same codebase runs in MuJoCo simulation or on live hardware by flipping one variable.

---

## Quick start — sim demo (no API key, no mic)

```bash
git clone --recurse-submodules https://github.com/your-org/nebius-hack.git
cd nebius-hack
pip install mujoco numpy
cd chai && python sim_demo.py
```

Opens a MuJoCo viewer: robot walks forward, detects the chair obstacle, sweeps it aside, and resumes.
See **[docs/RUNNING.md](docs/RUNNING.md)** for full instructions (Mac, Nebius VM, real hardware).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| MuJoCo | `pip install mujoco` |
| unitree_mujoco | G1 model XMLs — included as a git submodule |
| Nebius API key | for `main.py` (VLM + LLM) — not needed for `sim_demo.py` |
| Microphone | for Hindi speech input — not needed for `sim_demo.py` |
| `unitree-sdk2py` | real hardware only |

---

## Installation

```bash
git clone --recurse-submodules https://github.com/your-org/nebius-hack.git
cd nebius-hack
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

```bash
pip install uv          # once, if you don't have it
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt
```

**For real hardware** — uncomment and install the Unitree SDK:
```bash
pip install unitree-sdk2py
```

---

## Configuration

### API key

```bash
cp .env.example .env.local
# then edit .env.local and fill in your key
```

`.env.local` is git-ignored. Load it before running:

```bash
export $(cat .env.local | xargs)
```

### Sim vs real mode

Edit `chai/robot/config.py`:

```python
MODE = "sim"   # change to "real" for live hardware
```

Other settings in the same file:

```python
SIM_CONFIG = {
    "model_path": "unitree_mujoco/unitree_robots/g1/scene.xml",  # path to MuJoCo XML
    "camera": "front_camera"
}

REAL_CONFIG = {
    "interface": "enp3s0",        # Ethernet interface connected to the robot
    "robot_ip": "192.168.123.161"
}
```

---

## Running

### Sim demo (no API key)

```bash
cd chai && python sim_demo.py
```

### Full system (requires API key + mic)

```bash
cd chai && python main.py
```

Prints `[CHAI] System ready.` once the perception loop has started.

---

## Project structure

```
nebius-hack/
├── chai/
│   ├── sim_demo.py          # standalone demo — no API key, no mic needed
│   ├── main.py              # full system entry point
│   ├── state_machine.py     # CHAI state machine (Idle → Guide → Clear …)
│   ├── perception/
│   │   ├── vlm_loop.py      # background thread: captures frames, calls VLM
│   │   ├── audio.py         # microphone input + Google Speech Recognition
│   │   └── prompts.py       # system prompts for VLM queries
│   ├── voice/
│   │   ├── tts.py           # text-to-speech output (gTTS + playsound)
│   │   ├── translator.py    # Hindi → English via Llama LLM
│   │   └── phrases.py       # canned greeting / warning strings
│   ├── robot/
│   │   ├── config.py        # MODE toggle, API keys, sim/real config
│   │   ├── controller.py    # RobotController (sim stub or real SDK)
│   │   ├── arm.py           # arm sweep primitives (clear_left / clear_right)
│   │   └── locomotion.py    # walk_forward / stop; auto-loads trained checkpoint
│   ├── sim/
│   │   ├── scene.py         # patches MuJoCo XML with demo obstacles
│   │   └── perception_stub.py  # rule-based obstacle detection (no VLM)
│   ├── stretch/
│   │   └── train_policy.py  # RL policy training (feeds mjlab checkpoint)
│   └── checkpoints/
│       └── g1_locomotion.pt # trained locomotion policy (place here after training)
├── unitree_mujoco/          # git submodule — G1 model XMLs
└── docs/
    ├── RUNNING.md           # step-by-step run guide
    └── sim_resources.md     # Nebius VM + mjlab/video2robot setup
```

---

## Sim vs real

### Simulation

- `MODE = "sim"` (default)
- `sim_demo.py` — runs the full obstacle-clearing demo without any API key
- `chai/sim/scene.py` patches the scene XML to add a chair obstacle (~1.5 m ahead) and a person marker (~3 m ahead)
- `chai/sim/perception_stub.py` provides rule-based obstacle detection (no VLM needed)

### Policy training on Nebius VM

Train a locomotion policy on H100 GPUs using the hackathon-provided `mjlab` repo, then drop the checkpoint at `chai/checkpoints/g1_locomotion.pt`. `locomotion.py` auto-loads it at startup; if absent it falls back to the PD controller used in `sim_demo.py`.

See **[docs/sim_resources.md](docs/sim_resources.md)** for VM setup and **[docs/RUNNING.md](docs/RUNNING.md)** for the full training workflow.

### Real hardware checklist

1. Set `MODE = "real"` in `chai/robot/config.py`
2. Connect the robot over Ethernet; verify the interface name and IP match `REAL_CONFIG`
3. Install `unitree-sdk2py`
4. Power on the G1 and confirm it is in damping / ready state before running
5. Ensure the robot has clear space in front (≥ 2 m) before the first run

Full detail: **[docs/RUNNING.md — Real hardware](docs/RUNNING.md#c-real-hardware)**
