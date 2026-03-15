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

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| MuJoCo | `pip install mujoco` |
| Microphone | for Hindi speech input |
| Nebius API key | from [studio.nebius.ai](https://studio.nebius.ai) |
| unitree_mujoco | model XMLs for sim (see below) |
| `unitree-sdk2py` | real hardware only (see below) |

---

## Installation

```bash
git clone https://github.com/your-org/nebius-hack.git
cd nebius-hack
```

**Option A — uv (recommended, fast):**
```bash
pip install uv          # once, if you don't have it
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Option B — standard venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

**For real hardware** — uncomment and install the Unitree SDK:
```bash
pip install unitree-sdk2py
```

**MuJoCo model files** — the simulator expects the Unitree G1 scene XML at the path configured in `chai/robot/config.py` (default: `unitree_mujoco/unitree_robots/g1/scene.xml`). Clone the model repo alongside this one:
```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git
```

---

## Configuration

### API key

```bash
export NEBIUS_API_KEY="your-key-here"
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

```bash
cd chai
python3 main.py
```

The system prints `[CHAI] System ready.` once the perception loop has started, then enters the main state-machine loop.

---

## Sim vs real

### Simulation

- `MODE = "sim"` (default)
- Requires `unitree_mujoco` model files and `mujoco-python-viewer`
- `chai/sim/scene.py` patches the scene XML to add a chair obstacle (~1.5 m ahead) and a person marker (~3 m ahead) for demo testing

### Real hardware checklist

1. Set `MODE = "real"` in `chai/robot/config.py`
2. Connect the robot over Ethernet; verify the interface name and IP match `REAL_CONFIG`
3. Install `unitree-sdk2py`
4. Power on the G1 and confirm it is in damping / ready state before running
5. Ensure the robot has clear space in front (≥ 2 m) before the first run

---

## Project structure

```
nebius-hack/
├── chai/
│   ├── main.py              # entry point — wires all components together
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
│   │   └── locomotion.py    # walk_forward / stop
│   ├── sim/
│   │   └── scene.py         # patches MuJoCo XML with demo obstacles
│   └── stretch/
│       └── train_policy.py  # (stretch goal) RL grasp policy training
└── docs/
    └── CHAI_implementation_plan.md
```

---

## Stretch goal — RL grasp policy

`chai/stretch/train_policy.py` trains a PPO grasp-and-place policy on Nebius H100 instances using the `playground` library:

```bash
pip install mujoco playground
python chai/stretch/train_policy.py
```

If the policy transfers cleanly to hardware it can replace the scripted arm sweeps in `state_machine.py`. If not, the scripted sweeps handle the live demo while a simulation video demonstrates the trained policy.
