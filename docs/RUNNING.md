# CHAI — Running Guide

Three environments: local Mac (sim demo), Nebius VM (headless training), real hardware.

---

## A. Local Mac — sim_demo (no API key, no mic)

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/your-org/nebius-hack.git
cd nebius-hack
```

Already cloned without submodules?
```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
pip install mujoco numpy
```

### 3. Run

```bash
cd chai
python sim_demo.py
```

MuJoCo viewer opens. The robot walks forward, detects the chair obstacle, sweeps it aside with the right arm, and resumes. Announcements are printed to stdout and spoken via the macOS `say` command.

### Notes

- Requires a display — the passive viewer runs on the main thread.
- `say` is macOS only; on Linux you can install `espeak` and adjust `_speak()` in `sim_demo.py`.
- Scene XML is loaded from `unitree_mujoco/unitree_robots/g1/scene.xml` (relative to repo root). If you see a file-not-found error, confirm `git submodule update --init --recursive` completed and the path in `chai/sim_demo.py:29` matches.

---

## B. Nebius VM — headless sim + policy training

### 1. Create the VM

Follow the setup in [docs/sim_resources.md](sim_resources.md):

- Platform: NVIDIA H100 NVLink
- Boot disk: Ubuntu 24.04 LTS for NVIDIA GPUs (CUDA 12)
- Paste the cloud-init config with your SSH public key

Wait ~10 minutes after clicking **Create VM**.

### 2. Connect

```bash
ssh ubuntu@YOUR_VM_IP
```

Or add to `~/.ssh/config`:
```
Host nebius_hackathon
  HostName YOUR_VM_IP
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
```

### 3. Clone the repo

```bash
cd workspace
git clone --recurse-submodules https://github.com/your-org/nebius-hack.git
cd nebius-hack
pip install mujoco numpy torch
```

### 4. Headless MuJoCo rendering

The H100 VM has no display. Set the EGL device so MuJoCo can render offscreen:

```bash
export MUJOCO_EGL_DEVICE_ID=0
```

To record frames instead of opening a viewer, use `mujoco.Renderer`:

```python
import mujoco
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)
frame = renderer.render()   # numpy array (H, W, 3)
```

For interactive viewing over SSH use X11 forwarding:
```bash
ssh -X ubuntu@YOUR_VM_IP
```

### 5. Train a locomotion policy with mjlab

The VM's `workspace/` directory contains the hackathon repos. Follow the mjlab README:

```bash
cd workspace/mjlab
# follow mjlab/README.md — requires W&B API key and a "Motions" registry
```

Before training, register at:
- https://wandb.ai/ — create a **Motions** registry and generate an API key

### 6. Export the checkpoint

After training completes, copy the locomotion checkpoint to the CHAI repo:

```bash
mkdir -p nebius-hack/chai/checkpoints
cp PATH/TO/TRAINED/checkpoint.pt nebius-hack/chai/checkpoints/g1_locomotion.pt
```

`locomotion.py` checks for `chai/checkpoints/g1_locomotion.pt` at startup:
- If found → loads the TorchScript policy, logs `[LOCO] Loaded trained policy checkpoint.`
- If absent → falls back to PD controller, logs `[LOCO] Using PD position controller (no checkpoint).`

### 7. (Optional) Train motion retargeting with video2robot

```bash
cd workspace/video2robot
# follow video2robot/README.md
```

Requires registration at:
- https://smpl-x.is.tue.mpg.de/register.php
- https://smpl.is.tue.mpg.de/register.php

**Stop your VM instance when done — H100 time is expensive.**

---

## C. Real hardware

### Prerequisites

- `pip install unitree-sdk2py`
- G1 robot powered on and in damping/ready state
- Ethernet cable connected; interface and IP match `REAL_CONFIG` in `chai/robot/config.py`

### 1. Copy the trained checkpoint (optional)

If you trained on Nebius VM:

```bash
scp ubuntu@YOUR_VM_IP:workspace/nebius-hack/chai/checkpoints/g1_locomotion.pt \
    chai/checkpoints/g1_locomotion.pt
```

If no checkpoint is present, the PD controller is used.

### 2. Configure

Edit `chai/robot/config.py`:

```python
MODE = "real"

REAL_CONFIG = {
    "interface": "enp3s0",        # Ethernet interface — verify with `ip link`
    "robot_ip": "192.168.123.161"
}
```

### 3. Set your API key

```bash
cp .env.example .env.local
# fill in NEBIUS_API_KEY
export $(cat .env.local | xargs)
```

### 4. Pre-flight checklist

- [ ] `MODE = "real"` in `config.py`
- [ ] Ethernet connected, interface name confirmed
- [ ] G1 in damping / ready state
- [ ] Clear space ≥ 2 m in front of the robot
- [ ] `unitree-sdk2py` installed

### 5. Run

```bash
cd chai
python main.py
```

Prints `[CHAI] System ready.` once the perception loop starts.
