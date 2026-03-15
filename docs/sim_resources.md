# Nebius VM — Sim Resources & Training Pipeline

This document covers the Nebius VM setup and how the hackathon-provided `mjlab` / `video2robot` repos feed into CHAI's locomotion policy.

---

## Repos

- **mjlab**: https://github.com/DavidDobas/mjlab-hackathon — MuJoCo-based motion training; produces the locomotion checkpoint loaded by `chai/robot/locomotion.py`
- **video2robot**: https://github.com/DavidDobas/video2robot-hackathon — retargets human motions from video to the G1 skeleton; feeds reference motions into mjlab

---

## How these fit into CHAI

```
video2robot  →  reference motions  →  mjlab training  →  g1_locomotion.pt
                                                              ↓
                                              chai/robot/locomotion.py (auto-loads)
                                                              ↓
                                              chai/stretch/train_policy.py (RL grasp policy)
```

1. **video2robot** takes a video of a human walking / moving and retargets the motion to the G1 skeleton.
2. **mjlab** uses those reference motions (or its own) to train a locomotion policy on H100 GPUs.
3. The trained checkpoint is exported to `chai/checkpoints/g1_locomotion.pt`.
4. `locomotion.py` auto-loads it at startup — no code changes needed.
5. `chai/stretch/train_policy.py` can be extended to use mjlab motions for the grasp/sweep policy.

---

## SSH setup

Check for an existing key:
```bash
ls ~/.ssh/id_ed25519.pub
```

Create one if needed:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# press Enter to accept defaults
```

View the public key to paste into VM cloud-init:
```bash
cat ~/.ssh/id_ed25519.pub
```

---

## Nebius VM setup

1. Create a Nebius account at https://nebius.ai and set up billing (apply promo code).
2. Go to **Compute → Virtual machines → Create virtual machine**.
3. Select:
   - **Project**: `default-project-eu-north1`
   - **Platform**: NVIDIA H100 NVLink
   - **Boot disk**: Ubuntu 24.04 LTS for NVIDIA GPUs (CUDA 12)
   - **Public IP address**: Auto assign static IP
4. Under **User data**, enable **custom cloud-init config** and paste the cloud-init template. Replace `PASTE_YOUR_SSH_PUBLIC_KEY_HERE` with your actual key (multiple keys = one per line):
   ```yaml
   - ssh-ed25519 PASTE_YOUR_SSH_PUBLIC_KEY_HERE
   ```
5. Click **Create VM** and wait ~10 minutes.

---

## Connecting

Find the VM IP: **VM overview → Network → Public IPv4** (omit the `/32`).

Add to `~/.ssh/config`:
```
Host nebius_hackathon
  HostName YOUR_VM_IP_ADDRESS
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
```

Connect:
```bash
ssh nebius_hackathon
```

Or from VS Code / Cursor: **Command Palette → Remote-SSH: Connect to host → nebius_hackathon**, then open the `workspace` folder.

---

## Before using mjlab

1. Register at https://wandb.ai/
2. Create a registry called **Motions**
3. Generate a W&B API key in User Settings and log in:
   ```bash
   wandb login
   ```
4. Follow the mjlab README: https://github.com/DavidDobas/mjlab-hackathon

---

## Before using video2robot

Register at:
- https://smpl-x.is.tue.mpg.de/register.php
- https://smpl.is.tue.mpg.de/register.php

Then follow the video2robot README: https://github.com/DavidDobas/video2robot-hackathon

---

## Exporting the checkpoint to CHAI

After mjlab training completes:

```bash
# on the VM
cp /path/to/mjlab/checkpoints/best.pt \
   ~/workspace/nebius-hack/chai/checkpoints/g1_locomotion.pt

# or copy back to your local machine
scp nebius_hackathon:workspace/nebius-hack/chai/checkpoints/g1_locomotion.pt \
    chai/checkpoints/g1_locomotion.pt
```

Verify it loads:
```bash
cd chai && python -c "from robot.locomotion import LocomotionController; print('ok')"
# should print: [LOCO] Loaded trained policy checkpoint.
```

---

## Headless rendering on the VM

No display is attached. Set the EGL device before running any MuJoCo code:

```bash
export MUJOCO_EGL_DEVICE_ID=0
```

For offscreen frame capture:
```python
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)
frame = renderer.render()   # numpy (H, W, 3) — save with imageio or cv2
```

---

**Stop your VM instance when you're done — H100 time is expensive.**

Join the UFB Discord for support: https://discord.gg/j8AaY4Wnu (react to "🚀 Hackathon Access" in #onboarding).
