# Integrate video2robot + mjlab Pipeline into CHAI

## Context
The hackathon provides a two-repo pipeline for training robot motions from video:
1. **video2robot** — extracts human poses from video, retargets to G1 robot joints, outputs LAFAN CSV
2. **mjlab** — trains motion imitation RL policies from CSVs on Nebius H100, exports ONNX for deployment

This replaces the current stretch goal (`chai/stretch/train_policy.py` using the `playground` library) with the officially supported hackathon toolchain. The core CHAI system (state machine, VLM perception, sim) stays unchanged.

## Changes

### 1. Update `README.md`

**Replace** the "Stretch goal — RL grasp policy" section (lines 154–163) with a new **"Motion Training Pipeline (video2robot + mjlab)"** section covering:

- Overview: video → human pose → retarget to G1 → CSV → RL training → ONNX policy
- Prerequisites:
  - Register at https://smpl-x.is.tue.mpg.de and https://smpl.is.tue.mpg.de
  - Register at https://wandb.ai, create a "Motions" registry
- video2robot steps:
  1. On Nebius VM: `bash ./scripts/fetch_body_models.sh`
  2. Place video as `original.mp4` in data subfolder
  3. `python scripts/process_video.py --project data/{folder-name}`
- mjlab steps:
  1. `python src/mjlab/scripts/csv_to_npz.py --input_file {motion}.csv --input_fps 30 --output_name {name}`
  2. `uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name {org}/wandb-registry-Motions/{name} --env.scene.num-envs 8192 --agent.max-iterations 2000`
  3. `uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path {path} --num-envs 1`
  4. Download ONNX from W&B
- Integration: trained ONNX replaces scripted arm sweeps or locomotion checkpoint

**Add** a "Nebius VM Setup" section (after Installation) with:
- VM config: H100 NVLink, Ubuntu 24.04 LTS GPUs (CUDA 12), static IP
- SSH key setup and cloud-init
- VS Code Remote-SSH connection
- Link to repos on the VM at `/workspace/`
- Reminder to stop instance when done

**Update** project structure tree to show `stretch/` purpose change.

### 2. Update `docs/CHAI_implementation_plan.md`

- Replace Section 10 (Stretch Goal, lines 664–696) with video2robot + mjlab pipeline docs
- Update pitch notes (Section 12) to reference the new pipeline instead of `playground`

### 3. Update `chai/stretch/train_policy.py`

- Replace `playground`-based stub with a comment/docstring pointing to the mjlab workflow
- Or keep as a thin wrapper that documents how to load the ONNX output from mjlab

### Files to modify
- `/Users/cjache/repos/nebius-hack/README.md`
- `/Users/cjache/repos/nebius-hack/docs/CHAI_implementation_plan.md`
- `/Users/cjache/repos/nebius-hack/chai/stretch/train_policy.py`

## Verification
- Review updated README renders correctly
- Ensure all external URLs are correct and match the hackathon instructions
- CHAI sim still runs unchanged (`cd chai && python3 main.py` with `MODE="sim"`)
