"""
Standalone sim demo for CHAI — no API key, no microphone required.

Usage (from repo root):
    cd chai && python sim_demo.py

What you'll see:
  1. MuJoCo viewer opens with G1 standing, chair ~1.5m ahead
  2. Robot slides forward (PD controller + torso force)
  3. At ~0.8m from chair: stops, prints warning, sweeps right arm
  4. Announces "path clear", resumes forward motion
"""

import sys
import os
import time

# Allow imports from the chai package without installing
sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import mujoco.viewer

from sim.scene import patch_scene_xml
from robot.controller import RobotController
from sim.perception_stub import SimPerception
from voice.phrases import OBSTACLE_WARNINGS, RESUME_EN

SCENE_XML = os.path.join(
    os.path.dirname(__file__),
    "../unitree_mujoco/unitree_robots/g1/scene.xml"
)

# Demo state machine states (simplified — no mic/API)
from enum import Enum, auto

class State(Enum):
    GUIDING           = auto()
    OBSTACLE_DETECTED = auto()
    WARNING           = auto()
    CLEARING          = auto()
    RESUMING          = auto()
    DONE              = auto()


def _speak(text):
    """Print announcement; optionally use pyttsx3/espeak if available."""
    print(f"[CHAI] {text}")
    try:
        import subprocess
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def run_demo():
    # --- 1. Patch scene XML ---
    patched_xml = patch_scene_xml(SCENE_XML)
    print(f"[DEMO] Loaded patched scene: {patched_xml}")

    # --- 2. Load MuJoCo model ---
    model = mujoco.MjModel.from_xml_path(patched_xml)
    data  = mujoco.MjData(model)

    # --- 3. Create robot controller (bypass normal __init__) ---
    robot = RobotController.from_model_data(model, data)

    # --- 4. Perception stub ---
    perception = SimPerception(model, data)

    # --- 5. Launch passive viewer (must be main thread on macOS) ---
    state   = State.GUIDING
    warned  = False
    arm_end = 0.0

    print("[DEMO] Opening MuJoCo viewer... close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 4.0
        viewer.cam.azimuth  = 180
        viewer.cam.elevation = -20

        sim_start = time.time()

        while viewer.is_running():
            t = time.time()

            percept  = perception.get()
            obstacle = percept["obstacle"]

            # --- State machine tick ---
            if state == State.GUIDING:
                robot.walk_forward(speed=0.3)
                if obstacle.get("obstacle") and not warned:
                    warned = True
                    state  = State.OBSTACLE_DETECTED

            elif state == State.OBSTACLE_DETECTED:
                robot.stop()
                state = State.WARNING

            elif state == State.WARNING:
                side    = obstacle.get("side", "center")
                warning = OBSTACLE_WARNINGS["en"].get(side, OBSTACLE_WARNINGS["en"]["center"])
                _speak(warning)
                state = State.CLEARING

            elif state == State.CLEARING:
                _speak("Sweeping obstacle...")
                robot.arm.clear_right(duration=2.0)   # blocks ~2s via time.sleep
                state = State.RESUMING

            elif state == State.RESUMING:
                _speak(RESUME_EN)
                warned = False  # allow re-detection if needed
                state  = State.GUIDING

            # --- Step physics ---
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing at ~50 Hz
            elapsed = time.time() - t
            sleep_t = model.opt.timestep - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    run_demo()
