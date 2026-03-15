"""
Standalone sim demo for CHAI — no API key, no microphone required.

Usage (from repo root):
    cd chai && mjpython sim_demo.py        # macOS (required)
    cd chai && python   sim_demo.py        # Linux

What you'll see:
  1. MuJoCo viewer opens with G1 standing, person_marker ~3m ahead
  2. Robot turns toward and follows the person (who walks in a slow arc)
  3. At ~0.8m from chair: stops, prints warning, sweeps right arm
  4. Announces "path clear", resumes following the person
"""

import sys
import os
import time

# On macOS, mujoco.viewer.launch_passive requires mjpython.
# Auto-relaunch under mjpython if we're running under regular python.
if sys.platform == "darwin" and "mjpython" not in sys.executable:
    import shutil, subprocess
    mjpython = shutil.which("mjpython")
    if mjpython:
        print(f"[DEMO] Re-launching under mjpython: {mjpython}")
        os.execv(mjpython, [mjpython] + sys.argv)
    else:
        print("ERROR: On macOS, run this script with mjpython (not python).")
        print("  mjpython comes with the mujoco pip package:")
        print("  pip install mujoco   # then use 'mjpython' instead of 'python'")
        sys.exit(1)

# Allow imports from the chai package without installing
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
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

from enum import Enum, auto

class State(Enum):
    FOLLOWING         = auto()
    OBSTACLE_DETECTED = auto()
    WARNING           = auto()
    CLEARING          = auto()
    RESUMING          = auto()
    DONE              = auto()


def _speak(text):
    """Print announcement; optionally use say on macOS if available."""
    print(f"[CHAI] {text}")
    try:
        import subprocess
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _follow_person(robot, data):
    """Compute and send velocity to follow person_marker."""
    pelvis_pos  = data.body("pelvis").xpos[:2]
    person_pos  = data.body("person_marker").xpos[:2]
    dx, dy      = person_pos - pelvis_pos
    dist        = float(np.linalg.norm([dx, dy]))
    # Robot heading from pelvis quaternion (w,x,y,z)
    quat        = data.body("pelvis").xquat
    yaw         = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                              1 - 2*(quat[2]**2 + quat[3]**2))
    bearing     = np.arctan2(dy, dx)
    angle_err   = float(np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw)))
    vx          = min(0.4, dist * 0.15) if dist > 0.5 else 0.0
    omega       = float(np.clip(angle_err * 1.5, -0.8, 0.8))
    robot.loco.send_velocity(vx=vx, vy=0, omega=omega)


def _move_person(model, data, step):
    """Advance person_marker along a slow arc each simulation step."""
    # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "person_marker")
    qpos_addr = model.jnt_qposadr[joint_id]

    # Move ~0.2 m/s forward + slight left turn; update every step
    dt = model.opt.timestep
    speed = 0.2          # m/s forward
    turn  = 0.15         # rad/s left

    # Current angle-of-travel stored in freejoint z-rotation is complex;
    # simpler: track globally with a Python variable (passed in as mutable list).
    # We use data.qpos directly.
    x  = data.qpos[qpos_addr + 0]
    y  = data.qpos[qpos_addr + 1]

    # Accumulate heading via the step counter (constant turn rate)
    heading = step * dt * turn  # radians since t=0
    data.qpos[qpos_addr + 0] = x + speed * dt * np.cos(heading)
    data.qpos[qpos_addr + 1] = y + speed * dt * np.sin(heading)
    # Leave z and quaternion unchanged (upright cylinder)


def run_demo():
    # --- 1. Patch scene XML ---
    patched_xml = patch_scene_xml(SCENE_XML)
    print(f"[DEMO] Loaded patched scene: {patched_xml}")

    # --- 2. Load MuJoCo model ---
    model = mujoco.MjModel.from_xml_path(patched_xml)
    data  = mujoco.MjData(model)

    # --- 3. Create robot controller ---
    robot = RobotController.from_model_data(model, data)

    # --- 4. Perception stub ---
    perception = SimPerception(model, data)

    # --- 5. Launch viewer ---
    state    = State.FOLLOWING
    warned   = False
    arm_end  = 0.0
    sim_step = 0

    print("[DEMO] Opening MuJoCo viewer... close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance  = 4.0
        viewer.cam.azimuth   = 180
        viewer.cam.elevation = -20

        while viewer.is_running():
            t = time.time()

            percept  = perception.get()
            obstacle = percept["obstacle"]

            # --- State machine tick ---
            if state == State.FOLLOWING:
                _follow_person(robot, data)
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
                robot.arm.clear_right(duration=2.0)
                state = State.RESUMING

            elif state == State.RESUMING:
                _speak(RESUME_EN)
                warned = False
                state  = State.FOLLOWING

            # --- Move person along arc ---
            _move_person(model, data, sim_step)
            sim_step += 1

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
