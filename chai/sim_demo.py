"""
Standalone sim demo for CHAI — uses live VLM/LLM API calls via Nebius.

Usage (from repo root):
    export NEBIUS_API_KEY=<your key>
    source venv/bin/activate
    cd chai && python sim_demo.py          # all platforms (headless OpenCV window)
    cd chai && mjpython sim_demo.py        # macOS interactive viewer (needs: brew install glfw)

What you'll see:
  1. MuJoCo viewer opens with G1 standing, skin-toned person_marker ~1.5m behind
  2. VLM threads start in background (2 fps obstacle detection)
  3. Robot walks forward leading the person (blind-guide scenario)
  4. VLM detects chair → robot steers toward it (APPROACHING)
  5. LLM plans arm action → arm sweeps (CLEARING)
  6. VLM confirms path clear → robot resumes walking (RESUMING)
"""

import sys
import os
import time
import json

# Allow imports from the chai package without installing
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
import mujoco.viewer
from openai import OpenAI

from sim.scene import patch_scene_xml
from robot.controller import RobotController
from perception.vlm_loop import PerceptionLoop
from perception.prompts import PATH_CLEAR_PROMPT, ACTION_PLANNING_PROMPT
from robot.config import (
    TOKEN_FACTORY_BASE_URL, TOKEN_FACTORY_API_KEY, LLM_MODEL
)
from voice.phrases import OBSTACLE_WARNINGS, RESUME_EN

SCENE_XML = os.path.join(
    os.path.dirname(__file__),
    "../unitree_mujoco/unitree_robots/g1/chai_demo_scene.xml"
)

from enum import Enum, auto


class State(Enum):
    WAITING          = auto()   # idle until VLM has first result
    FOLLOWING        = auto()
    APPROACHING      = auto()   # VLM sees obstacle; steer toward it
    ACTION_PLANNING  = auto()   # LLM decides which arm action to use
    CLEARING         = auto()   # Arm sweeping
    CONFIRMING_CLEAR = auto()   # One-shot VLM re-query after sweep
    RESUMING         = auto()


def _speak(text):
    """Print announcement; optionally use say on macOS if available."""
    print(f"[CHAI] {text}")
    try:
        import subprocess
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _walk_forward(robot, data):
    """Robot leads — walk forward at steady pace."""
    robot.loco.send_velocity(vx=0.35, vy=0, omega=0)


def _move_person(model, data, step):
    """Advance person_marker along a slow arc each simulation step."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "person_marker")
    qpos_addr = model.jnt_qposadr[joint_id]

    dt    = model.opt.timestep
    speed = 0.2
    turn  = 0.15

    x = data.qpos[qpos_addr + 0]
    y = data.qpos[qpos_addr + 1]

    heading = step * dt * turn
    data.qpos[qpos_addr + 0] = x + speed * dt * np.cos(heading)
    data.qpos[qpos_addr + 1] = y + speed * dt * np.sin(heading)
    data.qpos[qpos_addr + 2] = 0.0   # pin Z — keep feet on ground


def _plan_action(obstacle_json: dict, llm_client) -> str:
    """Call LLM to decide which arm action to take. Returns 'sweep_left' or 'sweep_right'."""
    prompt = ACTION_PLANNING_PROMPT + f"\nObstacle: {json.dumps(obstacle_json)}"
    print(f"[ACTION] Asking LLM to plan arm action for: {obstacle_json}")
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    print(f"[ACTION] LLM response: {raw}")
    parsed = json.loads(raw)
    action = parsed.get("action", "sweep_right")
    reason = parsed.get("reason", "")
    print(f"[ACTION] {action} — {reason}")
    return action


def _run_headless(model, data, tick_fn):
    """Headless render loop using mujoco.Renderer + OpenCV."""
    try:
        import cv2
    except ImportError:
        print("[DEMO] pip install opencv-python  for headless display; running blind.")
        cv2 = None

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance  = 7.0
    cam.azimuth   = 135.0
    cam.elevation = -25.0

    frame_every = 10
    step = 0
    while True:
        tick_fn()
        mujoco.mj_step(model, data)
        if cv2 and step % frame_every == 0:
            renderer.update_scene(data, cam)
            rgb = renderer.render()
            bgr = rgb[:, :, ::-1]
            cv2.imshow("CHAI sim", bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        step += 1
    if cv2:
        cv2.destroyAllWindows()


def run_demo():
    # --- 1. Patch scene XML ---
    patched_xml = patch_scene_xml(SCENE_XML)
    print(f"[DEMO] Loaded patched scene: {patched_xml}")

    # --- 2. Load MuJoCo model ---
    model = mujoco.MjModel.from_xml_path(patched_xml)
    data  = mujoco.MjData(model)

    # --- 3. Create robot controller ---
    robot = RobotController.from_model_data(model, data)

    # --- 4. Create Nebius OpenAI client ---
    client = OpenAI(
        base_url=TOKEN_FACTORY_BASE_URL,
        api_key=TOKEN_FACTORY_API_KEY,
    )

    # --- 5. Start perception loop (background VLM threads) ---
    perception = PerceptionLoop(robot, client)
    perception.start()
    print("[DEMO] Perception loop started (VLM threads running at ~2 fps)")

    # --- 6. State machine setup ---
    state        = State.WAITING
    sim_step     = 0
    _clear_start = None
    _planned_action = None

    def tick_fn():
        nonlocal state, sim_step, _clear_start, _planned_action

        percept  = perception.get()
        obstacle = percept["obstacle"]
        obs_detected = obstacle.get("obstacle", False)
        obs_side     = obstacle.get("side", "center")
        obs_distance = obstacle.get("distance", None)

        if state == State.WAITING:
            robot.stop()
            if perception.ready:
                print("[SM] VLM ready — entering FOLLOWING")
                if obs_detected:
                    _speak(OBSTACLE_WARNINGS["en"].get(obs_side, OBSTACLE_WARNINGS["en"]["center"]))
                    state = State.APPROACHING
                else:
                    state = State.FOLLOWING

        elif state == State.FOLLOWING:
            _walk_forward(robot, data)
            if obs_detected:
                print(f"[SM] Obstacle detected: {obstacle} — entering APPROACHING")
                _speak(OBSTACLE_WARNINGS["en"].get(obs_side, OBSTACLE_WARNINGS["en"]["center"]))
                state = State.APPROACHING

        elif state == State.APPROACHING:
            # Steer toward obstacle based on VLM side
            if obs_side == "left":
                robot.loco.send_velocity(vx=0.25, vy=0, omega=+0.3)
            elif obs_side == "right":
                robot.loco.send_velocity(vx=0.25, vy=0, omega=-0.3)
            else:
                robot.loco.send_velocity(vx=0.25, vy=0, omega=0)

            if obs_distance and str(obs_distance).startswith("near"):
                print("[SM] Obstacle near — entering ACTION_PLANNING")
                robot.stop()
                state = State.ACTION_PLANNING

        elif state == State.ACTION_PLANNING:
            robot.stop()
            try:
                _planned_action = _plan_action(obstacle, client)
            except Exception as e:
                print(f"[ACTION] LLM error: {e} — defaulting to sweep_right")
                _planned_action = "sweep_right"
            state = State.CLEARING

        elif state == State.CLEARING:
            robot.stop()
            if _clear_start is None:
                _clear_start = time.time()
                _speak("Sweeping obstacle...")
            elapsed = time.time() - _clear_start
            # Use planned action; default to clear_right if unknown
            if _planned_action == "sweep_left":
                robot.arm.clear_left_tick(elapsed)
            else:
                robot.arm.clear_right_tick(elapsed)
            if elapsed >= 2.0:
                _clear_start = None
                state = State.CONFIRMING_CLEAR

        elif state == State.CONFIRMING_CLEAR:
            robot.stop()
            print("[SM] Confirming path clear with VLM...")
            try:
                result = perception.query_once(PATH_CLEAR_PROMPT)
                clear  = result.get("clear", False)
                reason = result.get("reason", "")
                print(f"[SM] Path clear check: {clear} — {reason}")
            except Exception as e:
                print(f"[SM] VLM confirm error: {e} — assuming clear")
                clear = True
            if clear:
                state = State.RESUMING
            else:
                print("[SM] Path still blocked — re-planning action")
                state = State.ACTION_PLANNING

        elif state == State.RESUMING:
            robot.stop()
            _speak(RESUME_EN)
            # Move chair out of path so it doesn't re-trigger
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chair")
            addr = model.jnt_qposadr[joint_id]
            data.qpos[addr + 1] = 2.5
            state = State.FOLLOWING

        if state in (State.WAITING, State.FOLLOWING):
            _move_person(model, data, sim_step)
        sim_step += 1

    # --- 7. Launch viewer (interactive or headless fallback) ---
    print("[DEMO] Opening MuJoCo viewer... close the window to exit.")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance  = 7.0
            viewer.cam.azimuth   = 135
            viewer.cam.elevation = -25

            while viewer.is_running():
                t = time.time()
                tick_fn()
                mujoco.mj_step(model, data)
                viewer.sync()
                elapsed = time.time() - t
                sleep_t = model.opt.timestep - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    except RuntimeError:
        print("[DEMO] Interactive viewer unavailable — falling back to headless (OpenCV).")
        print("[DEMO] For interactive viewer on macOS: brew install glfw && mjpython sim_demo.py")
        _run_headless(model, data, tick_fn)
    finally:
        perception.stop()


if __name__ == "__main__":
    run_demo()
