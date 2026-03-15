"""
Standalone sim demo for CHAI — uses live VLM/LLM API calls via Nebius.

Usage (from repo root):
    export NEBIUS_API_KEY=<your key>
    source venv/bin/activate
    cd chai && python sim_demo.py          # all platforms (headless OpenCV window)
    cd chai && mjpython sim_demo.py        # macOS interactive viewer (needs: brew install glfw)

What you'll see:
  1. MuJoCo viewer opens with G1 standing, skin-toned person_marker ~1.5m behind
  2. VLM threads start in background (2 fps obstacle + human detection)
  3. Robot waits for first VLM result, then LLM agent drives all decisions
  4. LLM sees perception + sim geometry → decides speed, speech, sweep direction
  5. Physics loop executes decisions non-blocking; sweep runs to completion once started
"""

import sys
import os
import time

# Allow imports from the chai package without installing
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
import mujoco.viewer
from openai import OpenAI

from sim.scene import patch_scene_xml
from robot.controller import RobotController
from perception.vlm_loop import PerceptionLoop
from robot.config import (
    TOKEN_FACTORY_BASE_URL, TOKEN_FACTORY_API_KEY, LLM_MODEL, SIM_CONFIG
)
from agent import RobotAgent

SCENE_XML = os.path.join(
    os.path.dirname(__file__),
    "../unitree_mujoco/unitree_robots/g1/chai_demo_scene.xml"
)


def _speak(text):
    """Print announcement; optionally use say on macOS if available."""
    print(f"[CHAI] {text}")
    try:
        import subprocess
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _move_person(model, data, step, moving=True):
    """Advance person_marker along a slow arc each simulation step."""
    joint_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "person_marker")
    qpos_addr = model.jnt_qposadr[joint_id]
    dof_addr  = model.jnt_dofadr[joint_id]

    if moving:
        dt      = model.opt.timestep
        speed   = 0.2
        turn    = 0.15
        x       = data.qpos[qpos_addr + 0]
        y       = data.qpos[qpos_addr + 1]
        heading = step * dt * turn
        data.qpos[qpos_addr + 0] = x + speed * dt * np.cos(heading)
        data.qpos[qpos_addr + 1] = y + speed * dt * np.sin(heading)

    # Always pin Z and keep upright orientation, zero all velocity
    data.qpos[qpos_addr + 2] = 0.0
    data.qpos[qpos_addr + 3] = 1.0
    data.qpos[qpos_addr + 4] = 0.0
    data.qpos[qpos_addr + 5] = 0.0
    data.qpos[qpos_addr + 6] = 0.0
    data.qvel[dof_addr : dof_addr + 6] = 0.0


def compute_chair_distance(model, data) -> float:
    """Return Euclidean distance from robot base to chair in XY plane."""
    chair_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chair")
    chair_addr = model.jnt_qposadr[chair_jid]
    chair_xy   = data.qpos[chair_addr : chair_addr + 2]
    robot_xy   = data.qpos[0:2]
    return float(np.linalg.norm(robot_xy - chair_xy))


def _run_headless(model, data, robot, tick_fn):
    """Headless render loop using mujoco.Renderer + OpenCV."""
    try:
        import cv2
    except ImportError:
        print("[DEMO] pip install opencv-python  for headless display; running blind.")
        cv2 = None

    renderer = mujoco.Renderer(model, height=900, width=1200)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance  = 7.0
    cam.azimuth   = 135.0
    cam.elevation = -25.0

    robot_cam_renderer = mujoco.Renderer(model, height=480, width=640)

    frame_every = 10
    robot_cam_every = 30
    step = 0
    while True:
        tick_fn()
        mujoco.mj_step(model, data)
        if step % robot_cam_every == 0:
            robot_cam_renderer.update_scene(data, camera=SIM_CONFIG["camera"])
            robot.push_sim_frame(robot_cam_renderer.render())
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
    if not TOKEN_FACTORY_API_KEY:
        print("[DEMO] WARNING: NEBIUS_API_KEY is not set — VLM calls will fail. Set it via .env.local or export.")
    else:
        print(f"[DEMO] Nebius API key loaded ({TOKEN_FACTORY_API_KEY[:8]}...)")

    print("[DEMO] Testing Nebius API connectivity...")
    try:
        _test = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with one word: ready"}],
            max_tokens=5,
        )
        print(f"[DEMO] API test OK — model replied: {_test.choices[0].message.content.strip()!r}")
    except Exception as _e:
        print(f"[DEMO] WARNING: API test FAILED — {_e}")

    # --- 5. Start perception loop (background VLM threads) ---
    perception = PerceptionLoop(robot, client)
    perception.start()
    print("[DEMO] Perception loop started (VLM threads running at ~2 fps)")

    # --- 6. Create and start LLM agent ---
    agent = RobotAgent(client, LLM_MODEL)
    agent.start()

    # --- 7. Tick state (replaces state machine) ---
    sim_step        = 0
    _sweep_start    = None
    _sweep_action   = None
    _kick_applied   = False
    _signal_start   = None
    _last_speech_ts = [0.0]   # list so nonlocal write works in nested fn

    def tick_fn():
        nonlocal sim_step, _sweep_start, _sweep_action, _kick_applied, _signal_start

        # 1. Sim geometry (ground truth, never stale)
        sim_dist = compute_chair_distance(model, data)

        # 2. Push perception to agent (non-blocking)
        if perception.ready:
            percept = perception.get()
            robot_context = {
                "sim_dist_to_obstacle_m": round(sim_dist, 2),
                "sweep_in_progress": _sweep_start is not None,
                "sweep_elapsed_s": round(time.time() - _sweep_start, 2) if _sweep_start else 0.0,
            }
            agent.update_perception(percept, robot_context)

        # 3. Get latest agent decision (non-blocking, with safety override)
        decision = agent.get_decision(sim_dist)

        # 4. Speak if new speech (deduplicated by decision timestamp)
        if decision.speech and decision.timestamp > _last_speech_ts[0]:
            _speak(decision.speech)
            _last_speech_ts[0] = decision.timestamp

        # 5. Execute action — sweep/signal take priority once started
        if _sweep_start is not None:
            elapsed = time.time() - _sweep_start
            if _sweep_action == "sweep_left":
                robot.arm.clear_left_tick(elapsed)
            else:
                robot.arm.clear_right_tick(elapsed)
            if elapsed >= 2.0:
                print("[DEMO] Sweep complete")
                _sweep_start = None
                _sweep_action = None
        elif _signal_start is not None:
            elapsed = time.time() - _signal_start
            if elapsed < 1.5:
                robot.loco.send_velocity(vx=0, vy=0, omega=-2.0)  # turn to face human
            elif elapsed < 4.5:
                robot.stop()
                robot.arm.wave_tick(elapsed - 1.5, duration=3.0)
            else:
                _signal_start = None
                print("[DEMO] Signal complete")
        elif decision.action == "kick_chair" and not _kick_applied:
            chair_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chair")
            chair_dof = model.jnt_dofadr[chair_jid]
            data.qvel[chair_dof:chair_dof+3] = [0.0, 7.0, 4.0]    # fly sideways + upward
            data.qvel[chair_dof+3:chair_dof+6] = [2.0, 0.0, 1.0]  # spin for drama
            _kick_applied = True
            print("[DEMO] KICK! Chair sent flying")
        elif decision.action == "signal_clear" and _signal_start is None:
            robot.stop()
            _signal_start = time.time()
            print("[DEMO] Signaling path clear to human")
        elif decision.action == "walk":
            robot.loco.send_velocity(vx=decision.vx or 0.35, vy=0, omega=0)
        elif decision.action == "slow":
            robot.loco.send_velocity(vx=decision.vx or 0.15, vy=0, omega=0)
        elif decision.action == "stop":
            robot.stop()
        elif decision.action in ("sweep_left", "sweep_right"):
            robot.stop()
            _sweep_start = time.time()
            _sweep_action = decision.action
            print(f"[DEMO] Starting {decision.action}")

        # 6. Move person (follow robot when moving)
        _move_person(model, data, sim_step,
                     moving=(decision.action in ("walk", "slow"))
                             and _signal_start is None)
        sim_step += 1

    # --- 8. Launch viewer (interactive or headless fallback) ---
    print("[DEMO] Opening MuJoCo viewer... close the window to exit.")
    try:
        camera_renderer = mujoco.Renderer(model, height=480, width=640)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance  = 7.0
            viewer.cam.azimuth   = 135
            viewer.cam.elevation = -25

            step = 0
            while viewer.is_running():
                t = time.time()
                tick_fn()
                mujoco.mj_step(model, data)
                if step % 30 == 0:
                    camera_renderer.update_scene(data, camera=SIM_CONFIG["camera"])
                    robot.push_sim_frame(camera_renderer.render())
                viewer.sync()
                step += 1
                elapsed = time.time() - t
                sleep_t = model.opt.timestep - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    except RuntimeError:
        print("[DEMO] Interactive viewer unavailable — falling back to headless (OpenCV).")
        print("[DEMO] For interactive viewer on macOS: brew install glfw && mjpython sim_demo.py")
        _run_headless(model, data, robot, tick_fn)
    finally:
        perception.stop()
        agent.stop()


if __name__ == "__main__":
    run_demo()
