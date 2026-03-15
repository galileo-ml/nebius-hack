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
import argparse
import tempfile

# Allow imports from the chai package without installing
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
import mujoco.viewer
from openai import OpenAI

from sim.scene import patch_scene_xml, inject_marble_mesh, inject_splat_mesh, HQ_MESH_GLB
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
    """Print announcement (audio disabled)."""
    print(f"[CHAI] {text}")


def _move_person(model, data, robot_xy):
    """Move person_marker toward robot, keeping ~2.5m following distance."""
    joint_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "person_marker")
    qpos_addr = model.jnt_qposadr[joint_id]
    dof_addr  = model.jnt_dofadr[joint_id]
    person_xy = data.qpos[qpos_addr : qpos_addr + 2].copy()
    delta = robot_xy - person_xy
    dist  = np.linalg.norm(delta)
    if dist > 2.5:                            # keep ~2.5m following distance
        dt        = model.opt.timestep
        speed     = 0.6                       # m/s
        direction = delta / dist
        data.qpos[qpos_addr + 0] += direction[0] * speed * dt
        data.qpos[qpos_addr + 1] += direction[1] * speed * dt
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

    renderer = mujoco.Renderer(model, height=480, width=640)
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


def run_demo(world_mesh: bool = False):
    # --- 1. Patch scene XML ---
    patched_xml = patch_scene_xml(SCENE_XML)
    print(f"[DEMO] Loaded patched scene: {patched_xml}")

    # --- 2. Load MuJoCo model ---
    if world_mesh:
        print("[DEMO] Loading World Labs event hall (HQ mesh + wall colliders + splat)")
        with open(patched_xml) as f:
            xml_str = f.read()
        xml_str = inject_marble_mesh(xml_str, glb_path=HQ_MESH_GLB, add_walls=True)
        xml_str_with_splat = xml_str
        try:
            xml_str_with_splat = inject_splat_mesh(xml_str)
        except Exception as e:
            print(f"[DEMO] WARNING: Splat injection failed ({e}) — mesh-only")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".xml", delete=False,
            dir=os.path.dirname(patched_xml)
        )
        tmp.write(xml_str_with_splat.encode())
        tmp.close()
        try:
            model = mujoco.MjModel.from_xml_path(tmp.name)
        except Exception as e:
            print(f"[DEMO] WARNING: Model load failed ({e}) — retrying without splat")
            tmp2 = tempfile.NamedTemporaryFile(
                suffix=".xml", delete=False,
                dir=os.path.dirname(patched_xml)
            )
            tmp2.write(xml_str.encode())
            tmp2.close()
            model = mujoco.MjModel.from_xml_path(tmp2.name)
    else:
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
    _sweep_applied  = False
    _kick_start     = None
    _kick_applied   = False
    _signal_start   = None
    _signal_applied = False
    _signal_stop_start = None
    _signal_stop_applied = False
    _last_speech_ts = [0.0]   # list so nonlocal write works in nested fn

    def tick_fn():
        nonlocal sim_step, _sweep_start, _sweep_action, _sweep_applied, _kick_start, _kick_applied, _signal_start, _signal_applied, _signal_stop_start, _signal_stop_applied

        # 1. Sim geometry (ground truth, never stale)
        sim_dist = compute_chair_distance(model, data)

        # 2. Push perception to agent (non-blocking)
        if perception.ready:
            percept = perception.get()
            robot_context = {
                "sim_dist_to_obstacle_m": round(sim_dist, 2),
                "action_in_progress": (
                    "sweep"        if _sweep_start       else
                    "kick"         if _kick_start        else
                    "signal_stop"  if _signal_stop_start else
                    "signal_clear" if _signal_start      else "none"
                ),
                "signal_stop_already_applied": _signal_stop_applied,
                "sweep_already_applied": _sweep_applied,
                "kick_already_applied": _kick_applied,
                "signal_clear_already_applied": _signal_applied,
            }
            agent.update_perception(percept, robot_context)

        # 3. Get latest agent decision (non-blocking, with safety override)
        decision = agent.get_decision(sim_dist)

        # 4. Speak if new speech (deduplicated by decision timestamp)
        if decision.speech and decision.timestamp > _last_speech_ts[0]:
            _speak(decision.speech)
            _last_speech_ts[0] = decision.timestamp

        # 5. Execute action — sweep/kick/signal take priority once started
        if _sweep_start is not None:
            elapsed = time.time() - _sweep_start
            if _sweep_action == "sweep_left":
                robot.arm.clear_left_tick(elapsed)
            else:
                robot.arm.clear_right_tick(elapsed)
                # Artificially push chair away during right sweep so it reliably clears
                if 0.9 <= elapsed <= 1.0:
                    try:
                        chair_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chair")
                        chair_dof = model.jnt_dofadr[chair_jid]
                        data.qvel[chair_dof:chair_dof+3]   = [2.0, -5.0, 2.0]
                        data.qvel[chair_dof+3:chair_dof+6] = [1.0, 1.0, 1.0]
                    except Exception:
                        pass
            if elapsed >= 2.0:
                print("[DEMO] Sweep complete")
                _sweep_start = None
                _sweep_action = None
        elif _kick_start is not None:
            elapsed = time.time() - _kick_start
            robot.arm.kick_tick(elapsed)
            # Artificially launch chair forward during kick follow-through
            if 0.6 <= elapsed <= 0.8:
                try:
                    chair_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chair")
                    chair_dof = model.jnt_dofadr[chair_jid]
                    data.qvel[chair_dof:chair_dof+3]   = [5.0, 0.0, 3.0]   # forward + up
                    data.qvel[chair_dof+3:chair_dof+6] = [0.5, 0.5, 0.5]
                except Exception:
                    pass
            if elapsed >= 1.5:
                print("[DEMO] Kick complete")
                _kick_start = None
        elif decision.action == "kick" and not _kick_applied:
            robot.stop()
            _kick_start   = time.time()
            _kick_applied = True
            print("[DEMO] Starting kick")
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
        elif _signal_stop_start is not None:
            elapsed = time.time() - _signal_stop_start
            if elapsed < 1.5:
                robot.loco.send_velocity(vx=0, vy=0, omega=-2.0)  # turn to face human
            elif elapsed < 4.5:
                robot.stop()
                robot.arm.stop_gesture_tick(elapsed - 1.5, duration=3.0)
            else:
                _signal_stop_start = None
                print("[DEMO] Signal stop complete")
        elif decision.action == "signal_clear" and not _signal_applied:
            robot.stop()
            _signal_start = time.time()
            _signal_applied = True
            print("[DEMO] Signaling path clear to human")
        elif decision.action == "signal_stop" and not _signal_stop_applied:
            robot.stop()
            _signal_stop_start = time.time()
            _signal_stop_applied = True
            print("[DEMO] Signaling stop to human")
        elif decision.action in ("sweep_left", "sweep_right") and not _sweep_applied:
            robot.stop()
            _sweep_start = time.time()
            _sweep_action = decision.action
            _sweep_applied = True
            print(f"[DEMO] Starting {decision.action}")
        elif decision.action == "walk":
            robot.loco.send_velocity(vx=decision.vx or 0.20, vy=0, omega=0)
        elif decision.action == "slow":
            robot.loco.send_velocity(vx=decision.vx or 0.10, vy=0, omega=0)
        elif decision.action == "stop":
            robot.stop()

        # 6. Move person (always follows robot)
        _move_person(model, data, data.qpos[0:2].copy())
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-mesh", action="store_true", default=False,
                        help="Load World Labs event hall (HQ mesh + wall colliders + Gaussian splat)")
    args = parser.parse_args()
    run_demo(world_mesh=args.world_mesh)
