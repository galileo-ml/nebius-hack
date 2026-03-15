# CHAI — Implementation Plan
**Companion Humanoid AI for accessibility and real-time language translation**

---

## Overview

CHAI is a Unitree G1 humanoid robot companion that guides visually impaired users through physical spaces. It detects approaching humans, greets them in English, translates their Hindi responses aloud in real-time, and physically clears obstacles from the user's path using a scripted arm sweep. The system is built on a Token Factory agentic loop, a pre-trained G1 locomotion policy, and scripted IK arm primitives — fully testable in MuJoCo on Mac, deployable to real hardware via Ethernet config swap.

---

## Repository Structure

```
chai/
├── main.py                  # Entry point, starts all loops
├── state_machine.py         # Core CHAI state machine
├── perception/
│   ├── vlm_loop.py          # Non-blocking VLM perception loop
│   ├── prompts.py           # All VLM prompt strings
│   └── audio.py             # Microphone capture + STT
├── voice/
│   ├── tts.py               # Text-to-speech output
│   ├── translator.py        # Token Factory Hindi → English
│   └── phrases.py           # Pre-written Hindi warning phrases
├── robot/
│   ├── controller.py        # Unified robot interface (sim + real)
│   ├── locomotion.py        # Velocity command wrapper
│   ├── arm.py               # Scripted IK arm sweep primitives
│   └── config.py            # Sim vs real toggle
├── sim/
│   └── scene.py             # MuJoCo scene setup with obstacles
└── stretch/
    └── train_policy.py      # Stretch goal: MuJoCo Playground RL
```

---

## Environment Setup

### Dependencies
```bash
# Core
pip install mujoco unitree-sdk2py openai requests sounddevice numpy

# Audio / TTS
pip install speechrecognition gtts playsound pydub

# Sim
pip install mujoco-python-viewer
```

### Config toggle — `robot/config.py`
```python
MODE = "sim"  # "sim" or "real"

SIM_CONFIG = {
    "model_path": "unitree_mujoco/unitree_robots/g1/scene.xml",
    "camera": "front_camera"
}

REAL_CONFIG = {
    "interface": "enp3s0",     # Ethernet interface name
    "robot_ip": "192.168.123.161"
}

TOKEN_FACTORY_BASE_URL = "https://api.studio.nebius.ai/v1"
TOKEN_FACTORY_API_KEY  = "YOUR_KEY"
VLM_MODEL              = "Qwen/Qwen2-VL-72B-Instruct"
LLM_MODEL              = "meta-llama/Llama-3.3-70B-Instruct"
```

---

## 1. Perception Layer — `perception/vlm_loop.py`

Two non-blocking perception threads running at ~2fps independently.

### Camera frame capture
```python
import mujoco, threading, base64, time, json
import numpy as np
from PIL import Image
import io

class PerceptionLoop:
    def __init__(self, robot_controller, token_factory_client):
        self.robot = robot_controller
        self.client = token_factory_client
        self.latest = {
            "human": {"detected": False, "distance": None, "approaching": False},
            "obstacle": {"detected": False, "position": None, "side": None}
        }
        self._lock = threading.Lock()
        self._running = False

    def _capture_frame_base64(self):
        """Get current camera frame as base64 string."""
        frame = self.robot.get_camera_frame()  # returns numpy RGB array
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()

    def _query_vlm(self, prompt, image_b64):
        response = self.client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=100
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
```

### VLM Prompts — `perception/prompts.py`

```python
HUMAN_DETECTION_PROMPT = """
Analyze this image from a robot's front camera.
Is there a person visible in the frame?
Respond ONLY with valid JSON, no explanation:
{
  "human": true/false,
  "distance": "close (<1m) | medium (1-3m) | far (>3m) | null",
  "approaching": true/false
}
"""

OBSTACLE_DETECTION_PROMPT = """
Analyze this image from a robot's front camera.
Is there a physical object (chair, box, bag, etc.) blocking or partially blocking
the path within 2 meters directly ahead?
Respond ONLY with valid JSON, no explanation:
{
  "obstacle": true/false,
  "type": "chair | box | bag | other | null",
  "side": "left | center | right | null",
  "distance": "near (<0.5m) | medium (0.5-1.5m) | far (>1.5m) | null"
}
"""
```

### Non-blocking perception threads

```python
    def _human_loop(self):
        while self._running:
            try:
                frame = self._capture_frame_base64()
                result = self._query_vlm(HUMAN_DETECTION_PROMPT, frame)
                with self._lock:
                    self.latest["human"] = result
            except Exception as e:
                print(f"[Human perception error] {e}")
            time.sleep(0.5)  # 2fps

    def _obstacle_loop(self):
        while self._running:
            try:
                frame = self._capture_frame_base64()
                result = self._query_vlm(OBSTACLE_DETECTION_PROMPT, frame)
                with self._lock:
                    self.latest["obstacle"] = result
            except Exception as e:
                print(f"[Obstacle perception error] {e}")
            time.sleep(0.5)

    def start(self):
        self._running = True
        threading.Thread(target=self._human_loop, daemon=True).start()
        threading.Thread(target=self._obstacle_loop, daemon=True).start()

    def get(self):
        with self._lock:
            return dict(self.latest)
```

---

## 2. Audio Layer — `perception/audio.py` + `voice/tts.py`

### Speech-to-text (microphone capture)
```python
import speech_recognition as sr

class AudioInput:
    def __init__(self, language="hi-IN", timeout=5):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.timeout = timeout

    def listen_once(self):
        """Block until speech is detected, return transcript string."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[CHAI] Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=self.timeout)
                text = self.recognizer.recognize_google(audio, language=self.language)
                print(f"[STT] Heard: {text}")
                return text
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
```

### Text-to-speech output
```python
from gtts import gTTS
import playsound, tempfile, os

class AudioOutput:
    def speak(self, text, lang="en"):
        """Speak text aloud. lang='en' or 'hi'."""
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        tts.save(path)
        playsound.playsound(path)
        os.unlink(path)
```

### Hindi → English translation — `voice/translator.py`
```python
class Translator:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def hindi_to_english(self, hindi_text):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a translator. The user will give you Hindi text. "
                    "Translate it naturally to English. "
                    "Return ONLY the English translation, nothing else."
                )},
                {"role": "user", "content": hindi_text}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
```

### Pre-written obstacle warnings — `voice/phrases.py`
```python
# Pre-written for reliability — do NOT LLM-generate these live
OBSTACLE_WARNINGS = {
    "en": {
        "center": "Please stop. There is an obstacle directly ahead. I am clearing the path.",
        "left":   "Please stop. There is an obstacle to your left. I am clearing it.",
        "right":  "Please stop. There is an obstacle to your right. I am clearing it.",
    }
}

GREETING_EN = "Hello! I am CHAI, your navigation companion. How can I help you today?"
RESUME_EN   = "The path is clear. Please continue."
```

---

## 3. State Machine — `state_machine.py`

```python
from enum import Enum, auto

class State(Enum):
    IDLE              = auto()
    HUMAN_DETECTED    = auto()
    GREETING          = auto()
    AWAIT_RESPONSE    = auto()
    TRANSLATING       = auto()
    GUIDING           = auto()
    OBSTACLE_DETECTED = auto()
    WARNING           = auto()
    CLEARING          = auto()
    RESUMING          = auto()

class CHAI:
    def __init__(self, perception, audio_in, audio_out, translator, robot):
        self.perception   = perception
        self.audio_in     = audio_in
        self.audio_out    = audio_out
        self.translator   = translator
        self.robot        = robot
        self.state        = State.IDLE
        self.warned_obstacles = set()  # tracks obstacles already warned about

    def _obstacle_key(self, obs):
        """Deduplicate warnings by position+type."""
        return f"{obs.get('type')}_{obs.get('side')}"

    def run(self):
        print("[CHAI] Starting main loop...")
        while True:
            percept = self.perception.get()
            self._tick(percept)

    def _tick(self, percept):
        human    = percept["human"]
        obstacle = percept["obstacle"]

        # --- IDLE ---
        if self.state == State.IDLE:
            if human["detected"] and human["approaching"]:
                self.state = State.HUMAN_DETECTED

        # --- HUMAN DETECTED ---
        elif self.state == State.HUMAN_DETECTED:
            self.robot.stop()
            self.state = State.GREETING

        # --- GREETING ---
        elif self.state == State.GREETING:
            self.audio_out.speak(GREETING_EN, lang="en")
            self.state = State.AWAIT_RESPONSE

        # --- AWAIT RESPONSE ---
        elif self.state == State.AWAIT_RESPONSE:
            spoken = self.audio_in.listen_once()
            if spoken:
                self._hindi_response = spoken
                self.state = State.TRANSLATING
            else:
                # No response heard — proceed to guiding
                self.state = State.GUIDING

        # --- TRANSLATING ---
        elif self.state == State.TRANSLATING:
            translation = self.translator.hindi_to_english(self._hindi_response)
            announcement = f"The person said: {translation}"
            print(f"[Translation] {announcement}")
            self.audio_out.speak(announcement, lang="en")
            self.state = State.GUIDING

        # --- GUIDING ---
        elif self.state == State.GUIDING:
            self.robot.walk_forward()
            if obstacle["detected"]:
                key = self._obstacle_key(obstacle)
                if key not in self.warned_obstacles:
                    self.warned_obstacles.add(key)
                    self._current_obstacle = obstacle
                    self.state = State.OBSTACLE_DETECTED

        # --- OBSTACLE DETECTED ---
        elif self.state == State.OBSTACLE_DETECTED:
            self.robot.stop()
            self.state = State.WARNING

        # --- WARNING ---
        elif self.state == State.WARNING:
            side = self._current_obstacle.get("side", "center")
            warning_text = OBSTACLE_WARNINGS["en"].get(side, OBSTACLE_WARNINGS["en"]["center"])
            self.audio_out.speak(warning_text, lang="en")
            self.state = State.CLEARING

        # --- CLEARING ---
        elif self.state == State.CLEARING:
            side = self._current_obstacle.get("side", "center")
            if side == "left":
                self.robot.arm.clear_left()
            else:
                self.robot.arm.clear_right()
            self.state = State.RESUMING

        # --- RESUMING ---
        elif self.state == State.RESUMING:
            self.audio_out.speak(RESUME_EN, lang="en")
            self.state = State.GUIDING
```

---

## 4. Robot Controller — `robot/controller.py`

Unified interface that works identically in sim and on real hardware.

```python
from robot.config import MODE, SIM_CONFIG, REAL_CONFIG
from robot.locomotion import LocomotionController
from robot.arm import ArmController

class RobotController:
    def __init__(self):
        self.mode = MODE
        if MODE == "sim":
            self._init_sim()
        else:
            self._init_real()
        self.arm = ArmController(self)

    def _init_sim(self):
        import mujoco
        self.model = mujoco.MjModel.from_xml_path(SIM_CONFIG["model_path"])
        self.data  = mujoco.MjData(self.model)
        self.loco  = LocomotionController(mode="sim", model=self.model, data=self.data)

    def _init_real(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
        ChannelFactoryInitialize(0, REAL_CONFIG["interface"])
        self.loco = LocomotionController(mode="real")

    def get_camera_frame(self):
        if self.mode == "sim":
            return self._sim_camera_frame()
        else:
            return self._real_camera_frame()

    def _sim_camera_frame(self):
        import mujoco
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data, camera=SIM_CONFIG["camera"])
        return renderer.render()

    def stop(self):
        self.loco.send_velocity(vx=0, vy=0, omega=0)

    def walk_forward(self, speed=0.3):
        self.loco.send_velocity(vx=speed, vy=0, omega=0)

    def turn(self, omega=0.5, duration=1.0):
        import time
        self.loco.send_velocity(vx=0, vy=0, omega=omega)
        time.sleep(duration)
        self.stop()
```

---

## 5. Locomotion — `robot/locomotion.py`

Wraps the pre-trained G1 policy from `unitree_rl_gym`.

```python
import time
import numpy as np

class LocomotionController:
    def __init__(self, mode, model=None, data=None):
        self.mode = mode
        if mode == "sim":
            self.model = model
            self.data  = data
            self._load_policy()
        else:
            self._init_real_sdk()

    def _load_policy(self):
        import torch
        # Pre-trained checkpoint from unitree_rl_gym deploy/pre_train/
        self.policy = torch.jit.load("checkpoints/g1_locomotion.pt")
        self.policy.eval()

    def _init_real_sdk(self):
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.core.channel import ChannelPublisher
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

    def send_velocity(self, vx=0.0, vy=0.0, omega=0.0):
        """Send high-level velocity target to policy controller."""
        if self.mode == "sim":
            self._sim_velocity_cmd(vx, vy, omega)
        else:
            self._real_velocity_cmd(vx, vy, omega)

    def _sim_velocity_cmd(self, vx, vy, omega):
        # Build observation from sim state, run policy, apply joint torques
        obs = self._build_observation(vx, vy, omega)
        with torch.no_grad():
            action = self.policy(torch.FloatTensor(obs)).numpy()
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)

    def _build_observation(self, vx, vy, omega):
        # Standard G1 RL obs: ang_vel, projected_gravity, commands, joint_pos, joint_vel, last_action
        ang_vel    = self.data.sensor("imu_gyro").data
        gravity    = self._projected_gravity()
        commands   = np.array([vx, vy, omega])
        joint_pos  = self.data.qpos[7:] - self.default_joint_pos
        joint_vel  = self.data.qvel[6:]
        return np.concatenate([ang_vel, gravity, commands, joint_pos, joint_vel])
```

---

## 6. Arm Sweep Primitives — `robot/arm.py`

Scripted IK trajectories — no training required, fully deterministic.

```python
import numpy as np
import time

class ArmController:
    def __init__(self, robot):
        self.robot = robot
        # G1 right arm joints (7 DOF): shoulder_pitch, shoulder_roll, shoulder_yaw,
        #                               elbow, wrist_roll, wrist_pitch, wrist_yaw
        self.RIGHT_ARM_JOINTS = list(range(13, 20))  # joint indices in G1 model
        self.LEFT_ARM_JOINTS  = list(range(20, 27))

    def clear_right(self, duration=2.0):
        """Extend right arm forward, sweep left, retract."""
        print("[ARM] Clearing obstacle on right...")
        self._execute_trajectory(self._right_sweep_trajectory(), duration)

    def clear_left(self, duration=2.0):
        """Extend left arm forward, sweep right, retract."""
        print("[ARM] Clearing obstacle on left...")
        self._execute_trajectory(self._left_sweep_trajectory(), duration)

    def _right_sweep_trajectory(self):
        # Waypoints: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, ...]
        # Tuned in MuJoCo to avoid self-collision
        return [
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # neutral
            np.array([-0.5, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0]),  # extend forward
            np.array([-0.5,  0.5, 0.0, 0.8, 0.0, 0.0, 0.0]),  # sweep laterally
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # retract
        ]

    def _left_sweep_trajectory(self):
        return [
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-0.5,  0.3, 0.0, 0.8, 0.0, 0.0, 0.0]),
            np.array([-0.5, -0.5, 0.0, 0.8, 0.0, 0.0, 0.0]),
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

    def _execute_trajectory(self, waypoints, total_duration):
        dt = total_duration / (len(waypoints) - 1)
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end   = waypoints[i + 1]
            steps = int(dt / 0.02)  # 50Hz control
            for s in range(steps):
                alpha = s / steps
                target = (1 - alpha) * start + alpha * end
                self._send_arm_joints(target)
                time.sleep(0.02)

    def _send_arm_joints(self, joint_targets):
        if self.robot.mode == "sim":
            joints = self.robot.robot_controller.RIGHT_ARM_JOINTS
            for i, j in enumerate(joints):
                self.robot.data.ctrl[j] = joint_targets[i]
            mujoco.mj_step(self.robot.model, self.robot.data)
        else:
            # Send via unitree_sdk2 low-level joint commands
            self._real_arm_cmd(joint_targets)
```

**Important:** Before running on real hardware, verify each waypoint in MuJoCo visually using the MuJoCo viewer to confirm no self-collision:

```python
# sim/verify_arm.py — run this standalone to inspect sweep trajectories
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene.xml")
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    arm = ArmController(None)
    for waypoint in arm._right_sweep_trajectory():
        arm._send_arm_joints(waypoint)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.5)
```

---

## 7. Sim Scene — `sim/scene.py`

Add a static chair obstacle to the MuJoCo scene for demo testing.

```xml
<!-- Add inside unitree_mujoco g1/scene.xml worldbody -->
<body name="chair" pos="1.5 0 0.25">
  <geom type="box" size="0.25 0.25 0.25" rgba="0.6 0.4 0.2 1" mass="5"/>
</body>

<body name="person_marker" pos="3.0 0 0.9">
  <geom type="cylinder" size="0.2 0.9" rgba="0.2 0.5 0.9 0.5"/>
</body>
```

---

## 8. Entry Point — `main.py`

```python
from openai import OpenAI
from perception.vlm_loop import PerceptionLoop
from perception.audio import AudioInput
from voice.tts import AudioOutput
from voice.translator import Translator
from robot.controller import RobotController
from robot.config import TOKEN_FACTORY_BASE_URL, TOKEN_FACTORY_API_KEY, LLM_MODEL
from state_machine import CHAI

def main():
    # Token Factory client (OpenAI-compatible)
    client = OpenAI(
        base_url=TOKEN_FACTORY_BASE_URL,
        api_key=TOKEN_FACTORY_API_KEY
    )

    robot       = RobotController()
    perception  = PerceptionLoop(robot, client)
    audio_in    = AudioInput(language="hi-IN")
    audio_out   = AudioOutput()
    translator  = Translator(client, LLM_MODEL)

    chai = CHAI(perception, audio_in, audio_out, translator, robot)

    perception.start()
    print("[CHAI] System ready.")
    chai.run()

if __name__ == "__main__":
    main()
```

---

## 9. Real Hardware Deployment Checklist

Before switching `MODE = "real"`:

- [ ] Connect MacBook to G1 via Ethernet
- [ ] Set static IP: `192.168.123.222`, netmask `255.255.255.0`
- [ ] Ping `192.168.123.161` to confirm connection
- [ ] Suspend G1 on stand (never test locomotion on ground first)
- [ ] Start G1 in zero-torque mode, press L2+R2 for debug mode
- [ ] Disable sport_mode for low-level control access
- [ ] Run arm sweep verification with robot suspended before ground test
- [ ] Test `stop()` command responds within 100ms before any walking test
- [ ] Record a backup video of sim demo before hardware session

Switch config:
```python
# robot/config.py
MODE = "real"
```

No other code changes needed.

---

## 10. Stretch Goal — `stretch/train_policy.py`

**Assigned to one teammate. Independent of main CHAI loop.**

Goal: train a grasp-and-place policy that picks up a lightweight object and places it to the side, replacing the scripted arm sweep.

```bash
# On Nebius H100 instance
pip install mujoco playground

python stretch/train_policy.py
```

```python
# stretch/train_policy.py
import playground

env = playground.load("G1JoystickFlatTerrain")  # or custom reach-and-place env

# Train with PPO
trainer = playground.Trainer(
    env=env,
    algorithm="ppo",
    num_envs=4096,
    total_timesteps=5_000_000
)
trainer.train()
trainer.save("checkpoints/g1_grasp_policy.onnx")
```

Integration path if policy transfers cleanly:
- Replace `arm.clear_right()` / `arm.clear_left()` calls in `state_machine.py` with policy inference
- If it doesn't transfer: keep scripted sweep for live demo, show policy working in sim as a video clip during the pitch

---

## 11. Demo Script (Rehearse This)

**Scene setup:** Chair placed ~1.5m in front of G1. One teammate walks toward robot from ~3m away.

1. CHAI boots, enters `IDLE`
2. Teammate walks toward robot → VLM detects human approaching
3. CHAI stops, speaks: *"Hello! I am CHAI, your navigation companion. How can I help you today?"*
4. Teammate responds in Hindi: *"Namaste, mujhe aage jaana hai"* ("Hello, I need to go forward")
5. CHAI listens, translates: *"The person said: Hello, I need to go forward"*
6. CHAI begins walking forward (`GUIDING` state)
7. VLM detects chair obstacle → CHAI stops
8. CHAI speaks: *"Please stop. There is an obstacle directly ahead. I am clearing the path."*
9. G1 arm extends, sweeps chair aside, retracts
10. CHAI speaks: *"The path is clear. Please continue."*
11. CHAI resumes walking forward

**Total demo time: ~15 seconds. Fully repeatable.**

**Backup plan:** If hardware unavailable or unstable — run identical loop in MuJoCo with viewer visible on laptop. Demo is equally legible.

---

## 12. Pitch Notes (3 Minutes)

- **Lead with the human:** "1.3 billion people live with vision impairment. Language barriers compound this daily."
- **Show, don't tell:** Start the demo immediately, narrate as it runs
- **Call out the tech stack naturally:** "CHAI uses Nebius Token Factory's VLM for real-time scene understanding, and a multilingual LLM for live Hindi translation — no pretrained translation model, it's generative"
- **Name the hard problem:** "The engineering challenge is grounding open-ended scene descriptions into reliable robot actions fast enough for real-time use"
- **Stretch goal as vision:** "With more time, we replace the scripted arm with a learned grasp policy trained on Nebius H100s in under 15 minutes"
