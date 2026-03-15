import threading
import base64
import time
import json
import io

import numpy as np
from PIL import Image

from perception.prompts import HUMAN_DETECTION_PROMPT, OBSTACLE_DETECTION_PROMPT
from robot.config import VLM_MODEL


class PerceptionLoop:
    def __init__(self, robot_controller, token_factory_client):
        self.robot  = robot_controller
        self.client = token_factory_client
        self.latest = {
            "human":    {"detected": False, "distance": None, "approaching": False},
            "obstacle": {"detected": False, "position": None, "side": None}
        }
        self._lock    = threading.Lock()
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

    def _human_loop(self):
        while self._running:
            try:
                frame  = self._capture_frame_base64()
                result = self._query_vlm(HUMAN_DETECTION_PROMPT, frame)
                with self._lock:
                    self.latest["human"] = result
            except Exception as e:
                print(f"[Human perception error] {e}")
            time.sleep(0.5)  # 2fps

    def _obstacle_loop(self):
        while self._running:
            try:
                frame  = self._capture_frame_base64()
                result = self._query_vlm(OBSTACLE_DETECTION_PROMPT, frame)
                with self._lock:
                    self.latest["obstacle"] = result
            except Exception as e:
                print(f"[Obstacle perception error] {e}")
            time.sleep(0.5)

    def start(self):
        self._running = True
        threading.Thread(target=self._human_loop,    daemon=True).start()
        threading.Thread(target=self._obstacle_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def get(self):
        with self._lock:
            return dict(self.latest)
