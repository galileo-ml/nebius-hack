"""
Proximity-based perception stub for sim mode.
Drop-in replacement for PerceptionLoop — no VLM or API key required.
Reads obstacle/human positions directly from MuJoCo state.
"""

import numpy as np


OBSTACLE_DETECT_DIST = 0.8   # metres
HUMAN_DETECT_DIST    = 2.0


class SimPerception:
    def __init__(self, model, data):
        self.model = model
        self.data  = data

    def get(self):
        robot_pos = self.data.body("pelvis").xpos.copy()
        obstacle  = self._check_obstacle(robot_pos)
        human     = self._check_human(robot_pos)
        return {"human": human, "obstacle": obstacle}

    def _check_obstacle(self, robot_pos):
        try:
            chair_pos = self.data.body("chair").xpos.copy()
            dist = float(np.linalg.norm(chair_pos[:2] - robot_pos[:2]))
            if dist < OBSTACLE_DETECT_DIST:
                return {"obstacle": True, "type": "chair", "side": "center", "distance": dist}
        except Exception:
            pass
        return {"obstacle": False}

    def _check_human(self, robot_pos):
        try:
            person_pos = self.data.body("person_marker").xpos.copy()
            dist = float(np.linalg.norm(person_pos[:2] - robot_pos[:2]))
            if dist < HUMAN_DETECT_DIST:
                return {"human": True, "approaching": True, "distance": dist}
        except Exception:
            pass
        return {"human": False, "approaching": False}
