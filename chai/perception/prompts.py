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

PATH_CLEAR_PROMPT = """
Analyze this image from a robot's front camera.
Is the path directly ahead clear of obstacles within 2 meters?
Respond ONLY with valid JSON, no explanation:
{"clear": true/false, "reason": "brief description"}
"""

ACTION_PLANNING_PROMPT = """
You control a robot arm that can sweep obstacles out of the way.
The robot's front camera has detected an obstacle described by the JSON below.
Which single arm action should the robot take to clear the path?
Respond ONLY with valid JSON, no explanation:
{"action": "sweep_left | sweep_right", "reason": "brief description"}
"""
