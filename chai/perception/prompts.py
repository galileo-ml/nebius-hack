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
  "distance": "near (<1.0m) | medium (1.0-2.0m) | far (>2.0m) | null"
}
"""

