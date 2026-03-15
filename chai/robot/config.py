import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env.local"))

MODE = "sim"  # "sim" or "real"

SIM_CONFIG = {
    "model_path": "unitree_mujoco/unitree_robots/g1/scene.xml",
    "camera": "front_camera"
}

REAL_CONFIG = {
    "interface": "enp3s0",       # Ethernet interface name
    "robot_ip": "192.168.123.161"
}

TOKEN_FACTORY_BASE_URL = "https://api.studio.nebius.ai/v1"
TOKEN_FACTORY_API_KEY  = os.environ.get("NEBIUS_API_KEY", "")
VLM_MODEL              = "Qwen/Qwen2.5-VL-72B-Instruct"
LLM_MODEL              = "meta-llama/Llama-3.3-70B-Instruct"
