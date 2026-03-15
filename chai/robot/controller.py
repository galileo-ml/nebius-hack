import time

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
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
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

    def _real_camera_frame(self):
        """Capture frame from real robot's head camera."""
        # Stub: replace with actual camera SDK call (e.g. OpenCV VideoCapture
        # connected to the G1's onboard camera stream).
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def stop(self):
        self.loco.send_velocity(vx=0, vy=0, omega=0)

    def walk_forward(self, speed=0.3):
        self.loco.send_velocity(vx=speed, vy=0, omega=0)

    def turn(self, omega=0.5, duration=1.0):
        self.loco.send_velocity(vx=0, vy=0, omega=omega)
        time.sleep(duration)
        self.stop()
