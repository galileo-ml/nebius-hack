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
        # Default neutral joint positions (43 DOF for G1)
        self.default_joint_pos = np.zeros(self.model.nv - 6)

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
        import torch
        import mujoco
        # Build observation from sim state, run policy, apply joint torques
        obs = self._build_observation(vx, vy, omega)
        with torch.no_grad():
            action = self.policy(torch.FloatTensor(obs)).numpy()
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)

    def _build_observation(self, vx, vy, omega):
        # Standard G1 RL obs: ang_vel, projected_gravity, commands, joint_pos, joint_vel, last_action
        ang_vel   = self.data.sensor("imu_gyro").data
        gravity   = self._projected_gravity()
        commands  = np.array([vx, vy, omega])
        joint_pos = self.data.qpos[7:] - self.default_joint_pos
        joint_vel = self.data.qvel[6:]
        return np.concatenate([ang_vel, gravity, commands, joint_pos, joint_vel])

    def _projected_gravity(self):
        """Project world gravity vector into robot body frame."""
        # Quaternion (w, x, y, z) from free joint
        quat = self.data.qpos[3:7]
        w, x, y, z = quat
        # Rotation matrix column for world z-axis in body frame
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = 1 - 2 * (x * x + y * y)
        return np.array([gx, gy, gz])

    def _apply_action(self, action):
        """Write policy action (joint position targets) to MuJoCo ctrl."""
        n = min(len(action), len(self.data.ctrl))
        self.data.ctrl[:n] = action[:n]

    def _real_velocity_cmd(self, vx, vy, omega):
        """Send velocity command via unitree_sdk2 high-level API."""
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        # Stub: high-level velocity API depends on G1 firmware version
        # Replace with actual SportModeState subscriber + cmd publisher
        print(f"[REAL] velocity cmd: vx={vx:.2f} vy={vy:.2f} omega={omega:.2f}")
