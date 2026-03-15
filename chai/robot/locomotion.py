import os
import time
import numpy as np


# PD gains: legs (0-11), waist (12-14), arms (15-28)
_KP = np.array([100]*12 + [50]*3 + [20]*14, dtype=float)
_KD = np.array([ 10]*12 + [ 5]*3 + [ 2]*14, dtype=float)

CHECKPOINT = os.path.join(os.path.dirname(__file__), "../../checkpoints/g1_locomotion.pt")


class LocomotionController:
    def __init__(self, mode, model=None, data=None):
        self.mode = mode
        self.policy = None
        if mode == "sim":
            self.model = model
            self.data  = data
            self._torso_id = None
            self._load_policy_or_pd()
        else:
            self._init_real_sdk()

    def _load_policy_or_pd(self):
        if os.path.exists(CHECKPOINT):
            try:
                import torch
                self.policy = torch.jit.load(CHECKPOINT)
                self.policy.eval()
                self.default_joint_pos = np.zeros(self.model.nv - 6)
                print("[LOCO] Loaded trained policy checkpoint.")
                return
            except Exception as e:
                print(f"[LOCO] Failed to load checkpoint ({e}), falling back to PD controller.")
        print("[LOCO] Using PD position controller (no checkpoint).")
        # Find torso body index for external force application
        try:
            import mujoco
            self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        except Exception:
            self._torso_id = 1  # fallback

    def _init_real_sdk(self):
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.core.channel import ChannelPublisher
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

    def send_velocity(self, vx=0.0, vy=0.0, omega=0.0):
        if self.mode == "sim":
            self._sim_velocity_cmd(vx, vy, omega)
        else:
            self._real_velocity_cmd(vx, vy, omega)

    def _sim_velocity_cmd(self, vx, vy, omega):
        if self.policy is not None:
            self._policy_step(vx, vy, omega)
        else:
            self._pd_step(vx, vy, omega)

    def _pd_step(self, vx, vy, omega):
        import mujoco
        # Standing pose: all joints at zero
        target = np.zeros(self.model.nu)
        qpos = self.data.qpos[7:]   # skip free-joint (pos + quat)
        qvel = self.data.qvel[6:]
        n = min(len(target), len(_KP))
        torques = _KP[:n] * (target[:n] - qpos[:n]) - _KD[:n] * qvel[:n]
        self.data.ctrl[:n] = torques

        # Apply external forward force on torso to produce visible motion
        if vx != 0.0 and self._torso_id is not None:
            force_scale = 30.0
            self.data.xfrc_applied[self._torso_id, 0] = vx * force_scale
        elif self._torso_id is not None:
            self.data.xfrc_applied[self._torso_id, 0] = 0.0

    def _policy_step(self, vx, vy, omega):
        import torch
        import mujoco
        obs = self._build_observation(vx, vy, omega)
        with torch.no_grad():
            action = self.policy(torch.FloatTensor(obs)).numpy()
        n = min(len(action), len(self.data.ctrl))
        self.data.ctrl[:n] = action[:n]

    def _build_observation(self, vx, vy, omega):
        ang_vel   = self.data.sensor("imu_gyro").data
        gravity   = self._projected_gravity()
        commands  = np.array([vx, vy, omega])
        joint_pos = self.data.qpos[7:] - self.default_joint_pos
        joint_vel = self.data.qvel[6:]
        return np.concatenate([ang_vel, gravity, commands, joint_pos, joint_vel])

    def _projected_gravity(self):
        quat = self.data.qpos[3:7]
        w, x, y, z = quat
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = 1 - 2 * (x * x + y * y)
        return np.array([gx, gy, gz])

    def _real_velocity_cmd(self, vx, vy, omega):
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        print(f"[REAL] velocity cmd: vx={vx:.2f} vy={vy:.2f} omega={omega:.2f}")
