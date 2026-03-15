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
            self._x   = 0.0
            self._y   = 0.0
            self._yaw = 0.0
            self._z0  = None
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
        target = np.zeros(self.model.nu)

        # Standing posture constants
        HIP_PITCH_BIAS = 0.25   # thighs angled forward at rest (rad)
        KNEE_BIAS      = 0.35   # constant knee bend for compliance (rad)
        ANKLE_BIAS     = -0.20  # ankle compensates for knee bend (rad)

        if vx > 0.01 and self.policy is None:
            t     = self.data.time
            freq  = 1.5
            amp   = min(0.65, vx * 1.8)
            phase = 2 * np.pi * freq * t

            # L leg: forward swing when sin(phase) > 0
            target[0]  =  HIP_PITCH_BIAS + amp * np.sin(phase)              # L hip_pitch
            target[1]  =  0.08 * np.sin(phase)                              # L hip_roll (lateral sway)
            target[2]  =  0.12 * np.sin(phase)                              # L hip_yaw (cross-step)
            target[3]  =  KNEE_BIAS + amp * 0.6 * max(0, np.sin(phase))     # L knee lifts during forward swing
            target[4]  =  ANKLE_BIAS - amp * 0.4 * np.sin(phase)            # L ankle

            # R leg: opposite phase
            target[6]  =  HIP_PITCH_BIAS + amp * np.sin(phase + np.pi)      # R hip_pitch
            target[7]  = -0.08 * np.sin(phase)                              # R hip_roll (mirror)
            target[8]  =  0.12 * np.sin(phase + np.pi)                      # R hip_yaw (cross-step)
            target[9]  =  KNEE_BIAS + amp * 0.6 * max(0, np.sin(phase + np.pi))  # R knee
            target[10] =  ANKLE_BIAS - amp * 0.4 * np.sin(phase + np.pi)   # R ankle

            # Arm swing unchanged
            target[15] =  0.3 * np.sin(phase + np.pi)   # L shoulder_pitch
            target[22] =  0.3 * np.sin(phase)            # R shoulder_pitch

        else:
            # Standing still — maintain posture so legs don't collapse to 0
            target[0]  = HIP_PITCH_BIAS;  target[6]  = HIP_PITCH_BIAS
            target[3]  = KNEE_BIAS;       target[9]  = KNEE_BIAS
            target[4]  = ANKLE_BIAS;      target[10] = ANKLE_BIAS

        qpos = self.data.qpos[7:]
        qvel = self.data.qvel[6:]
        n = min(len(target), len(_KP))
        torques = _KP[:n] * (target[:n] - qpos[:n]) - _KD[:n] * qvel[:n]
        self.data.ctrl[:n] = torques

        # Capture initial standing height once
        if self._z0 is None:
            self._z0 = float(self.data.qpos[2])

        # Integrate desired pose (dead reckoning)
        dt = self.model.opt.timestep
        self._yaw += omega * dt
        self._x   += vx * np.cos(self._yaw) * dt
        self._y   += vx * np.sin(self._yaw) * dt

        # Apply kinematic override: position + upright orientation
        cy, sy = np.cos(self._yaw / 2), np.sin(self._yaw / 2)
        self.data.qpos[0] = self._x
        self.data.qpos[1] = self._y
        self.data.qpos[2] = self._z0
        self.data.qpos[3] = cy   # w
        self.data.qpos[4] = 0.0  # x
        self.data.qpos[5] = 0.0  # y
        self.data.qpos[6] = sy   # z

        # Set base velocity so physics integrates from the right state
        self.data.qvel[0] = vx * np.cos(self._yaw)
        self.data.qvel[1] = vx * np.sin(self._yaw)
        self.data.qvel[2] = 0.0
        self.data.qvel[3:6] = 0.0

        # Clear any residual external forces
        self.data.xfrc_applied[:] = 0.0

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
