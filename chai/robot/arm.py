import mujoco
import numpy as np
import time


class ArmController:
    def __init__(self, robot):
        self.robot = robot
        # G1 right arm joints (7 DOF): shoulder_pitch, shoulder_roll, shoulder_yaw,
        #                               elbow, wrist_roll, wrist_pitch, wrist_yaw
        # G1 29-DOF actuator layout: 0-5 left leg, 6-11 right leg, 12-14 waist,
        # 15-21 left arm, 22-28 right arm
        self.RIGHT_ARM_JOINTS = list(range(22, 29))
        self.LEFT_ARM_JOINTS  = list(range(15, 22))

    def clear_right_tick(self, elapsed, duration=2.0):
        """Non-blocking per-tick arm sweep. Call each physics tick with elapsed wall time."""
        waypoints = self._right_sweep_trajectory()
        n_segments = len(waypoints) - 1
        seg_dur = duration / n_segments
        seg_idx = min(int(elapsed / seg_dur), n_segments - 1)
        alpha = (elapsed - seg_idx * seg_dur) / seg_dur
        alpha = float(np.clip(alpha, 0.0, 1.0))
        target = (1 - alpha) * waypoints[seg_idx] + alpha * waypoints[seg_idx + 1]
        self._send_arm_joints(target, self.RIGHT_ARM_JOINTS)

    def clear_left_tick(self, elapsed, duration=2.0):
        """Non-blocking per-tick left arm sweep. Call each physics tick with elapsed wall time."""
        waypoints = self._left_sweep_trajectory()
        n_segments = len(waypoints) - 1
        seg_dur = duration / n_segments
        seg_idx = min(int(elapsed / seg_dur), n_segments - 1)
        alpha = (elapsed - seg_idx * seg_dur) / seg_dur
        alpha = float(np.clip(alpha, 0.0, 1.0))
        target = (1 - alpha) * waypoints[seg_idx] + alpha * waypoints[seg_idx + 1]
        self._send_arm_joints(target, self.LEFT_ARM_JOINTS)

    def wave_tick(self, elapsed, duration=3.0):
        """Non-blocking per-tick wave gesture. Call each physics tick with elapsed wall time."""
        waypoints = [
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # neutral
            np.array([-1.2,  0.4, 0.0, 0.3, 0.0, 0.0, 0.0]),  # raise arm high
            np.array([-1.2,  0.1, 0.0, 0.3, 0.0, 0.0, 0.0]),  # wave in
            np.array([-1.2,  0.5, 0.0, 0.3, 0.0, 0.0, 0.0]),  # wave out
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # retract
        ]
        n_segments = len(waypoints) - 1
        seg_dur = duration / n_segments
        seg_idx = min(int(elapsed / seg_dur), n_segments - 1)
        alpha = (elapsed - seg_idx * seg_dur) / seg_dur
        alpha = float(np.clip(alpha, 0.0, 1.0))
        target = (1 - alpha) * waypoints[seg_idx] + alpha * waypoints[seg_idx + 1]
        self._send_arm_joints(target, self.LEFT_ARM_JOINTS)

    def stop_gesture_tick(self, elapsed, duration=3.0):
        """Non-blocking per-tick stop gesture (hold up hand)."""
        waypoints = [
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # neutral
            np.array([-1.5,  0.2, 0.0, 1.0, 0.0, 0.5, 0.0]),  # hold hand up
            np.array([-1.5,  0.2, 0.0, 1.0, 0.0, 0.5, 0.0]),  # hold
            np.array([-1.5,  0.2, 0.0, 1.0, 0.0, 0.5, 0.0]),  # hold
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # retract
        ]
        n_segments = len(waypoints) - 1
        seg_dur = duration / n_segments
        seg_idx = min(int(elapsed / seg_dur), n_segments - 1)
        alpha = (elapsed - seg_idx * seg_dur) / seg_dur
        alpha = float(np.clip(alpha, 0.0, 1.0))
        target = (1 - alpha) * waypoints[seg_idx] + alpha * waypoints[seg_idx + 1]
        self._send_arm_joints(target, self.LEFT_ARM_JOINTS)

    def clear_right(self, duration=2.0):
        """Extend right arm forward, sweep left, retract."""
        print("[ARM] Clearing obstacle on right...")
        self._execute_trajectory(self._right_sweep_trajectory(), duration, self.RIGHT_ARM_JOINTS)

    def clear_left(self, duration=2.0):
        """Extend left arm forward, sweep right, retract."""
        print("[ARM] Clearing obstacle on left...")
        self._execute_trajectory(self._left_sweep_trajectory(), duration, self.LEFT_ARM_JOINTS)

    def _right_sweep_trajectory(self):
        # Waypoints: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, ...]
        # Tuned in MuJoCo to avoid self-collision
        return [
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # neutral
            np.array([-1.0, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0]),  # extend forward
            np.array([-0.5,  0.5, 0.0, 0.8, 0.0, 0.0, 0.0]),  # sweep laterally
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # retract
        ]

    def _left_sweep_trajectory(self):
        return [
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-1.0,  0.3, 0.0, 0.8, 0.0, 0.0, 0.0]),
            np.array([-1.0, -0.5, 0.0, 0.8, 0.0, 0.0, 0.0]),
            np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

    def _execute_trajectory(self, waypoints, total_duration, joints):
        dt = total_duration / (len(waypoints) - 1)
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end   = waypoints[i + 1]
            steps = int(dt / 0.02)  # 50Hz control
            for s in range(steps):
                alpha = s / steps
                target = (1 - alpha) * start + alpha * end
                self._send_arm_joints(target, joints)
                time.sleep(0.02)

    def _send_arm_joints(self, joint_targets, joints=None):
        if joints is None:
            joints = self.RIGHT_ARM_JOINTS
        if self.robot.mode == "sim":
            for i, j in enumerate(joints):
                self.robot.data.ctrl[j] = joint_targets[i]
            # Physics stepping is handled by the main loop
        else:
            self._real_arm_cmd(joint_targets)

    def _real_arm_cmd(self, joint_targets):
        """Send joint position targets via unitree_sdk2 low-level API."""
        # Stub: replace with actual LowCmd_ motor command construction
        print(f"[REAL ARM] joint targets: {joint_targets}")
