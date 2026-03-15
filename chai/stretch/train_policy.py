"""
Stretch goal: train a grasp-and-place policy on Nebius H100 instances.

Run on Nebius:
    pip install mujoco playground
    python stretch/train_policy.py

Integration path if policy transfers cleanly:
  - Replace arm.clear_right() / arm.clear_left() calls in state_machine.py
    with policy inference.
  - If it doesn't transfer: keep scripted sweep for live demo,
    show policy working in sim as a video clip during the pitch.
"""

import playground


def train():
    env = playground.load("G1JoystickFlatTerrain")  # or custom reach-and-place env

    trainer = playground.Trainer(
        env=env,
        algorithm="ppo",
        num_envs=4096,
        total_timesteps=5_000_000
    )
    trainer.train()
    trainer.save("checkpoints/g1_grasp_policy.onnx")
    print("[Stretch] Policy saved to checkpoints/g1_grasp_policy.onnx")


if __name__ == "__main__":
    train()
