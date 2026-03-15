# Overview

Below are two links to github repos that detail how to use the Nebious platform to simulate the new unitree robot.

- https://github.com/DavidDobas/video2robot-hackathon
- https://github.com/DavidDobas/mjlab-hackathon

# Instructions

Physical AI UFB hackathon setup

In this guide, we will set up a pipeline for training robot motions from video. We will train on Nebius, with H100 GPU.

We are using the following repositories:
video2robot: https://github.com/DavidDobas/video2robot-hackathon
mjlab: https://github.com/DavidDobas/mjlab-hackathon

Join UFB Discord for support! https://discord.gg/j8AaY4Wnu
In onboarding channel, react to “🚀 Hackathon Access” and you’ll be added to #nebius-build-hackathon
SSH setup
Check if you already have a key
ls ~/.ssh/id_ed25519.pub
If the file exists, you’re done.
Create a new key (if it doesn’t exist)
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-keygen -t nebius -C "teodoro@gmail.com"
Press Enter to accept defaults.
View the public key
cat ~/.ssh/id_ed25519.pub
Nebius setup
Create an account in Nebius.
Go to Billing and set up your card. Then apply promo code
Then go Compute -> Virtual machines -> Create virtual machine
Select 
Project: default-project-eu-north1
Platform: NVIDIA® H100 NVLink
Boot disk: Ubuntu 24.04 LTS for NVIDIA® GPUs (CUDA® 12)
Public IP address: Auto assign static IP
In User data, check Enable custom cloud-init config
Paste the whole cloud-init from here. Replace the PASTE_YOUR_SSH_PUBLIC_KEY_HERE with your actual ssh key. You can have multiple ssh keys for multiple people, then each key should be a separate line
	 - ssh-ed25519 PASTE_YOUR_SSH_PUBLIC_KEY_HERE
Click Create VM and wait around 10 minutes
Connect from your IDE (Cursor, VS Code, …)
Find the IP address of your VM: VM overview -> Network -> Public IPv4 (without the /32)
In VS code, or cursor
View -> Command Palette -> Remote-SSH: Open SSH Configuration File
Paste the following and save:
Host nebius_hackathon
  HostName YOUR_VM_IP_ADDRESS
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
Command Palette -> Remote-SSH: Connect to host -> nebius_hackathon
Open folder -> workspace
You should see two folders: mjlab and video2robot
Follow Readmes of the corresponding repositories
- video2robot: https://github.com/DavidDobas/video2robot-hackathon
- mjlab: https://github.com/DavidDobas/mjlab-hackathon

ALWAYS STOP YOUR INSTANCE AFTER FINISH SOMETHING!
IT’S EXPENSIVE
Before using video2robot
Register at https://smpl-x.is.tue.mpg.de/register.php and https://smpl.is.tue.mpg.de/register.php
Detailed instructions in the README: https://github.com/DavidDobas/video2robot-hackathon
Before using mjlab
Register at https://wandb.ai/
Create a Registry called Motions
In User settings, create an API key and use to log-in
Detailed instructions in the README: https://github.com/DavidDobas/mjlab-hackathon

