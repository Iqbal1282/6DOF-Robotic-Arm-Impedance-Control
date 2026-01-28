# Six-DOF Robot Reinforcement Learning for Impedance Control
---
**Author**: Md. Iqbal Hossain
**Project:** RL-guided Impedance Control for Safe Medical Robotics
**Environment:** Python 3.9+, PyTorch, Stable-Baselines3
---

## Overview

This project implements a **6-DOF robotic arm simulation** with **impedance/admittance controllers** and integrates **Reinforcement Learning (PPO)** for **adaptive impedance control**.

The RL agent learns to **modulate impedance gains safely**, allowing the robot to:

* Track desired end-effector trajectories
* Maintain safe contact forces with the environment
* Adapt to variable tissue stiffness in medical robotics scenarios

All robot dynamics, kinematics, and controllers are simulated in Python, providing a **safe platform for RL training and testing**.

---

## Features

* **Robot Simulation**

  * Full 6-DOF robot model
  * Forward and inverse kinematics
  * Dynamics: mass, Coriolis, gravity
  * PD, computed torque, impedance, and admittance controllers
* **Reinforcement Learning**

  * Proximal Policy Optimization (PPO) agent
  * RL-guided adaptation of impedance gains
  * Safe torque clipping and observation normalization
* **Visualization**

  * 3D robot animation using Matplotlib
  * Live tracking of trajectories and end-effector motion
* **Gym-compatible environment**

  * Compatible with Stable-Baselines3
  * Can be extended to other RL algorithms (SAC, TD3)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/iqbal1282/6DOF-Robotic-Arm-Impedance-Control
.git
cd 6DOF-Robotic-Arm-Impedance-Control

```

2. Create and activate a Python environment (recommended):

```bash
conda create -n robot_rl python=3.9
conda activate robot_rl
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies include:**

* `numpy`
* `matplotlib`
* `gym`
* `stable-baselines3`
* `torch`

---

## Project Structure

```
6dof-rl-robot/
├── robot/
│   ├── model.py          # 6-DOF robot definition
│   ├── kinematics.py     # Forward/inverse kinematics
│   ├── jacobian.py       # Jacobian computations
│   └── dynamics.py       # Mass, Coriolis, gravity
├── controllers/
│   ├── joint_pd.py       # Joint PD controller
│   ├── computed_torque.py
│   ├── impedance.py
│   └── admittance.py
├── simulation/
│   ├── ode_system.py
│   ├── external_force.py # Simulated tissue/environment force
│   └── simulate.py
├── visualization/
│   ├── trajectory_plot.py
│   └── robot_animation.py
├── gym_robot/
│   └── envs/
│       └── robot_env.py  # RL Gym environment
├── rl_main.py            # Training and testing script
├── requirements.txt
└── README.md
```

---

## Usage

### 1. Training

Train PPO on the simulated robot:

```bash
python rl_main.py
```

* Checkpoints are saved every 50k steps in `checkpoints/`
* Training duration: ~200,000 steps (adjustable in `rl_main.py`)
* Observation: `[q, qdot, position_error, external_force]`
* Action: scaling for impedance gains `[ΔK, ΔD]`

---

### 2. Testing / Visualization

After training, the script automatically runs **test episodes** with **3D visualization**:

* End-effector trajectory
* Robot joint motion
* Interaction forces

You can adjust:

```python
TEST_EPISODES = 5
visualize = True
```

in `rl_main.py`.

---

### 3. Customization

* **Controllers:** switch between PD, impedance, or admittance
* **RL algorithm:** replace PPO with SAC/TD3 for continuous control
* **Environment:** modify `external_force()` for different tissue models or variable stiffness

---

## Citation / Usage in Research

If you use this code for research or publication, please cite:

```
Md. Iqbal Hossain, RL-Guided Impedance Control for Safe Medical Robotics, 2026.
```

---

## Future Work

* Sim-to-real transfer to real 6-DOF robot arms
* Multi-modal sensory input (force, ultrasound, imaging)
* Hypernetwork-based impedance adaptation
* Safety verification for clinical applications