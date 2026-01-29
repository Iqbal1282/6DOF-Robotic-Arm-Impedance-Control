import os
import numpy as np
from stable_baselines3 import PPO

from gym_robot.envs.robot_env import RobotImpedanceEnv
from visualization.trajectory_comparison import plot_trajectory_comparison
from visualization.impedance_analysis import plot_impedance_analysis


# ------------------------------
# PARAMETERS
# ------------------------------
MODEL_PATH = "ppo_robot_impedance.zip"
DT = 0.01
STEPS = 500


# ------------------------------
# ROLLOUTS
# ------------------------------
def rollout_no_rl(env, steps=STEPS):
    traj = []
    z, f, kz = [], [], []

    obs = env.reset()
    for _ in range(steps):
        action = np.zeros(6)
        obs, _, done, _ = env.step(action)

        pos = env.get_end_effector_position()
        traj.append(pos)

        z.append(pos[2])
        f.append(env.current_force_z)
        kz.append(env.current_kz)

        if done:
            break

    return np.array(traj), np.array(z), np.array(f), np.array(kz)


def rollout_rl(env, model, steps=STEPS):
    traj = []
    z, f, kz = [], [], []

    obs = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        pos = env.get_end_effector_position()
        traj.append(pos)

        z.append(pos[2])
        f.append(env.current_force_z)
        kz.append(env.current_kz)

        if done:
            break

    return np.array(traj), np.array(z), np.array(f), np.array(kz)


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":

    env = RobotImpedanceEnv(dt=DT)
    model = PPO.load(MODEL_PATH)

    # --------------------------
    # Rollouts
    # --------------------------
    traj_no_rl, z_no_rl, f_no_rl, kz_no_rl = rollout_no_rl(env)
    traj_rl, z_rl, f_rl, kz_rl = rollout_rl(env, model)

    # --------------------------
    # Figure 1: Trajectories
    # --------------------------
    plot_trajectory_comparison(
        traj_no_rl,
        traj_rl,
        start=traj_no_rl[0],
        target=env.xd
    )

    # --------------------------
    # Figure 2: Physics behavior
    # --------------------------
    plot_impedance_analysis(
        z_no_rl=z_no_rl,
        z_rl=z_rl,
        f_no_rl=f_no_rl,
        f_rl=f_rl,
        kz_no_rl=kz_no_rl,
        kz_rl=kz_rl,
        dt=DT
    )

    # --------------------------
    # Print summary
    # --------------------------
    print("Start position:", traj_no_rl[0])
    print("Target position:", env.xd)
    print("RL final position:", traj_rl[-1])
    print("No-RL final position:", traj_no_rl[-1])
