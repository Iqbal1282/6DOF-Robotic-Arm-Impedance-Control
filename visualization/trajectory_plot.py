# visualization/trajectory_plot.py
import numpy as np
import matplotlib.pyplot as plt
from robot.kinematics import forward_kinematics


def plot_joint_trajectories(t, q_history):
    """
    Plot joint angles vs time
    """
    plt.figure(figsize=(10, 5))

    for i in range(q_history.shape[0]):
        plt.plot(t, q_history[i], label=f"q{i+1}")

    plt.xlabel("Time (s)")
    plt.ylabel("Joint angle (rad)")
    plt.title("Joint Trajectories")
    plt.legend()
    plt.grid()
    plt.show()


def plot_end_effector_trajectory(dh_params, q_history):
    """
    Plot end-effector 3D trajectory
    """
    x, y, z = [], [], []

    for k in range(q_history.shape[1]):
        q = q_history[:, k]
        T, _ = forward_kinematics(dh_params, q)
        pos = T[0:3, 3]
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector Trajectory")
    ax.grid(True)

    plt.show()
