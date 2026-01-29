import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_comparison(
    traj_no_rl,
    traj_rl,
    start,
    target,
    surface_z=0.30
):
    """
    Compare fixed impedance vs RL-guided impedance

    Parameters
    ----------
    traj_no_rl : (N,3)
    traj_rl : (N,3)
    start : (3,)
    target : (3,)
    """

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # ---- tissue surface ----
    xx, yy = np.meshgrid(
        np.linspace(-0.2, 0.7, 10),
        np.linspace(-0.3, 0.5, 10)
    )
    zz = surface_z * np.ones_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.15)

    # ---- trajectories ----
    ax.plot(
        traj_no_rl[:, 0],
        traj_no_rl[:, 1],
        traj_no_rl[:, 2],
        "--",
        color="gray",
        linewidth=2,
        label="Fixed Impedance (No RL)"
    )

    ax.plot(
        traj_rl[:, 0],
        traj_rl[:, 1],
        traj_rl[:, 2],
        color="blue",
        linewidth=3,
        label="RL-Guided Impedance"
    )

    # ---- points ----
    ax.scatter(*start, color="green", s=80, label="Start")
    ax.scatter(*target, color="red", s=80, marker="x", label="Target")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("RL-Guided Impedance Control vs Fixed Impedance")

    ax.legend()
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.show()
