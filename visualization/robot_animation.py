# visualization/robot_animation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from robot.kinematics import forward_kinematics


def animate_robot(dh_params, q_history, interval=30):
    """
    Animate 6-DOF robot motion
    """

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1.2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("6-DOF Robot Animation")

    line, = ax.plot([], [], [], "o-", lw=3)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        q = q_history[:, frame]
        _, T_list = forward_kinematics(dh_params, q)

        xs = [0]
        ys = [0]
        zs = [0]

        for T in T_list:
            pos = T[0:3, 3]
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])

        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        return line,

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=q_history.shape[1],
        init_func=init,
        interval=interval,
        blit=True
    )

    plt.show()
