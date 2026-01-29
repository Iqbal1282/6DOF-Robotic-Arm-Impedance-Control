import numpy as np
import matplotlib.pyplot as plt


def plot_impedance_analysis(
    z_no_rl,
    z_rl,
    f_no_rl,
    f_rl,
    kz_no_rl,
    kz_rl,
    dt=0.01
):
    t = np.arange(len(z_no_rl)) * dt

    plt.figure(figsize=(15, 4))

    # -----------------------------
    # 1. Z POSITION
    # -----------------------------
    plt.subplot(1, 3, 1)
    plt.plot(t, z_no_rl, 'r--', label="No RL")
    plt.plot(t, z_rl, 'b', label="RL-guided")
    plt.xlabel("Time (s)")
    plt.ylabel("z position (m)")
    plt.title("End-effector z(t)")
    plt.legend()
    plt.grid()

    # -----------------------------
    # 2. CONTACT FORCE
    # -----------------------------
    plt.subplot(1, 3, 2)
    plt.plot(t, f_no_rl, 'r--', label="No RL")
    plt.plot(t, f_rl, 'b', label="RL-guided")
    plt.xlabel("Time (s)")
    plt.ylabel("Contact force (N)")
    plt.title("Contact Force Fz(t)")
    plt.legend()
    plt.grid()

    # -----------------------------
    # 3. STIFFNESS
    # -----------------------------
    plt.subplot(1, 3, 3)
    plt.plot(t, kz_no_rl, 'r--', label="No RL")
    plt.plot(t, kz_rl, 'b', label="RL-guided")
    plt.xlabel("Time (s)")
    plt.ylabel("Stiffness Kz (N/m)")
    plt.title("Impedance Adaptation Kz(t)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
