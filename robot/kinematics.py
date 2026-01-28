# robot/kinematics.py
import numpy as np


def dh_transform(a, alpha, d, theta):
    """Standard DH transformation matrix"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])


def forward_kinematics(dh_params, q):
    """
    Compute forward kinematics

    Args:
        dh_params: (6x4) DH table
        q: joint angles (6,)

    Returns:
        T_0e: homogeneous transform of end-effector
        T_list: list of transforms for each joint
    """
    T = np.eye(4)
    T_list = []

    for i in range(6):
        a, alpha, d, _ = dh_params[i]
        theta = q[i]
        A = dh_transform(a, alpha, d, theta)
        T = T @ A
        T_list.append(T)

    return T, T_list
