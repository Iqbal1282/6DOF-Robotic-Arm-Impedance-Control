# robot/jacobian.py
import numpy as np
from robot.kinematics import forward_kinematics


def compute_jacobian(dh_params, q):
    """
    Compute 6x6 geometric Jacobian
    """

    T_0e, T_list = forward_kinematics(dh_params, q)

    o_n = T_0e[0:3, 3]
    J = np.zeros((6, 6))

    o_prev = np.zeros(3)
    z_prev = np.array([0, 0, 1])

    for i in range(6):
        if i > 0:
            o_prev = T_list[i - 1][0:3, 3]
            z_prev = T_list[i - 1][0:3, 2]

        Jv = np.cross(z_prev, o_n - o_prev)
        Jw = z_prev

        J[0:3, i] = Jv
        J[3:6, i] = Jw

    return J
