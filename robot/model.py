# robot/model.py
import numpy as np


class RobotModel:
    """
    6-DOF serial manipulator (UR5-like)
    """

    def __init__(self):
        # DH parameters: [a, alpha, d, theta]
        # theta is variable (joint angle)

        self.dh_params = np.array([
            [0.0,      np.pi/2,  0.0892,  0.0],
            [-0.425,   0.0,      0.0,     0.0],
            [-0.392,   0.0,      0.0,     0.0],
            [0.0,      np.pi/2,  0.1093,  0.0],
            [0.0,     -np.pi/2,  0.09475, 0.0],
            [0.0,      0.0,      0.0825,  0.0]
        ])

        self.n_joints = 6
        self.gravity = np.array([0, 0, -9.81])

    def get_dh(self):
        return self.dh_params.copy()
