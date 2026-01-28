# robot/dynamics.py
import numpy as np


def mass_matrix(q):
    """
    Simplified inertia matrix
    (can later be replaced by Pinocchio or full RBD)
    """
    n = len(q)
    M = np.eye(n)

    for i in range(n):
        M[i, i] = 1.0 + 0.2 * np.cos(q[i])

    return M


def coriolis_matrix(q, qdot):
    """
    Simplified Coriolis matrix
    """
    n = len(q)
    C = np.zeros((n, n))

    for i in range(n):
        C[i, i] = 0.1 * qdot[i]

    return C


def gravity_vector(q):
    """
    Simplified gravity torque
    """
    g = np.zeros(len(q))

    for i in range(len(q)):
        g[i] = 0.5 * np.sin(q[i])

    return g
