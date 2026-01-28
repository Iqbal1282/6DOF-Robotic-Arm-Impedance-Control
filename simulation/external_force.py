# simulation/external_force.py
import numpy as np


def external_force(t, q):
    """
    Simulated external interaction force
    """

    F = np.zeros(3)

    # apply force after 2 seconds
    if t > 2.0:
        F[0] = 5.0 * np.sin(2 * np.pi * 0.5 * t)
        F[1] = 2.0
        F[2] = 0.0

    return F
