import numpy as np
from robot.kinematics import forward_kinematics
from robot.model import RobotModel


robot = RobotModel()

def external_force(t, q, k_env=800.0, d_env=20.0, surface_z=0.30):
    """
    Environment (tissue) contact model

    Parameters
    ----------
    t : float
        simulation time
    q : ndarray (6,)
        joint angles
    k_env : float
        environment stiffness (tissue stiffness)
    d_env : float
        environment damping
    surface_z : float
        contact surface height

    Returns
    -------
    F_ext : ndarray (3,)
        external force applied at end-effector
    """

    # Forward kinematics
    dh = robot.get_dh()
    T, _ = forward_kinematics(dh, q)
    z = T[2, 3]   # end-effector height

    # End-effector velocity approx (only z needed)
    # (simple finite difference approximation)
    penetration = surface_z - z

    Fz = 0.0

    # Contact occurs only when penetrating tissue surface
    if penetration > 0:
        Fz = k_env * penetration  # spring force

    F_ext = np.array([0.0, 0.0, Fz])

    return F_ext
