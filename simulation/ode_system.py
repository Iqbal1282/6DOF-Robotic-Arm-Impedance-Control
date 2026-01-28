# simulation/ode_system.py
import numpy as np
from robot.dynamics import mass_matrix, coriolis_matrix, gravity_vector


def robot_ode(t, state, controller, controller_type, robot, traj_func, force_func):
    """
    State = [q, qdot]

    controller_type:
        "pd"
        "computed_torque"
        "impedance"
    """

    n = robot.n_joints
    q = state[:n]
    qdot = state[n:]

    # desired trajectory
    qd, qdot_d, qddot_d, xd, xd_dot = traj_func(t)

    # external force
    F_ext = force_func(t, q)

    # controller selection
    if controller_type == "pd":
        tau = controller.compute(q, qdot, qd, qdot_d)

    elif controller_type == "computed_torque":
        tau = controller.compute(q, qdot, qd, qdot_d, qddot_d)

    elif controller_type == "impedance":
        tau = controller.compute(
            robot.get_dh(),
            q,
            qdot,
            xd,
            xd_dot,
            F_ext
        )

    else:
        raise ValueError("Unknown controller type")

    # dynamics
    M = mass_matrix(q)
    C = coriolis_matrix(q, qdot)
    g = gravity_vector(q)

    qddot = np.linalg.inv(M) @ (tau - C @ qdot - g)

    return np.concatenate([qdot, qddot])
