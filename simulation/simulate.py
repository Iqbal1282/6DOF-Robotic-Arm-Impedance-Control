# simulation/simulate.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from robot.model import RobotModel
from controllers.joint_pd import JointPDController
from controllers.computed_torque import ComputedTorqueController
from controllers.impedance import ImpedanceController
from simulation.external_force import external_force
from simulation.ode_system import robot_ode


# -------------------------
# Desired trajectory
# -------------------------
def trajectory(t):
    qd = 0.5 * np.sin(0.5 * t) * np.ones(6)
    qdot_d = 0.25 * np.cos(0.5 * t) * np.ones(6)
    qddot_d = -0.125 * np.sin(0.5 * t) * np.ones(6)

    xd = np.array([0.4, 0.2, 0.3])
    xd_dot = np.zeros(3)

    return qd, qdot_d, qddot_d, xd, xd_dot


def main():

    robot = RobotModel()

    # choose controller
    controller_type = "impedance"
    # controller_type = "pd"
    # controller_type = "computed_torque"

    if controller_type == "pd":
        controller = JointPDController(
            Kp=100 * np.ones(6),
            Kd=20 * np.ones(6)
        )

    elif controller_type == "computed_torque":
        controller = ComputedTorqueController(
            Kp=50 * np.ones(6),
            Kd=15 * np.ones(6)
        )

    elif controller_type == "impedance":
        controller = ImpedanceController(
            Md=[2, 2, 2],
            Dd=[30, 30, 30],
            Kd=[100, 100, 100]
        )

    # initial state
    q0 = np.zeros(6)
    qdot0 = np.zeros(6)
    state0 = np.concatenate([q0, qdot0])

    # simulate
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)

    sol = solve_ivp(
        robot_ode,
        t_span,
        state0,
        t_eval=t_eval,
        args=(controller, controller_type, robot, trajectory, external_force)
    )

    # plot
    plt.figure()
    for i in range(6):
        plt.plot(sol.t, sol.y[i], label=f"q{i+1}")

    plt.xlabel("Time (s)")
    plt.ylabel("Joint angle (rad)")
    plt.title(f"6-DOF Robot – {controller_type} control")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
