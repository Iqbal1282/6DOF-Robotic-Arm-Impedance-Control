# main.py
import numpy as np
from scipy.integrate import solve_ivp

from robot.model import RobotModel
from simulation.ode_system import robot_ode
from simulation.external_force import external_force
from visualization.trajectory_plot import (
    plot_joint_trajectories,
    plot_end_effector_trajectory
)
from visualization.robot_animation import animate_robot

from controllers.joint_pd import JointPDController
from controllers.computed_torque import ComputedTorqueController
from controllers.impedance import ImpedanceController


# --------------------------------------------------
# Desired trajectory definition
# --------------------------------------------------
def trajectory(t):
    """
    Desired joint + task-space trajectory
    """

    # joint-space reference
    qd = 0.5 * np.sin(0.5 * t) * np.ones(6)
    qdot_d = 0.25 * np.cos(0.5 * t) * np.ones(6)
    qddot_d = -0.125 * np.sin(0.5 * t) * np.ones(6)

    # task-space reference (for impedance)
    xd = np.array([0.45, 0.15, 0.30])
    xd_dot = np.zeros(3)

    return qd, qdot_d, qddot_d, xd, xd_dot


# --------------------------------------------------
# Main experiment
# --------------------------------------------------
def main():

    # ------------------------
    # Robot
    # ------------------------
    robot = RobotModel()

    # ------------------------
    # Controller selection
    # ------------------------
    controller_type = "impedance"
    # controller_type = "pd"
    # controller_type = "computed_torque"

    if controller_type == "pd":
        controller = JointPDController(
            Kp=120 * np.ones(6),
            Kd=25 * np.ones(6)
        )

    elif controller_type == "computed_torque":
        controller = ComputedTorqueController(
            Kp=60 * np.ones(6),
            Kd=20 * np.ones(6)
        )

    elif controller_type == "impedance":
        controller = ImpedanceController(
            Md=[2.0, 2.0, 2.0],
            Dd=[35.0, 35.0, 35.0],
            Kd=[120.0, 120.0, 120.0]
        )

    else:
        raise ValueError("Invalid controller type")

    # ------------------------
    # Initial state
    # ------------------------
    q0 = np.zeros(6)
    qdot0 = np.zeros(6)
    state0 = np.concatenate([q0, qdot0])

    # ------------------------
    # Simulation setup
    # ------------------------
    t_span = (0.0, 10.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1200)

    print(f"Running simulation with {controller_type.upper()} controller...")

    sol = solve_ivp(
        fun=robot_ode,
        t_span=t_span,
        y0=state0,
        t_eval=t_eval,
        args=(
            controller,
            controller_type,
            robot,
            trajectory,
            external_force
        ),
        rtol=1e-6,
        atol=1e-8
    )

    # ------------------------
    # Extract results
    # ------------------------
    q_history = sol.y[:6]
    t = sol.t

    print("Simulation finished.")

    # ------------------------
    # Visualization
    # ------------------------
    plot_joint_trajectories(t, q_history)
    plot_end_effector_trajectory(robot.get_dh(), q_history)
    animate_robot(robot.get_dh(), q_history)


# --------------------------------------------------
if __name__ == "__main__":
    main()
