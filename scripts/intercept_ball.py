"""
Fixed-Time MPC Interceptor for a 3-DOF PRR arm.
Uses trajectory estimation and receding horizon control to intercept a thrown ball.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
import sys

try:
    import casadi as ca
except ImportError:
    print("CasADi is required.  pip install casadi")
    sys.exit(1)


class ArmParams:
    l1          = 1.0
    l2          = 1.0
    base_z_off  = 0.06

    start_pos   = [0.0, 0.0, 0.1]
    start_ori   = p.getQuaternionFromEuler([math.pi / 2, 0.0, 0.0])

    q_min       = np.array([-2.0,   0.05,  -3.09])
    q_max       = np.array([ 2.0,   2.00,  -0.05])

    dq_max      = np.array([ 2.0,   5.0,    5.0])
    ddq_max     = np.array([ 4.0,  10.0,   10.0])
    torque_lim  = np.array([500.0, 200.0,  200.0])

    N           = 20

    home        = np.array([0.0, 1.0, -2.0])


def fk_np(q: np.ndarray, par: ArmParams) -> np.ndarray:
    """Compute end-effector position in world XZ."""
    s, q1, q2 = q[0], q[1], q[2]
    x = s + par.l1 * np.cos(q1) + par.l2 * np.cos(q1 + q2)
    z = par.base_z_off + par.l1 * np.sin(q1) + par.l2 * np.sin(q1 + q2)
    return np.array([x, z])


class TrajectoryEstimator:
    """Collects observations and fits ballistic trajectory using least squares."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.ts = []
        self.xs = []
        self.ys = []
        self.zs = []

    def add(self, t: float, pos: np.ndarray):
        """Record one observation with simulated sensor noise."""
        noise = np.random.normal(0, 0.005, size=3)
        pos   = np.asarray(pos) + noise
        self.ts.append(t)
        self.xs.append(pos[0])
        self.ys.append(pos[1])
        self.zs.append(pos[2])

    def estimate(self):
        """Fit trajectory and return state and crossing time."""
        if len(self.ts) < 5:
            return None, None

        T = np.array(self.ts)
        X = np.array(self.xs)
        Y = np.array(self.ys)
        Z = np.array(self.zs)

        # Fit X linear
        Ax = np.vstack([np.ones_like(T), T]).T
        px, *_ = np.linalg.lstsq(Ax, X, rcond=None)
        x0, vx = px

        # Fit Y linear
        Ay = np.vstack([np.ones_like(T), T]).T
        py, *_ = np.linalg.lstsq(Ay, Y, rcond=None)
        y0, vy = py

        # Fit Z quadratic
        Az = np.vstack([np.ones_like(T), T, T**2]).T
        pz, *_ = np.linalg.lstsq(Az, Z, rcond=None)
        z0, vz, acc_half = pz

        state6 = np.array([x0, y0, z0, vx, vy, vz])

        # Time when ball crosses y=0
        if abs(vy) < 1e-6:
            return state6, None

        t_cross = -y0 / vy
        if t_cross < 0:
            return state6, None

        return state6, t_cross


def ball_xz_at(state6: np.ndarray, t: float, g: float = 9.81) -> np.ndarray:
    """Predict ball position at time t."""
    x  = state6[0] + state6[3] * t
    z  = state6[2] + state6[5] * t - 0.5 * g * t**2
    return np.array([x, z])


class FixedTimeMPC:
    """MPC solver for fixed-time interception using CasADi/IPOPT."""

    def __init__(self, par: ArmParams):
        self.par      = par
        self.last_X   = None
        self.last_U   = None
        self._build()

    def _build(self):
        """Construct the CasADi optimization problem."""
        opti = ca.Opti()
        N    = self.par.N

        # Decision variables
        X = opti.variable(6, N + 1)
        U = opti.variable(3, N)

        pos = X[:3, :]
        vel = X[3:, :]

        # Parameters set at solve time
        P_x0     = opti.parameter(6)
        P_target = opti.parameter(2)
        P_dt     = opti.parameter()

        # Cost function with weighted control effort
        w_slider   = 0.01
        w_shoulder = 1.0
        w_elbow    = 1.0
        W = ca.diag(ca.vertcat(w_slider, w_shoulder, w_elbow))

        J = 0
        for k in range(N):
            J += ca.mtimes([U[:, k].T, W, U[:, k]])

        # Terminal velocity penalty
        J += 10.0 * ca.sumsqr(vel[:, -1])

        opti.minimize(J)

        # Dynamics constraints using Euler integration
        for k in range(N):
            opti.subject_to(pos[:, k+1] == pos[:, k] + vel[:, k] * P_dt
                            + 0.5 * U[:, k] * P_dt**2)
            opti.subject_to(vel[:, k+1] == vel[:, k] + U[:, k] * P_dt)

        # Initial state constraint
        opti.subject_to(X[:, 0] == P_x0)

        # Box constraints on position, velocity, and acceleration
        for i in range(3):
            opti.subject_to(opti.bounded(
                self.par.q_min[i], pos[i, :], self.par.q_max[i]))
            opti.subject_to(opti.bounded(
                -self.par.dq_max[i], vel[i, :], self.par.dq_max[i]))
            opti.subject_to(opti.bounded(
                -self.par.ddq_max[i], U[i, :], self.par.ddq_max[i]))

        # Terminal constraint: FK must match ball target
        q_f = pos[:, -1]
        l1, l2 = self.par.l1, self.par.l2

        ee_x = q_f[0] + l1 * ca.cos(q_f[1]) + l2 * ca.cos(q_f[1] + q_f[2])
        ee_z = self.par.base_z_off + l1 * ca.sin(q_f[1]) + l2 * ca.sin(q_f[1] + q_f[2])

        opti.subject_to(ee_x == P_target[0])
        opti.subject_to(ee_z == P_target[1])

        # Solver options
        opts = {
            "ipopt.print_level":   0,
            "ipopt.sb":            "yes",
            "print_time":          0,
            "ipopt.max_iter":      300,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        self.opti     = opti
        self.X        = X
        self.U        = U
        self.P_x0     = P_x0
        self.P_target = P_target
        self.P_dt     = P_dt

    def solve(self, robot_state: np.ndarray,
              ball_target_xz: np.ndarray,
              T_remain: float):
        """Solve MPC and return trajectory."""
        N  = self.par.N
        dt = T_remain / N

        self.opti.set_value(self.P_x0,     robot_state)
        self.opti.set_value(self.P_target,  ball_target_xz)
        self.opti.set_value(self.P_dt,      dt)

        # Warm start from previous solution
        if self.last_X is not None:
            self.opti.set_initial(self.X, self.last_X)
            self.opti.set_initial(self.U, self.last_U)
        else:
            X0 = np.zeros((6, N + 1))
            for k in range(N + 1):
                X0[:, k] = robot_state
            self.opti.set_initial(self.X, X0)
            self.opti.set_initial(self.U, np.zeros((3, N)))

        try:
            sol     = self.opti.solve()
            q_traj  = sol.value(self.X)[:3, :]
            v_traj  = sol.value(self.X)[3:, :]
            u_traj  = sol.value(self.U)

            self.last_X = sol.value(self.X)
            self.last_U = sol.value(self.U)

            return True, q_traj, v_traj, u_traj

        except Exception as e:
            return False, None, None, None

    def reset_warmstart(self):
        self.last_X = None
        self.last_U = None


def interp_traj(t_local: float, T_plan: float, q_traj, v_traj, N: int):
    """Interpolate trajectory between collocation nodes."""
    alpha = np.clip(t_local / max(T_plan, 1e-6), 0.0, 1.0)
    idx_f = alpha * N
    k     = int(np.clip(idx_f, 0, N - 1))
    frac  = idx_f - k

    q_des  = (1 - frac) * q_traj[:, k]  + frac * q_traj[:, k + 1]
    dq_des = (1 - frac) * v_traj[:, k]  + frac * v_traj[:, k + 1]
    return q_des, dq_des


def main():
    # Setup simulation
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    plane_id = p.loadURDF("plane.urdf")

    par = ArmParams()

    # Locate URDF
    _here     = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(_here, "urdf", "3dof_planar_slider.urdf")
    if not os.path.exists(urdf_path):
        urdf_path = os.path.abspath("urdf/3dof_planar_slider.urdf")
    if not os.path.exists(urdf_path):
        raise FileNotFoundError("Cannot find 3dof_planar_slider.urdf")

    robot_id = p.loadURDF(urdf_path,
                          basePosition=par.start_pos,
                          baseOrientation=par.start_ori,
                          useFixedBase=True)

    # Find actuated joint indices
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        jtype = p.getJointInfo(robot_id, j)[2]
        if jtype in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
            joint_indices.append(j)
    assert len(joint_indices) == 3, f"Expected 3 actuated joints, got {len(joint_indices)}"

    # Initialize to home position
    for i, idx in enumerate(joint_indices):
        p.resetJointState(robot_id, idx, par.home[i])

    estimator = TrajectoryEstimator()
    mpc       = FixedTimeMPC(par)

    # Visual plane marking y=0
    vis_plane = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[3, 0.002, 3],
                                    rgbaColor=[0, 1, 0, 0.15])
    p.createMultiBody(0, -1, vis_plane, [0, 0, 1.0])

    p.resetDebugVisualizerCamera(cameraDistance=4.0,
                                  cameraYaw=0,
                                  cameraPitch=-15,
                                  cameraTargetPosition=[0, 0, 1])

    ball_id = -1
    SIM_DT  = 1.0 / 240.0

    print("=" * 60)
    print("  FIXED-TIME MPC INTERCEPTOR")
    print("  Press [SPACE] to throw a ball.  [Q] to quit.")
    print("=" * 60)

    running = True
    while running:

        # Wait for spacebar input
        while True:
            keys = p.getKeyboardEvents()
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                break
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                running = False
                break
            p.stepSimulation()
            time.sleep(SIM_DT)
        if not running:
            break

        # Return arm to home position
        print("\n-- Resetting to home --")
        for _ in range(240):
            for i, idx in enumerate(joint_indices):
                p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL,
                                        targetPosition=par.home[i],
                                        force=par.torque_lim[i],
                                        maxVelocity=par.dq_max[i] * 0.5)
            p.stepSimulation()
            time.sleep(SIM_DT)
        mpc.reset_warmstart()

        # Spawn ball with random position and velocity
        if ball_id >= 0:
            p.removeBody(ball_id)

        bx0 = np.random.uniform(3.5, 5.0)
        by0 = np.random.uniform(2.0, 4.0)
        bz0 = np.random.uniform(1.0, 2.0)

        bvx = np.random.uniform(-5.0, -3.0)
        bvy = np.random.uniform(-4.0, -2.0)
        bvz = np.random.uniform(3.0, 6.0)

        vis_ball = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                                       rgbaColor=[1, 0.2, 0, 1])
        col_ball = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        ball_id  = p.createMultiBody(0.2, col_ball, vis_ball,
                                     [bx0, by0, bz0])
        p.resetBaseVelocity(ball_id, [bvx, bvy, bvz])

        print(f"Ball spawned at ({bx0:.2f}, {by0:.2f}, {bz0:.2f})")
        print(f"  velocity     ({bvx:.2f}, {bvy:.2f}, {bvz:.2f})")

        # Observation phase to collect trajectory data
        estimator.reset()
        obs_duration  = 0.15
        obs_steps     = int(obs_duration / SIM_DT)
        obs_t0        = 0.0

        for step in range(obs_steps):
            p.stepSimulation()
            pos3, _ = p.getBasePositionAndOrientation(ball_id)
            t_obs   = step * SIM_DT
            estimator.add(t_obs, pos3)
            time.sleep(SIM_DT)

        t_elapsed_obs = obs_steps * SIM_DT

        # Estimate trajectory and crossing time
        state6, t_cross_from_obs = estimator.estimate()
        if state6 is None or t_cross_from_obs is None:
            print("Estimation failed or ball not approaching Y=0. Skipping.")
            continue

        T_remain = t_cross_from_obs - t_elapsed_obs
        if T_remain < 0.15:
            print(f"Ball crosses Y=0 too soon (T_remain={T_remain:.3f}s). Skipping.")
            continue

        ball_target_xz = ball_xz_at(state6, t_cross_from_obs)
        print(f"  T_cross     = {t_cross_from_obs:.3f} s  (from obs t=0)")
        print(f"  T_remain    = {T_remain:.3f} s")
        print(f"  Target EE   = ({ball_target_xz[0]:.3f}, {ball_target_xz[1]:.3f})")

        # Reachability check
        max_reach = par.l1 + par.l2
        target_z_rel = ball_target_xz[1] - par.base_z_off
        if ball_target_xz[1] < 0.0:
            print("Target below ground. Skipping.")
            continue

        # Receding horizon control loop
        print("-- MPC tracking started --")

        mpc_interval   = 0.05
        failsafe_thres = 0.10

        current_q_traj = None
        current_v_traj = None
        plan_T         = None
        plan_wall_t    = None

        rhc_t0         = time.time()
        next_mpc_wall  = rhc_t0
        solver_frozen  = False
        solve_count    = 0

        while True:
            wall_now   = time.time()
            t_since    = wall_now - rhc_t0
            T_remain_now = T_remain - t_since

            if T_remain_now < -0.3:
                break

            # Re-solve MPC periodically
            if (not solver_frozen) and (wall_now >= next_mpc_wall):
                if T_remain_now < failsafe_thres:
                    solver_frozen = True
                    print(f"  Failsafe: T_remain={T_remain_now:.3f}s -- freezing solver")
                else:
                    q_now  = np.array([p.getJointState(robot_id, j)[0]
                                       for j in joint_indices])
                    dq_now = np.array([p.getJointState(robot_id, j)[1]
                                       for j in joint_indices])
                    robot_state = np.concatenate([q_now, dq_now])

                    # Use live ball state for better accuracy
                    bpos, _ = p.getBasePositionAndOrientation(ball_id)
                    bvel, _ = p.getBaseVelocity(ball_id)
                    bpos = np.array(bpos)
                    bvel = np.array(bvel)

                    if abs(bvel[1]) > 1e-6:
                        t_cross_live = -bpos[1] / bvel[1]
                    else:
                        t_cross_live = T_remain_now

                    if t_cross_live < failsafe_thres:
                        solver_frozen = True
                        print(f"  Failsafe (live): t_cross_live={t_cross_live:.3f}s")
                    else:
                        live_target_x = bpos[0] + bvel[0] * t_cross_live
                        live_target_z = bpos[2] + bvel[2] * t_cross_live \
                                        - 0.5 * 9.81 * t_cross_live**2
                        live_target   = np.array([live_target_x, live_target_z])

                        ok, qt, vt, ut = mpc.solve(robot_state, live_target,
                                                    t_cross_live)
                        solve_count += 1
                        if ok:
                            current_q_traj = qt
                            current_v_traj = vt
                            plan_T         = t_cross_live
                            plan_wall_t    = wall_now
                        else:
                            if solve_count == 1:
                                print("  First MPC solve failed -- target may be "
                                      "out of workspace.")

                    next_mpc_wall = wall_now + mpc_interval

            # Interpolate trajectory and send motor commands
            if current_q_traj is not None and plan_wall_t is not None:
                t_local = wall_now - plan_wall_t
                q_des, dq_des = interp_traj(t_local, plan_T,
                                            current_q_traj, current_v_traj,
                                            par.N)
                for i, idx in enumerate(joint_indices):
                    p.setJointMotorControl2(
                        robot_id, idx, p.POSITION_CONTROL,
                        targetPosition=float(q_des[i]),
                        targetVelocity=float(dq_des[i]),
                        force=par.torque_lim[i],
                        maxVelocity=par.dq_max[i])

            p.stepSimulation()
            time.sleep(SIM_DT)

        # Report results
        q_final = np.array([p.getJointState(robot_id, j)[0]
                            for j in joint_indices])
        ee_final  = fk_np(q_final, par)
        ball_pos3, _ = p.getBasePositionAndOrientation(ball_id)
        ball_xz_now  = np.array([ball_pos3[0], ball_pos3[2]])

        miss = np.linalg.norm(ee_final - ball_target_xz)
        print(f"  End-effector : ({ee_final[0]:.3f}, {ee_final[1]:.3f})")
        print(f"  Ball target  : ({ball_target_xz[0]:.3f}, {ball_target_xz[1]:.3f})")
        print(f"  Miss distance: {miss*1000:.1f} mm")
        print(f"  Solver calls : {solve_count}")
        print("-- attempt complete -- press [SPACE] for next throw, [Q] to quit --\n")

        # Let physics run so user can see result
        for _ in range(480):
            p.stepSimulation()
            time.sleep(SIM_DT)

    p.disconnect()
    print("Simulation ended.")


if __name__ == "__main__":
    main()
