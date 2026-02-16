#!/usr/bin/env python3
"""
intercept_ball.py — Fixed-Time Receding Horizon MPC for 3-DOF PRR Arm Ball Interception
========================================================================================

A 3-DOF planar robot (Prismatic–Revolute–Revolute) intercepts a 3D ballistic
projectile as it crosses the robot's XZ operational plane (Y = 0).

Key design choices vs. the old "time-optimal" formulation:
  • The intercept time T* is NOT an optimisation variable.
    It is computed analytically from the ball's Y-axis trajectory:
        ball_y(t) = y0 + vy*t  →  T_cross = -y0 / vy
  • The MPC minimises control effort and penalises heavy revolute usage,
    encouraging the prismatic slider to absorb gross X-motion.
  • The horizon shrinks every RHC cycle (dt = T_remain / N) and the solver
    is warm-started with the previous solution.
  • A failsafe freezes the solver when T_remain < 0.1 s and plays the last
    plan open-loop.

Author : Copilot (refactored from project codebase)
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

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class ArmParams:
    """Physical constants and limits for the PRR arm."""
    l1          = 1.0       # link-1 length  (m)
    l2          = 1.0       # link-2 length  (m)
    base_z_off  = 0.06      # shoulder height above rail  (m)

    # PyBullet spawn pose (pi/2 about X rotates URDF XY → world XZ)
    start_pos   = [0.0, 0.0, 0.1]
    start_ori   = p.getQuaternionFromEuler([math.pi / 2, 0.0, 0.0])

    # Joint limits  [slider, shoulder, elbow]
    q_min       = np.array([-2.0,   0.05,  -3.09])
    q_max       = np.array([ 2.0,   2.00,  -0.05])

    # Velocity & acceleration caps
    dq_max      = np.array([ 2.0,   5.0,    5.0])
    ddq_max     = np.array([ 4.0,  10.0,   10.0])
    torque_lim  = np.array([500.0, 200.0,  200.0])

    # MPC discretisation nodes
    N           = 20

    # Home configuration  (slider centred, arm "up-and-folded")
    home        = np.array([0.0, 1.0, -2.0])


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FORWARD KINEMATICS  (NumPy — for readout / logging only)
# ═══════════════════════════════════════════════════════════════════════════════
def fk_np(q: np.ndarray, par: ArmParams) -> np.ndarray:
    """
    End-effector position in world XZ.
        x = slider + l1·cos(q1) + l2·cos(q1+q2)
        z = base_z  + l1·sin(q1) + l2·sin(q1+q2)
    Returns shape (2,).
    """
    s, q1, q2 = q[0], q[1], q[2]
    x = s + par.l1 * np.cos(q1) + par.l2 * np.cos(q1 + q2)
    z = par.base_z_off + par.l1 * np.sin(q1) + par.l2 * np.sin(q1 + q2)
    return np.array([x, z])


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TRAJECTORY ESTIMATOR  (Least-Squares on noisy observations)
# ═══════════════════════════════════════════════════════════════════════════════
class TrajectoryEstimator:
    """
    Collects (t, pos3D) samples and fits:
        x(t) = x0 + vx·t              (linear)
        y(t) = y0 + vy·t              (linear — no gravity sideways)
        z(t) = z0 + vz·t + ½·a·t²     (quadratic — gravity in z)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.ts = []
        self.xs = []
        self.ys = []
        self.zs = []

    def add(self, t: float, pos: np.ndarray):
        """Record one (possibly noisy) observation."""
        noise = np.random.normal(0, 0.005, size=3)
        pos   = np.asarray(pos) + noise
        self.ts.append(t)
        self.xs.append(pos[0])
        self.ys.append(pos[1])
        self.zs.append(pos[2])

    def estimate(self):
        """
        Returns (state6, t_cross) or (None, None) if not enough data.
        state6 = [x0, y0, z0, vx, vy, vz] at observation t=0.
        t_cross = time (relative to t=0 of observations) when ball_y = 0.
        """
        if len(self.ts) < 5:
            return None, None

        T = np.array(self.ts)
        X = np.array(self.xs)
        Y = np.array(self.ys)
        Z = np.array(self.zs)

        # --- Fit X (linear) ---
        Ax = np.vstack([np.ones_like(T), T]).T
        px, *_ = np.linalg.lstsq(Ax, X, rcond=None)
        x0, vx = px

        # --- Fit Y (linear) ---
        Ay = np.vstack([np.ones_like(T), T]).T
        py, *_ = np.linalg.lstsq(Ay, Y, rcond=None)
        y0, vy = py

        # --- Fit Z (quadratic) ---
        Az = np.vstack([np.ones_like(T), T, T**2]).T
        pz, *_ = np.linalg.lstsq(Az, Z, rcond=None)
        z0, vz, acc_half = pz            # acc_half ≈ -g/2

        state6 = np.array([x0, y0, z0, vx, vy, vz])

        # Time when y(t) = 0  →  y0 + vy·t = 0  →  t = -y0/vy
        if abs(vy) < 1e-6:
            # Ball is not approaching  → no valid crossing
            return state6, None

        t_cross = -y0 / vy
        if t_cross < 0:
            # Crossing is in the past → already missed
            return state6, None

        return state6, t_cross


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  BALL STATE PREDICTOR  (used inside & outside CasADi)
# ═══════════════════════════════════════════════════════════════════════════════
def ball_xz_at(state6: np.ndarray, t: float, g: float = 9.81) -> np.ndarray:
    """Predict ball [x, z] at time t from observation-frame state6."""
    x  = state6[0] + state6[3] * t
    z  = state6[2] + state6[5] * t - 0.5 * g * t**2
    return np.array([x, z])


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FIXED-TIME MPC  (CasADi / IPOPT)
# ═══════════════════════════════════════════════════════════════════════════════
class FixedTimeMPC:
    """
    At every RHC call we receive:
        • robot_state  (6,)  = [q0, q1, q2, dq0, dq1, dq2]
        • ball_xz_target (2,) = desired end-effector [x, z] at horizon end
        • T_remain  (float)   = seconds until the ball crosses Y=0

    The MPC minimises  ‖U‖²  (weighted) subject to:
        – double-integrator dynamics  (Euler)
        – joint position / velocity / acceleration limits
        – FK(q_final) == ball_xz_target   (terminal equality)

    The horizon length is FIXED to T_remain (not optimised).
    """

    def __init__(self, par: ArmParams):
        self.par      = par
        self.last_X   = None          # warm-start cache
        self.last_U   = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        """Construct the CasADi Opti problem (called once)."""
        opti = ca.Opti()
        N    = self.par.N

        # ---- decision variables ----
        X = opti.variable(6, N + 1)       # state trajectory  [q; dq]
        U = opti.variable(3, N)           # control (joint accelerations)

        pos = X[:3, :]
        vel = X[3:, :]

        # ---- parameters (set every solve call) ----
        P_x0     = opti.parameter(6)      # current robot state
        P_target = opti.parameter(2)      # desired EE [x, z] at final node
        P_dt     = opti.parameter()       # dt = T_remain / N

        # ---- cost ----
        #   • Heavy weight on revolute accelerations → prefer slider motion
        #   • Small weight on slider acceleration    → allow gross X motion
        w_slider   = 0.01
        w_shoulder = 1.0
        w_elbow    = 1.0
        W = ca.diag(ca.vertcat(w_slider, w_shoulder, w_elbow))

        J = 0
        for k in range(N):
            J += ca.mtimes([U[:, k].T, W, U[:, k]])

        # Small terminal-velocity penalty (arrive smoothly)
        J += 10.0 * ca.sumsqr(vel[:, -1])

        opti.minimize(J)

        # ---- dynamics (Euler integration) ----
        for k in range(N):
            opti.subject_to(pos[:, k+1] == pos[:, k] + vel[:, k] * P_dt
                            + 0.5 * U[:, k] * P_dt**2)
            opti.subject_to(vel[:, k+1] == vel[:, k] + U[:, k] * P_dt)

        # ---- initial-state constraint ----
        opti.subject_to(X[:, 0] == P_x0)

        # ---- box constraints (applied at every node) ----
        for i in range(3):
            opti.subject_to(opti.bounded(
                self.par.q_min[i], pos[i, :], self.par.q_max[i]))
            opti.subject_to(opti.bounded(
                -self.par.dq_max[i], vel[i, :], self.par.dq_max[i]))
            opti.subject_to(opti.bounded(
                -self.par.ddq_max[i], U[i, :], self.par.ddq_max[i]))

        # ---- terminal FK == ball target ----
        q_f = pos[:, -1]
        l1, l2 = self.par.l1, self.par.l2

        ee_x = q_f[0] + l1 * ca.cos(q_f[1]) + l2 * ca.cos(q_f[1] + q_f[2])
        ee_z = self.par.base_z_off + l1 * ca.sin(q_f[1]) + l2 * ca.sin(q_f[1] + q_f[2])

        opti.subject_to(ee_x == P_target[0])
        opti.subject_to(ee_z == P_target[1])

        # ---- solver options ----
        opts = {
            "ipopt.print_level":   0,
            "ipopt.sb":            "yes",
            "print_time":          0,
            "ipopt.max_iter":      300,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        # store handles
        self.opti     = opti
        self.X        = X
        self.U        = U
        self.P_x0     = P_x0
        self.P_target = P_target
        self.P_dt     = P_dt

    # ------------------------------------------------------------------
    def solve(self, robot_state: np.ndarray,
              ball_target_xz: np.ndarray,
              T_remain: float):
        """
        Solve the fixed-time MPC.

        Returns
        -------
        success : bool
        q_traj  : ndarray (3, N+1)   joint positions at collocation nodes
        v_traj  : ndarray (3, N+1)   joint velocities
        u_traj  : ndarray (3, N)     accelerations (controls)
        """
        N  = self.par.N
        dt = T_remain / N

        # Set parameters
        self.opti.set_value(self.P_x0,     robot_state)
        self.opti.set_value(self.P_target,  ball_target_xz)
        self.opti.set_value(self.P_dt,      dt)

        # ---- warm start ----
        if self.last_X is not None:
            self.opti.set_initial(self.X, self.last_X)
            self.opti.set_initial(self.U, self.last_U)
        else:
            # Cold start: hold current config, zero velocity / accel
            X0 = np.zeros((6, N + 1))
            for k in range(N + 1):
                X0[:, k] = robot_state
            self.opti.set_initial(self.X, X0)
            self.opti.set_initial(self.U, np.zeros((3, N)))

        # ---- solve ----
        try:
            sol     = self.opti.solve()
            q_traj  = sol.value(self.X)[:3, :]
            v_traj  = sol.value(self.X)[3:, :]
            u_traj  = sol.value(self.U)

            # Cache for warm start
            self.last_X = sol.value(self.X)
            self.last_U = sol.value(self.U)

            return True, q_traj, v_traj, u_traj

        except Exception as e:
            # Solver failed — do NOT wipe the cache (might recover next cycle)
            return False, None, None, None

    def reset_warmstart(self):
        self.last_X = None
        self.last_U = None


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  UTILITY:  Smooth trajectory interpolation between collocation nodes
# ═══════════════════════════════════════════════════════════════════════════════
def interp_traj(t_local: float, T_plan: float, q_traj, v_traj, N: int):
    """
    Given elapsed time within the current plan, return (q_des, dq_des)
    by linearly interpolating between collocation nodes.
    """
    alpha = np.clip(t_local / max(T_plan, 1e-6), 0.0, 1.0)
    idx_f = alpha * N
    k     = int(np.clip(idx_f, 0, N - 1))
    frac  = idx_f - k

    q_des  = (1 - frac) * q_traj[:, k]  + frac * q_traj[:, k + 1]
    dq_des = (1 - frac) * v_traj[:, k]  + frac * v_traj[:, k + 1]
    return q_des, dq_des


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ------------------------------------------------------------------ setup
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    plane_id = p.loadURDF("plane.urdf")

    par = ArmParams()

    # Locate URDF relative to this script (intercept_ball.py lives at repo root)
    _here     = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(_here, "urdf", "3dof_planar_slider.urdf")
    if not os.path.exists(urdf_path):
        # Fallback: try cwd-relative
        urdf_path = os.path.abspath("urdf/3dof_planar_slider.urdf")
    if not os.path.exists(urdf_path):
        raise FileNotFoundError("Cannot find 3dof_planar_slider.urdf")

    robot_id = p.loadURDF(urdf_path,
                          basePosition=par.start_pos,
                          baseOrientation=par.start_ori,
                          useFixedBase=True)

    # Discover actuated joint indices (prismatic + revolute)
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        jtype = p.getJointInfo(robot_id, j)[2]
        if jtype in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
            joint_indices.append(j)
    assert len(joint_indices) == 3, f"Expected 3 actuated joints, got {len(joint_indices)}"

    # Send arm to home configuration
    for i, idx in enumerate(joint_indices):
        p.resetJointState(robot_id, idx, par.home[i])

    # Instantiate modules
    estimator = TrajectoryEstimator()
    mpc       = FixedTimeMPC(par)

    # Visual: a thin green plane to mark Y = 0
    vis_plane = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[3, 0.002, 3],
                                    rgbaColor=[0, 1, 0, 0.15])
    p.createMultiBody(0, -1, vis_plane, [0, 0, 1.0])

    # Camera: nice side-view of the XZ plane
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

    # ================================================================ outer loop
    running = True
    while running:

        # ---------- idle: step physics, wait for spacebar ----------
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

        # ---------- 1. smooth return to home ----------
        print("\n── Resetting to home ──")
        for _ in range(240):                      # 1 second
            for i, idx in enumerate(joint_indices):
                p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL,
                                        targetPosition=par.home[i],
                                        force=par.torque_lim[i],
                                        maxVelocity=par.dq_max[i] * 0.5)
            p.stepSimulation()
            time.sleep(SIM_DT)
        mpc.reset_warmstart()

        # ---------- 2. spawn ball with 3D velocity ----------
        if ball_id >= 0:
            p.removeBody(ball_id)

        # Random start position (far away, off-plane)
        bx0 = np.random.uniform(3.5, 5.0)
        by0 = np.random.uniform(2.0, 4.0)        # OFF the XZ plane
        bz0 = np.random.uniform(1.0, 2.0)

        # Velocities: towards robot (−x, −y), lofted (+z)
        bvx = np.random.uniform(-5.0, -3.0)
        bvy = np.random.uniform(-4.0, -2.0)      # must be negative to cross y=0
        bvz = np.random.uniform(3.0, 6.0)

        vis_ball = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                                       rgbaColor=[1, 0.2, 0, 1])
        col_ball = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        ball_id  = p.createMultiBody(0.2, col_ball, vis_ball,
                                     [bx0, by0, bz0])
        p.resetBaseVelocity(ball_id, [bvx, bvy, bvz])

        print(f"Ball spawned at ({bx0:.2f}, {by0:.2f}, {bz0:.2f})")
        print(f"  velocity     ({bvx:.2f}, {bvy:.2f}, {bvz:.2f})")

        # ---------- 3. observation phase (collect ~0.15 s of data) ----------
        estimator.reset()
        obs_duration  = 0.15                       # seconds of observation
        obs_steps     = int(obs_duration / SIM_DT)
        obs_t0        = 0.0                        # logical clock starts here

        for step in range(obs_steps):
            p.stepSimulation()
            pos3, _ = p.getBasePositionAndOrientation(ball_id)
            t_obs   = step * SIM_DT
            estimator.add(t_obs, pos3)
            time.sleep(SIM_DT)

        # Time already elapsed in the simulation since the ball was thrown
        t_elapsed_obs = obs_steps * SIM_DT

        # ---------- 4. estimate trajectory & compute T_cross ----------
        state6, t_cross_from_obs = estimator.estimate()
        if state6 is None or t_cross_from_obs is None:
            print("✗ Estimation failed or ball not approaching Y=0. Skipping.")
            continue

        # T_remain is how much time is left AFTER the observation window
        T_remain = t_cross_from_obs - t_elapsed_obs
        if T_remain < 0.15:
            print(f"✗ Ball crosses Y=0 too soon (T_remain={T_remain:.3f}s). Skipping.")
            continue

        # Predicted ball XZ at the crossing instant
        ball_target_xz = ball_xz_at(state6, t_cross_from_obs)
        print(f"  T_cross     = {t_cross_from_obs:.3f} s  (from obs t=0)")
        print(f"  T_remain    = {T_remain:.3f} s")
        print(f"  Target EE   = ({ball_target_xz[0]:.3f}, {ball_target_xz[1]:.3f})")

        # Quick reachability sanity check
        max_reach = par.l1 + par.l2
        target_z_rel = ball_target_xz[1] - par.base_z_off
        if ball_target_xz[1] < 0.0:
            print("✗ Target below ground. Skipping.")
            continue
        # The slider can move ±2 m, so effective X reach is [-2 + 0, 2 + 2] = [-2, 4].
        # No hard skip here — let the solver decide feasibility.

        # ---------- 5. Receding Horizon Control loop ----------
        print("── MPC tracking started ──")

        mpc_interval   = 0.05                      # re-solve every 50 ms (20 Hz)
        failsafe_thres = 0.10                      # stop re-solving below this

        # Plan storage
        current_q_traj = None
        current_v_traj = None
        plan_T         = None                      # T_remain when plan was made
        plan_wall_t    = None                      # wall-clock of last plan

        # Timing
        rhc_t0         = time.time()               # wall-clock reference
        next_mpc_wall  = rhc_t0                    # first solve immediately
        solver_frozen  = False
        solve_count    = 0

        while True:
            wall_now   = time.time()
            t_since    = wall_now - rhc_t0         # wall time since RHC started
            T_remain_now = T_remain - t_since      # remaining to crossing

            # ---- exit when horizon exhausted ----
            if T_remain_now < -0.3:
                break

            # ---- MPC re-solve (if horizon still large enough) ----
            if (not solver_frozen) and (wall_now >= next_mpc_wall):
                if T_remain_now < failsafe_thres:
                    solver_frozen = True
                    print(f"  ⚠ Failsafe: T_remain={T_remain_now:.3f}s — freezing solver")
                else:
                    # Gather live robot state
                    q_now  = np.array([p.getJointState(robot_id, j)[0]
                                       for j in joint_indices])
                    dq_now = np.array([p.getJointState(robot_id, j)[1]
                                       for j in joint_indices])
                    robot_state = np.concatenate([q_now, dq_now])

                    # Re-predict ball target using LIVE ball state for robustness
                    bpos, _ = p.getBasePositionAndOrientation(ball_id)
                    bvel, _ = p.getBaseVelocity(ball_id)
                    bpos = np.array(bpos)
                    bvel = np.array(bvel)

                    # Recompute crossing time from live state
                    if abs(bvel[1]) > 1e-6:
                        t_cross_live = -bpos[1] / bvel[1]
                    else:
                        t_cross_live = T_remain_now  # fallback

                    if t_cross_live < failsafe_thres:
                        solver_frozen = True
                        print(f"  ⚠ Failsafe (live): t_cross_live={t_cross_live:.3f}s")
                    else:
                        # Ball XZ at live-crossing
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
                                print("  ✗ First MPC solve failed — target may be "
                                      "out of workspace.")

                    next_mpc_wall = wall_now + mpc_interval

            # ---- interpolate plan & send motor commands ----
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

            # ---- step physics ----
            p.stepSimulation()
            time.sleep(SIM_DT)

        # ---------- 6. post-attempt diagnostics ----------
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
        print("── attempt complete ── press [SPACE] for next throw, [Q] to quit ──\n")

        # Let physics keep running so user can see the result
        for _ in range(480):        # 2 seconds of coast
            p.stepSimulation()
            time.sleep(SIM_DT)

    p.disconnect()
    print("Simulation ended.")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
