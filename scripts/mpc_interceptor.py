"""
Time-optimal MPC interceptor for 3-DOF planar arm with warm-start re-planning.
Uses CasADi optimization with real-time trajectory updates during execution.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
import sys
import matplotlib.pyplot as plt

try:
    import casadi as ca
except ImportError:
    print("Error: CasADi not found. Run 'pip install casadi'")
    sys.exit(1)


class ArmParameters:
    """Physical parameters and limits for the 3-DOF arm."""
    l1: float = 1.0  
    l2: float = 1.0  
    base_z_offset: float = 0.06
    start_pos = np.array([0, 0, 0.1])
    start_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    
    q_min = np.array([-2.0, 0.05, -3.09])
    q_max = np.array([2.0, 1.60, -0.05])
    
    dq_max = np.array([2.0, 5.0, 5.0])
    ddq_max = np.array([4.0, 10.0, 10.0])
    torque_limits = np.array([500.0, 200.0, 200.0])
    N = 20


class TrajectoryEstimator:
    """Estimates ball trajectory from noisy observations using least squares."""
    
    def __init__(self):
        self.history_t = []
        self.history_x = []
        self.history_z = []
        
    def reset(self):
        self.history_t = []
        self.history_x = []
        self.history_z = []
        
    def add_observation(self, t, pos):
        noise = np.random.normal(0, 0.005, size=3)
        pos = pos + noise
        self.history_t.append(t)
        self.history_x.append(pos[0])
        self.history_z.append(pos[2])

    def estimate_state(self):
        """Returns estimated ball state [x, y, z, vx, vy, vz]."""
        if len(self.history_t) < 5: 
            return None
        
        T = np.array(self.history_t)
        X = np.array(self.history_x)
        Z = np.array(self.history_z)
        
        A_x = np.vstack([np.ones(len(T)), T]).T
        params_x, _, _, _ = np.linalg.lstsq(A_x, X, rcond=None)
        x0_est, vx0_est = params_x
        
        A_z = np.vstack([np.ones(len(T)), T, T**2]).T
        params_z, _, _, _ = np.linalg.lstsq(A_z, Z, rcond=None)
        z0_est, vz0_est, acc_term = params_z
        
        return np.array([x0_est, 0, z0_est, vx0_est, 0, vz0_est])


class TimeOptimalMPC:
    """MPC solver with warm-starting for real-time re-planning."""
    
    def __init__(self, params: ArmParameters):
        self.params = params
        self.opti = None
        self.last_X = None
        self.last_U = None
        self.build_solver()

    def build_solver(self):
        """Constructs the CasADi optimization problem."""
        self.opti = ca.Opti()
        
        self.T = self.opti.variable()
        self.X = self.opti.variable(6, self.params.N + 1)
        pos = self.X[:3, :]
        vel = self.X[3:, :]
        self.U = self.opti.variable(3, self.params.N)
        
        self.P_robot_init = self.opti.parameter(6)
        self.P_ball_init = self.opti.parameter(6)
        
        J = self.T * 10.0 + ca.sumsqr(self.U) * 0.001
        self.opti.minimize(J)
        
        self.opti.subject_to(self.T >= 0.1)
        self.opti.subject_to(self.T <= 2.0)
        
        dt = self.T / self.params.N
        
        # Dynamics constraints
        for k in range(self.params.N):
            self.opti.subject_to(pos[:, k+1] == pos[:, k] + vel[:, k]*dt + 0.5*self.U[:, k]*dt**2)
            self.opti.subject_to(vel[:, k+1] == vel[:, k] + self.U[:, k]*dt)
        
        self.opti.subject_to(self.X[:, 0] == self.P_robot_init)

        # Joint and velocity limits
        for i in range(3):
            self.opti.subject_to(self.opti.bounded(self.params.q_min[i], pos[i, :], self.params.q_max[i]))
            self.opti.subject_to(self.opti.bounded(-self.params.dq_max[i], vel[i, :], self.params.dq_max[i]))
            self.opti.subject_to(self.opti.bounded(-self.params.ddq_max[i], self.U[i, :], self.params.ddq_max[i]))

        # Terminal constraint: end-effector matches ball position
        q_final = pos[:, -1]
        l1, l2 = self.params.l1, self.params.l2
        
        rx = q_final[0] + l1*ca.cos(q_final[1]) + l2*ca.cos(q_final[1] + q_final[2])
        rz = self.params.base_z_offset + l1*ca.sin(q_final[1]) + l2*ca.sin(q_final[1] + q_final[2])
        
        bx0, bz0 = self.P_ball_init[0], self.P_ball_init[2]
        bvx, bvz = self.P_ball_init[3], self.P_ball_init[5]
        g = 9.81
        
        bx_T = bx0 + bvx * self.T
        bz_T = bz0 + bvz * self.T - 0.5 * g * self.T**2
        
        self.opti.subject_to(rx == bx_T)
        self.opti.subject_to(rz == bz_T)
        
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)

    def solve(self, robot_state, ball_state, T_guess=0.8):
        """Solves MPC with warm-start from previous solution."""
        self.opti.set_value(self.P_robot_init, robot_state)
        self.opti.set_value(self.P_ball_init, ball_state)
        self.opti.set_initial(self.T, T_guess)
        
        # Warm-start from previous solution if available
        if self.last_X is not None:
            self.opti.set_initial(self.X, self.last_X)
            self.opti.set_initial(self.U, self.last_U)
        else:
            X_guess = np.zeros((6, self.params.N + 1))
            for k in range(self.params.N + 1):
                X_guess[:3, k] = robot_state[:3]
            self.opti.set_initial(self.X, X_guess)
            self.opti.set_initial(self.U, np.zeros((3, self.params.N)))
        
        try:
            sol = self.opti.solve()
            T_opt = sol.value(self.T)
            X_sol = sol.value(self.X)
            pos_traj = X_sol[:3, :]
            vel_traj = X_sol[3:, :]
            acc_traj = sol.value(self.U)
            
            self.last_X = X_sol
            self.last_U = acc_traj
            return T_opt, pos_traj, vel_traj, acc_traj
            
        except Exception:
            self.last_X = None
            self.last_U = None
            return None, None, None, None


def get_ee_pos(q, params):
    """Computes end-effector position from joint angles."""
    slider_pos, q1, q2 = q
    l1, l2 = params.l1, params.l2
    x_rel = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    z_rel = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([slider_pos + x_rel, z_rel + params.base_z_offset])


def save_throw_plots(out_dir, throw_idx, logs):
    """Saves per-throw performance plots for MPC optimization variables and interception error."""
    os.makedirs(out_dir, exist_ok=True)

    t = np.array(logs["time"])
    if t.size == 0:
        return

    q = np.array(logs["q"])
    dq = np.array(logs["dq"])
    u = np.array(logs["u"])
    err = np.array(logs["error"])
    mpc_t = np.array(logs["mpc_times"])
    mpc_T = np.array(logs["mpc_T"])

    # Graph 1: Optimization variable X (state trajectory, actual execution)
    fig1, ax1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1[0].plot(t, q[:, 0], label='q_slider [m]')
    ax1[0].plot(t, q[:, 1], label='q_shoulder [rad]')
    ax1[0].plot(t, q[:, 2], label='q_elbow [rad]')
    ax1[0].set_ylabel('Joint Position')
    ax1[0].set_title(f'Throw {throw_idx}: Optimization Variable X (Positions)')
    ax1[0].grid(True)
    ax1[0].legend()

    ax1[1].plot(t, dq[:, 0], label='dq_slider [m/s]')
    ax1[1].plot(t, dq[:, 1], label='dq_shoulder [rad/s]')
    ax1[1].plot(t, dq[:, 2], label='dq_elbow [rad/s]')
    ax1[1].set_xlabel('Time [s]')
    ax1[1].set_ylabel('Joint Velocity')
    ax1[1].set_title('Optimization Variable X (Velocities)')
    ax1[1].grid(True)
    ax1[1].legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, f'throw_{throw_idx:03d}_X_state.png'), dpi=150)
    plt.close(fig1)

    # Graph 2: Optimization variable U (applied commanded accelerations)
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.plot(t, u[:, 0], label='u_slider [m/s²]')
    ax2.plot(t, u[:, 1], label='u_shoulder [rad/s²]')
    ax2.plot(t, u[:, 2], label='u_elbow [rad/s²]')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Joint Acceleration Command')
    ax2.set_title(f'Throw {throw_idx}: Optimization Variable U')
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f'throw_{throw_idx:03d}_U_control.png'), dpi=150)
    plt.close(fig2)

    # Graph 3: Optimization variable T (optimized horizon over re-solves)
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    if mpc_t.size > 0:
        ax3.plot(mpc_t, mpc_T, marker='o', label='Optimized T')
    ax3.set_xlabel('Simulation Time [s]')
    ax3.set_ylabel('Optimized Interception Time T [s]')
    ax3.set_title(f'Throw {throw_idx}: Optimization Variable T Across MPC Re-solves')
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, f'throw_{throw_idx:03d}_T_horizon.png'), dpi=150)
    plt.close(fig3)

    # Graph 4: End-effector to ball distance
    fig4, ax4 = plt.subplots(figsize=(10, 4.5))
    ax4.plot(t, err * 1000.0, color='crimson', label='Miss Distance')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Distance [mm]')
    ax4.set_title(f'Throw {throw_idx}: End-Effector Distance to Ball')
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, f'throw_{throw_idx:03d}_distance.png'), dpi=150)
    plt.close(fig4)

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    params = ArmParameters()
    estimator = TrajectoryEstimator()
    mpc = TimeOptimalMPC(params)
    
    urdf_filename = "3dof_planar_slider.urdf"
    urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/', urdf_filename))
    if not os.path.exists(urdf_path):
        urdf_path = os.path.abspath(os.path.join(os.getcwd(), urdf_filename))
    if not os.path.exists(urdf_path):
        urdf_path = urdf_filename 

    robot_id = p.loadURDF(urdf_path, params.start_pos, params.start_orientation, useFixedBase=True)
    
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, j)[2] in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
            joint_indices.append(j)
    
    home_pos = [0.0, 1.0, -2.0]
    
    for i, idx in enumerate(joint_indices):
        p.resetJointState(robot_id, idx, home_pos[i])

    print("========================================")
    print("   SIMULATION STARTED")
    print("========================================")
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docs/mpc_throw_plots'))
    throw_counter = 0
    
    ball_id = -1
    
    while True:
        throw_counter += 1
        print("\nPhysics running... Press [SPACE] to throw the ball.")
        
        # Non-blocking wait loop
        waiting_for_throw = True
        while waiting_for_throw:
            keys = p.getKeyboardEvents()
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                waiting_for_throw = False
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("Resetting to home position smoothly...")
        for _ in range(120): 
            for i, idx in enumerate(joint_indices):
                p.setJointMotorControl2(
                    robot_id, idx, p.POSITION_CONTROL, 
                    targetPosition=home_pos[i], 
                    force=params.torque_limits[i],
                    maxVelocity=params.dq_max[i] * 0.5 
                )
            p.stepSimulation()
            time.sleep(1./240.)
            
        mpc.last_X = None
        mpc.last_U = None
        
        log_time = []
        log_error = []
        log_q = []
        log_dq = []
        log_u = []
        log_mpc_times = []
        log_mpc_T = []
        
        if ball_id >= 0: p.removeBody(ball_id)
        start_x = np.random.uniform(3.5, 4.5)
        start_z = np.random.uniform(1.0, 1.5)
        vel_x = np.random.uniform(-4.5, -3.5) 
        vel_z = np.random.uniform(3.0, 5.0)   
        
        vis_ball = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1,0,0,1])
        col_ball = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        ball_id = p.createMultiBody(0.2, col_ball, vis_ball, [start_x, 0, start_z])
        p.resetBaseVelocity(ball_id, [vel_x, 0, vel_z])
        
        print(f"Ball Thrown! Init Vel: [{vel_x:.2f}, {vel_z:.2f}]")
        
        estimator.reset()
        est_start = time.time()
        while time.time() - est_start < 0.1:
            p.stepSimulation()
            pos, _ = p.getBasePositionAndOrientation(ball_id)
            estimator.add_observation(time.time() - est_start, pos)
            time.sleep(1./240.)
            
        ball_state_est = estimator.estimate_state()
        if ball_state_est is None:
            print("Estimation Failed.")
            continue
            
        q_curr = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        dq_curr = [p.getJointState(robot_id, j)[1] for j in joint_indices]
        robot_state = np.array(q_curr + dq_curr)
        
        T_opt, q_traj, v_traj, u_traj = mpc.solve(robot_state, ball_state_est)
        
        if T_opt is None:
            print("MPC Failed: Target Unreachable. Nudging elbow joint...")
            curr_q2 = p.getJointState(robot_id, joint_indices[2])[0]
            nudge_target = curr_q2 - 0.5 if curr_q2 > -1.5 else curr_q2 + 0.5
            
            for _ in range(120):
                p.setJointMotorControl2(
                    robot_id, joint_indices[2], p.POSITION_CONTROL,
                    targetPosition=nudge_target,
                    force=params.torque_limits[2]
                )
                p.stepSimulation()
                time.sleep(1./240.)
            continue
            
        print(f"Interception possible in {T_opt:.3f} seconds.")
        
        sim_dt = 1.0 / 240.0
        mpc_interval = 0.05 
        next_mpc_time = 0.0
        t_now = 0.0
        
        T_remain = T_opt 
        current_q_traj = q_traj
        current_v_traj = v_traj
        current_u_traj = u_traj
        last_mpc_time = 0.0
        log_mpc_times.append(0.0)
        log_mpc_T.append(T_opt)
        
        while T_remain > 0.01:
            if t_now >= next_mpc_time and T_remain > 0.15:
                q_curr = [p.getJointState(robot_id, j)[0] for j in joint_indices]
                dq_curr = [p.getJointState(robot_id, j)[1] for j in joint_indices]
                robot_state = np.array(q_curr + dq_curr)
                
                b_pos, _ = p.getBasePositionAndOrientation(ball_id)
                b_vel, _ = p.getBaseVelocity(ball_id)
                ball_state_live = np.array([b_pos[0], 0, b_pos[2], b_vel[0], 0, b_vel[2]])
                
                new_T, new_q, new_v, new_u = mpc.solve(robot_state, ball_state_live, T_guess=T_remain)
                
                if new_T is not None:
                    T_remain = new_T
                    current_q_traj = new_q
                    current_v_traj = new_v
                    current_u_traj = new_u
                    last_mpc_time = t_now
                    log_mpc_times.append(t_now)
                    log_mpc_T.append(new_T)
                
                next_mpc_time += mpc_interval

            if current_q_traj is not None:
                t_plan = t_now - last_mpc_time
                idx_float = (t_plan / max(T_remain, 0.001)) * params.N
                idx = int(np.clip(idx_float, 0, params.N - 1))
                
                target_q = current_q_traj[:, idx+1] if idx+1 <= params.N else current_q_traj[:, -1]
                target_v = current_v_traj[:, idx+1] if idx+1 <= params.N else [0, 0, 0]
                target_u = current_u_traj[:, idx] if current_u_traj is not None else np.zeros(3)
                
                for i, j_id in enumerate(joint_indices):
                    p.setJointMotorControl2(
                        robot_id, j_id, p.POSITION_CONTROL,
                        targetPosition=target_q[i],
                        targetVelocity=target_v[i],
                        force=params.torque_limits[i],
                        maxVelocity=params.dq_max[i]
                    )

            p.stepSimulation()
            
            curr_q = [p.getJointState(robot_id, j)[0] for j in joint_indices]
            curr_dq = [p.getJointState(robot_id, j)[1] for j in joint_indices]
            ee_pos = get_ee_pos(curr_q, params)
            ball_pos, _ = p.getBasePositionAndOrientation(ball_id)
            
            log_time.append(t_now)
            log_error.append(np.linalg.norm(ee_pos - np.array([ball_pos[0], ball_pos[2]])))
            log_q.append(curr_q)
            log_dq.append(curr_dq)
            log_u.append(target_u if current_q_traj is not None else np.zeros(3))

            time.sleep(sim_dt)
            t_now += sim_dt
            T_remain -= sim_dt

        print("Interception Attempt Finished.")
        
        if len(log_error) > 0:
            min_error = np.min(log_error)
            print(f"Minimum Miss Distance: {min_error*1000:.2f} mm")

        save_throw_plots(
            plots_dir,
            throw_counter,
            {
                "time": log_time,
                "q": log_q,
                "dq": log_dq,
                "u": log_u,
                "error": log_error,
                "mpc_times": log_mpc_times,
                "mpc_T": log_mpc_T,
            },
        )
        print(f"Saved performance graphs to: {plots_dir}")

if __name__ == "__main__":
    main()