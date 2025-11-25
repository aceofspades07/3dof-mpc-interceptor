import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
import sys
import matplotlib.pyplot as plt

# Try importing CasADi. If missing, warn user
try:
    import casadi as ca
except ImportError:
    print("Error: CasADi not found. Please run 'pip install casadi'")
    sys.exit(1)

# Configuration
class ArmParameters:
    # Robot Dimensions
    l1: float = 1.0  
    l2: float = 1.0  
    base_z_offset: float = 0.06
    start_pos = np.array([0, 0, 0.1])
    start_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    rail_min = -2.0
    rail_max = 2.0
    dq_max = np.array([2.0, 5.0, 5.0])    # Velocity Limits [Slider, Shldr, Elb]
    ddq_max = np.array([4.0, 10.0, 10.0]) # Acceleration Limits
    torque_limits = np.array([500.0, 200.0, 200.0])
    
    N = 20  # Number of discretization steps

# Estimation
class TrajectoryEstimator:
    def __init__(self):
        self.history_t = []
        self.history_x = []
        self.history_z = []
        
    def reset(self):
        self.history_t = []
        self.history_x = []
        self.history_z = []
        
    def add_observation(self, t, pos):
        # Add artificial noise 
        noise = np.random.normal(0, 0.005, size=3) # 5mm noise
        pos = pos + noise
        
        self.history_t.append(t)
        self.history_x.append(pos[0])
        self.history_z.append(pos[2]) # World Z is Up

    def estimate_state(self):
        # Performs least squares regression to find projectile parameters.
        
        if len(self.history_t) < 5: 
            return None
        
        T = np.array(self.history_t)
        X = np.array(self.history_x)
        Z = np.array(self.history_z)
        
        # Linear regression
        A_x = np.vstack([np.ones(len(T)), T]).T
        params_x, _, _, _ = np.linalg.lstsq(A_x, X, rcond=None)
        x0_est, vx0_est = params_x
        
        # Quadratic regression
        A_z = np.vstack([np.ones(len(T)), T, T**2]).T
        params_z, _, _, _ = np.linalg.lstsq(A_z, Z, rcond=None)
        z0_est, vz0_est, acc_term = params_z
        # Derived Gravity
        g_est = -2 * acc_term
        
        # Return estimates
        return np.array([x0_est, 0, z0_est, vx0_est, 0, vz0_est])

# MPC Solver
class TimeOptimalMPC:
    def __init__(self, params: ArmParameters):
        self.params = params
        self.opti = None
        self.build_solver()

    def build_solver(self):
        self.opti = ca.Opti()
        
        # Total Time to Catch
        self.T = self.opti.variable()
        
        # States [q0, q1, q2, v0, v1, v2] , size (6, N+1)
        self.X = self.opti.variable(6, self.params.N + 1)
        pos = self.X[:3, :]
        vel = self.X[3:, :]
        
        # Controls Accelerations [u0, u1, u2] , size (3, N)
        self.U = self.opti.variable(3, self.params.N)
        
        # Parameters
        self.P_robot_init = self.opti.parameter(6) # Current robot state
        self.P_ball_init = self.opti.parameter(6)  # Estimated ball state at t=0
        
        # Cost Function
        # Minimize Time T + minimal control effort
        J = self.T * 10.0 + ca.sumsqr(self.U) * 0.001
        self.opti.minimize(J)
        
        # time constraints
        self.opti.subject_to(self.T >= 0.1)
        self.opti.subject_to(self.T <= 2.0)
        
        dt = self.T / self.params.N
        
        # Dynamics (Trapezoidal collocation / Euler)
        for k in range(self.params.N):
            # Position Integration
            self.opti.subject_to(pos[:, k+1] == pos[:, k] + vel[:, k]*dt + 0.5*self.U[:, k]*dt**2)
            # Velocity Integration
            self.opti.subject_to(vel[:, k+1] == vel[:, k] + self.U[:, k]*dt)
        
        # 3. Initial Condition
        self.opti.subject_to(self.X[:, 0] == self.P_robot_init)
        
        # 4. Physical Limits
        # Slider Limits
        self.opti.subject_to(self.opti.bounded(self.params.rail_min, pos[0, :], self.params.rail_max))
        # Velocity Limits
        for i in range(3):
            self.opti.subject_to(self.opti.bounded(-self.params.dq_max[i], vel[i, :], self.params.dq_max[i]))
            self.opti.subject_to(self.opti.bounded(-self.params.ddq_max[i], self.U[i, :], self.params.ddq_max[i]))

        # --- Terminal Constraints (The Catch) ---
        # Robot FK at time T
        q_final = pos[:, -1]
        l1, l2 = self.params.l1, self.params.l2
        
        # CasADi Symbolic FK
        rx = q_final[0] + l1*ca.cos(q_final[1]) + l2*ca.cos(q_final[1] + q_final[2])
        rz = self.params.base_z_offset + l1*ca.sin(q_final[1]) + l2*ca.sin(q_final[1] + q_final[2])
        
        # Ball Prediction at time T
        bx0, bz0 = self.P_ball_init[0], self.P_ball_init[2]
        bvx, bvz = self.P_ball_init[3], self.P_ball_init[5]
        g = 9.81
        
        bx_T = bx0 + bvx * self.T
        bz_T = bz0 + bvz * self.T - 0.5 * g * self.T**2
        
        # Match Positions
        self.opti.subject_to(rx == bx_T)
        self.opti.subject_to(rz == bz_T)
        
        # "Elbow Up" Heuristic Constraint (keep q2 negative)
        self.opti.subject_to(pos[2, :] <= 0)
        
        # Solver Setup
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)

    def solve(self, robot_state, ball_state):
        self.opti.set_value(self.P_robot_init, robot_state)
        self.opti.set_value(self.P_ball_init, ball_state)
        
        # Warm start T with a guess (e.g., 0.8 seconds)
        self.opti.set_initial(self.T, 0.8)
        
        try:
            sol = self.opti.solve()
            T_opt = sol.value(self.T)
            # Return full trajectory of joint positions and velocities
            pos_traj = sol.value(self.X)[:3, :]
            vel_traj = sol.value(self.X)[3:, :]
            return T_opt, pos_traj, vel_traj
        except Exception as e:
            # print(f"MPC Infeasible: {e}")
            return None, None, None

def get_ee_pos(q, params):
    """Forward kinematics helper for visualization/logging."""
    slider_pos, q1, q2 = q
    l1, l2 = params.l1, params.l2
    x_rel = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    z_rel = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([slider_pos + x_rel, z_rel + params.base_z_offset])

# --- 4. MAIN SIMULATION LOOP ---
def main():
    # Setup
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Parameters & Solvers
    params = ArmParameters()
    estimator = TrajectoryEstimator()
    mpc = TimeOptimalMPC(params)
    
    # Load Robot
    urdf_filename = "3dof_planar_slider.urdf"
    urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/', urdf_filename))
    if not os.path.exists(urdf_path):
        urdf_path = os.path.abspath(os.path.join(os.getcwd(), urdf_filename))
    if not os.path.exists(urdf_path):
        urdf_path = urdf_filename 

    robot_id = p.loadURDF(urdf_path, params.start_pos, params.start_orientation, useFixedBase=True)
    
    # Get Joint Indices
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, j)[2] in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
            joint_indices.append(j)
    
    # Initialize Robot at Home
    home_pos = [0.0, 1.5, -2.0] # Slider 0, Arm retracted
    for i, idx in enumerate(joint_indices):
        p.resetJointState(robot_id, idx, home_pos[i])

    print("========================================")
    print("   PRESS [ENTER] TO THROW BALL")
    print("========================================")
    
    ball_id = -1
    
    while True:
        input("Ready? Press Enter...")
        
        # --- Data Logging Lists ---
        log_time = []
        log_robot_x = []
        log_robot_z = []
        log_ball_x = []
        log_ball_z = []
        log_error = []
        log_slider = []
        log_vel_shoulder = []
        log_vel_elbow = []

        # 1. Spawn Ball
        if ball_id >= 0: p.removeBody(ball_id)
        start_x = np.random.uniform(3.5, 4.5)
        start_z = np.random.uniform(1.0, 1.5)
        vel_x = np.random.uniform(-4.5, -3.5) # Throwing towards robot
        vel_z = np.random.uniform(3.0, 5.0)   # Lofted shot
        
        vis_ball = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1,0,0,1])
        col_ball = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        ball_id = p.createMultiBody(0.2, col_ball, vis_ball, [start_x, 0, start_z])
        p.resetBaseVelocity(ball_id, [vel_x, 0, vel_z])
        
        print(f"Ball Thrown! Init Vel: [{vel_x:.2f}, {vel_z:.2f}]")
        
        # 2. Estimation Phase (Collect Data for 0.1s)
        estimator.reset()
        est_start = time.time()
        while time.time() - est_start < 0.1:
            p.stepSimulation()
            pos, _ = p.getBasePositionAndOrientation(ball_id)
            estimator.add_observation(time.time() - est_start, pos)
            time.sleep(1./240.)
            
        # 3. Fit Trajectory
        ball_state_est = estimator.estimate_state()
        if ball_state_est is None:
            print("Estimation Failed.")
            continue
            
        print(f"Est State: {ball_state_est}")
        
        # 4. Solve MPC
        # Get current robot state
        q_curr = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        dq_curr = [p.getJointState(robot_id, j)[1] for j in joint_indices]
        robot_state = np.array(q_curr + dq_curr)
        
        # Offset time: The ball has already moved 0.1s during estimation
        # We need to project the ball state back to t=0 or shift MPC time.
        # Simpler: Use current estimation as t=0 for MPC.
        
        print("Solving MPC...")
        T_opt, q_traj, v_traj = mpc.solve(robot_state, ball_state_est)
        
        if T_opt is None:
            print("MPC Failed: Target Unreachable")
            continue
            
        print(f"Catch possible in {T_opt:.3f} seconds!")
        
        # 5. Execute Trajectory
        # We have the optimal path in q_traj (21 points). We interpolate.
        exec_start = time.time()
        
        while time.time() - exec_start < T_opt + 0.5: # Run a bit past catch time
            t_now = time.time() - exec_start
            
            # Find index in trajectory (Clamp if past T_opt)
            idx_float = (t_now / T_opt) * params.N
            idx = int(np.clip(idx_float, 0, params.N-1))

            if t_now <= T_opt:
                # Get desired position and velocity for this moment
                target_q = q_traj[:, idx+1]
                target_v = v_traj[:, idx+1]
            else:
                # Hold final position
                 target_q = q_traj[:, -1]
                 target_v = [0, 0, 0]

            # Send Command
            for i, j_id in enumerate(joint_indices):
                p.setJointMotorControl2(
                    robot_id, j_id, p.POSITION_CONTROL,
                    targetPosition=target_q[i],
                    targetVelocity=target_v[i], # Feedforward velocity
                    force=params.torque_limits[i],
                    maxVelocity=params.dq_max[i]
                )

            p.stepSimulation()

            # --- Data Logging ---
            # Get actual robot state
            curr_q = [p.getJointState(robot_id, j)[0] for j in joint_indices]
            curr_dq = [p.getJointState(robot_id, j)[1] for j in joint_indices]
            ee_pos = get_ee_pos(curr_q, params)
            
            # Get actual ball state
            ball_pos, _ = p.getBasePositionAndOrientation(ball_id)
            
            # Log
            log_time.append(t_now)
            log_robot_x.append(ee_pos[0])
            log_robot_z.append(ee_pos[1])
            log_ball_x.append(ball_pos[0])
            log_ball_z.append(ball_pos[2])
            
            dist = np.linalg.norm(ee_pos - np.array([ball_pos[0], ball_pos[2]]))
            log_error.append(dist)
            
            log_slider.append(curr_q[0])
            log_vel_shoulder.append(curr_dq[1])
            log_vel_elbow.append(curr_dq[2])

            time.sleep(1./240.)
        
        print("Catch Attempt Finished.")
        
        # --- Metrics & Graphs ---
        min_error = np.min(log_error)
        idx_min = np.argmin(log_error)
        time_at_min = log_time[idx_min]
        
        print(f"\n--- FINAL METRICS ---")
        print(f"Minimum Miss Distance: {min_error*1000:.2f} mm")
        print(f"Time of Closest Approach: {time_at_min:.3f} s")
        print(f"Planned Intercept Time: {T_opt:.3f} s")
        print(f"Peak Shoulder Vel: {np.max(np.abs(log_vel_shoulder)):.2f} rad/s")
        print(f"Peak Elbow Vel: {np.max(np.abs(log_vel_elbow)):.2f} rad/s")
        
        # Plotting
        plt.figure(figsize=(12, 10))

        # Graph 1: Trajectory Path (X-Z)
        plt.subplot(2, 2, 1)
        plt.plot(log_ball_x, log_ball_z, 'r--', label='Ball Path')
        plt.plot(log_robot_x, log_robot_z, 'g-', label='Robot Path')
        # Mark catch point
        plt.scatter([log_ball_x[idx_min]], [log_ball_z[idx_min]], c='blue', marker='x', s=100, label='Intercept')
        plt.title('Interception Trajectory')
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.legend()
        plt.grid(True)

        # Graph 2: Distance to Ball
        plt.subplot(2, 2, 2)
        plt.plot(log_time, np.array(log_error)*1000, 'b-')
        plt.axvline(x=T_opt, color='r', linestyle='--', label=f'Planned T={T_opt:.2f}s')
        plt.title('Distance to Ball vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (mm)')
        plt.legend()
        plt.grid(True)

        # Graph 3: Joint Velocities
        plt.subplot(2, 2, 3)
        plt.plot(log_time, log_vel_shoulder, label='Shoulder')
        plt.plot(log_time, log_vel_elbow, label='Elbow')
        plt.axhline(y=params.dq_max[1], color='r', linestyle=':', alpha=0.5)
        plt.axhline(y=-params.dq_max[1], color='r', linestyle=':', alpha=0.5)
        plt.title('Joint Velocities')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (rad/s)')
        plt.legend()
        plt.grid(True)

        # Graph 4: Slider Position
        plt.subplot(2, 2, 4)
        plt.plot(log_time, log_slider, 'm-')
        plt.title('Slider Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
        time.sleep(1.0)

if __name__ == "__main__":
    main()