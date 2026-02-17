"""
Time-Optimal MPC Ball Catching for a 3-DOF PRR arm.
Uses trajectory estimation and time-optimal control to catch a thrown ball.
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
    print("Error: CasADi not found. Please run 'pip install casadi'")
    sys.exit(1)


class ArmParameters:
    l1: float = 1.0  
    l2: float = 1.0  
    base_z_offset: float = 0.06
    
    start_pos = np.array([0, 0, 0.1])
    start_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    
    rail_min = -2.0
    rail_max = 2.0
    dq_max = np.array([2.0, 5.0, 5.0])
    ddq_max = np.array([4.0, 10.0, 10.0])
    torque_limits = np.array([500.0, 200.0, 200.0])
    
    N = 20


class TrajectoryEstimator:
    """Estimates ball trajectory from noisy observations."""
    
    def __init__(self):
        self.history_t = []
        self.history_x = []
        self.history_z = []
        
    def reset(self):
        self.history_t = []
        self.history_x = []
        self.history_z = []
        
    def add_observation(self, t, pos):
        # Add sensor noise
        noise = np.random.normal(0, 0.005, size=3)
        noisy_pos = pos + noise
        
        self.history_t.append(t)
        self.history_x.append(noisy_pos[0])
        self.history_z.append(noisy_pos[2])
        
    def estimate_state(self):
        """Fit ballistic trajectory using least squares."""
        if len(self.history_t) < 5: return None
        
        T = np.array(self.history_t)
        X = np.array(self.history_x)
        Z = np.array(self.history_z)
        
        # Fit linear X model
        A_x = np.vstack([np.ones(len(T)), T]).T
        params_x, _, _, _ = np.linalg.lstsq(A_x, X, rcond=None)
        x0_est, vx_est = params_x
        
        # Fit quadratic Z model
        A_z = np.vstack([np.ones(len(T)), T, T**2]).T
        params_z, _, _, _ = np.linalg.lstsq(A_z, Z, rcond=None)
        z0_est, vz_est, acc_term = params_z
        
        return np.array([x0_est, 0, z0_est, vx_est, 0, vz_est])


class TimeOptimalMPC:
    """MPC solver that optimizes both trajectory and catch time."""
    
    def __init__(self, params: ArmParameters):
        self.params = params
        self.opti = None
        self.build_solver()

    def build_solver(self):
        self.opti = ca.Opti()
        
        # Time is an optimization variable
        self.T = self.opti.variable()
        
        self.X = self.opti.variable(6, self.params.N + 1)
        pos = self.X[:3, :]
        vel = self.X[3:, :]
        
        self.U = self.opti.variable(3, self.params.N)
        
        self.P_robot_init = self.opti.parameter(6)
        self.P_ball_init = self.opti.parameter(6)
        
        # Minimize time plus control effort
        J = self.T * 10.0 + ca.sumsqr(self.U) * 0.001
        self.opti.minimize(J)
        
        # Time bounds
        self.opti.subject_to(self.T >= 0.1)
        self.opti.subject_to(self.T <= 2.0)
        
        dt = self.T / self.params.N
        
        # Dynamics constraints
        for k in range(self.params.N):
            self.opti.subject_to(pos[:, k+1] == pos[:, k] + vel[:, k]*dt + 0.5*self.U[:, k]*dt**2)
            self.opti.subject_to(vel[:, k+1] == vel[:, k] + self.U[:, k]*dt)
        
        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.P_robot_init)
        
        # Physical limits
        self.opti.subject_to(self.opti.bounded(self.params.rail_min, pos[0, :], self.params.rail_max))
        for i in range(3):
            self.opti.subject_to(self.opti.bounded(-self.params.dq_max[i], vel[i, :], self.params.dq_max[i]))
            self.opti.subject_to(self.opti.bounded(-self.params.ddq_max[i], self.U[i, :], self.params.ddq_max[i]))

        # Terminal constraint: FK matches ball position
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
        
        # Elbow up heuristic
        self.opti.subject_to(pos[2, :] <= 0)
        
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)

    def solve(self, robot_state, ball_state):
        self.opti.set_value(self.P_robot_init, robot_state)
        self.opti.set_value(self.P_ball_init, ball_state)
        
        self.opti.set_initial(self.T, 0.8)
        
        try:
            sol = self.opti.solve()
            T_opt = sol.value(self.T)
            pos_traj = sol.value(self.X)[:3, :]
            vel_traj = sol.value(self.X)[3:, :]
            return T_opt, pos_traj, vel_traj
        except Exception as e:
            return None, None, None

# --- 4. MAIN SIMULATION LOOP ---
def main():
    # Setup
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    params = ArmParameters()
    estimator = TrajectoryEstimator()
    mpc = TimeOptimalMPC(params)
    
    # Load robot URDF
    urdf_filename = "3dof_planar_slider.urdf"
    urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/', urdf_filename))
    if not os.path.exists(urdf_path):
        urdf_path = os.path.abspath(os.path.join(os.getcwd(), urdf_filename))
    if not os.path.exists(urdf_path):
        urdf_path = urdf_filename 

    robot_id = p.loadURDF(urdf_path, params.start_pos, params.start_orientation, useFixedBase=True)
    
    # Get joint indices for controllable joints
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, j)[2] in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
            joint_indices.append(j)
    
    # Initialize robot at home position
    home_pos = [0.0, 1.5, -2.0]
    for i, idx in enumerate(joint_indices):
        p.resetJointState(robot_id, idx, home_pos[i])

    print("========================================")
    print("   PRESS [ENTER] TO THROW BALL")
    print("========================================")
    
    ball_id = -1
    
    while True:
        input("Ready? Press Enter...")
        
        # Spawn ball with random initial conditions
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
        
        # Estimation phase: collect observations for 0.1s
        estimator.reset()
        est_start = time.time()
        while time.time() - est_start < 0.1:
            p.stepSimulation()
            pos, _ = p.getBasePositionAndOrientation(ball_id)
            estimator.add_observation(time.time() - est_start, pos)
            time.sleep(1./240.)
            
        # Fit trajectory from observations
        ball_state_est = estimator.estimate_state()
        if ball_state_est is None:
            print("Estimation Failed.")
            continue
            
        print(f"Est State: {ball_state_est}")
        
        # Get current robot state
        q_curr = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        dq_curr = [p.getJointState(robot_id, j)[1] for j in joint_indices]
        robot_state = np.array(q_curr + dq_curr)
        
        # Solve MPC for optimal catch trajectory
        print("Solving MPC...")
        T_opt, q_traj, v_traj = mpc.solve(robot_state, ball_state_est)
        
        if T_opt is None:
            print("MPC Failed: Target Unreachable")
            continue
            
        print(f"Catch possible in {T_opt:.3f} seconds!")
        
        # Execute trajectory by interpolating through waypoints
        exec_start = time.time()
        
        while time.time() - exec_start < T_opt:
            t_now = time.time() - exec_start

            # Find trajectory index
            idx_float = (t_now / T_opt) * params.N
            idx = int(np.clip(idx_float, 0, params.N-1))

            target_q = q_traj[:, idx+1]
            target_v = v_traj[:, idx+1]

            curr_q = [p.getJointState(robot_id, j)[0] for j in joint_indices]
            curr_v = [p.getJointState(robot_id, j)[1] for j in joint_indices]

            ball_pos, _ = p.getBasePositionAndOrientation(ball_id)

            # Debug output
            print("----- MPC Step -----")
            print(f"Elapsed: {t_now:.3f} / {T_opt:.3f} s (idx {idx})")
            print(f"Target joint pos: {target_q}")
            print(f"Target joint vel: {target_v}")
            print(f"Current joint pos: {curr_q}")
            print(f"Current joint vel: {curr_v}")
            print(f"Ball position: {ball_pos}")

            # Send position commands with feedforward velocity
            for i, j_id in enumerate(joint_indices):
                p.setJointMotorControl2(
                    robot_id, j_id, p.POSITION_CONTROL,
                    targetPosition=target_q[i],
                    targetVelocity=target_v[i],
                    force=params.torque_limits[i],
                    maxVelocity=params.dq_max[i]
                )

            p.stepSimulation()
            time.sleep(1./240.)
        
        print("Catch Attempt Finished.")
        time.sleep(1.0)

if __name__ == "__main__":
    main()