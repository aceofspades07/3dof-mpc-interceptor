import numpy as np
import pybullet as p
import pybullet_data
import time
import scipy.optimize as scipy
from dataclasses import dataclass
import os

@dataclass
class ArmParameters:
    # Parameters for 2-link planar arm
    l1: float = 1.0  # Length of link 1 (m)
    l2: float = 1.0  # Length of link 2 (m)
    m1: float = 1.0  # Mass of link 1 (kg)
    m2: float = 0.8  # Mass of link 2 (kg)
    q_min: np.ndarray = np.array([0, 0])
    q_max: np.ndarray = np.array([2*np.pi, 2*np.pi])
    dq_max: np.ndarray = np.array([2.0, 2.0])  # rad/s
    ddq_max: np.ndarray = np.array([5.0, 5.0])  # rad/s^2

@dataclass
class FlightParameters:
    # Estimated parameters of parabolic flight
    x0: float = 0.0
    z0: float = 0.0
    vx0: float = 0.0
    vz0: float = 0.0
    g: float = 9.81
    valid: bool = False

class TwoLinkArm:
    
    def __init__(self, params: ArmParameters):
        self.params = params
        
    def forward_kinematics(self, q: np.ndarray) -> tuple: # Compute end-effector position and orientation
        
        l1, l2 = self.params.l1, self.params.l2
        q1, q2 = q[0], q[1] 
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        z = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        pos = [x , z] 
        
        angle = q1 + q2
        
        return np.array(pos), angle
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        
        l1, l2 = self.params.l1, self.params.l2
        q1, q2 = q[0], q[1]
        
        # Jacobian for [x, z] velocities
        J = np.array([
            [-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2)],
            [l1*np.cos(q1) + l2*np.cos(q1+q2), l2*np.cos(q1+q2)]
        ])
        
        return J

    def inverse_kinematics(self, pos: np.ndarray, angle: float = None) -> list:
        
        x, z = pos[0], pos[1]
        l1, l2 = self.params.l1, self.params.l2

        r2 = x**2 + z**2
        if r2 < (l1 - l2)**2 - 1e-5 or r2 > (l1 + l2)**2 + 1e-5:
            return []

        D = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
        
        D = max(-1.0, min(1.0, D))

        solutions = []
        for sign in [1.0, -1.0]:
            q2 = np.arctan2(sign * np.sqrt(max(0.0, 1 - D**2)), D)

            q1 = np.arctan2(z, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))

            q1 = np.arctan2(np.sin(q1), np.cos(q1))
            q2 = np.arctan2(np.sin(q2), np.cos(q2))

            q = np.array([q1, q2])

            if angle is not None:
                ee_angle = q1 + q2
                angle_diff = np.arctan2(np.sin(ee_angle - angle), np.cos(ee_angle - angle))
                if abs(angle_diff) > 0.2:  # ~11 degrees tolerance
                    continue

            solutions.append(q)

        return solutions
    
    def inverse_dynamics(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        
        l1, l2 = self.params.l1, self.params.l2
        m1, m2 = self.params.m1, self.params.m2
        g = 9.81
        
        q1, q2 = q[0], q[1]
        dq1, dq2 = dq[0], dq[1]
        
        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2*m2*l1*l2*np.cos(q2)
        M12 = m2 * l2**2 + m2*l1*l2*np.cos(q2)
        M21 = M12
        M22 = m2 * l2**2
        M = np.array([[M11, M12], [M21, M22]])
        
        h = -m2*l1*l2*np.sin(q2)
        C = np.array([
            [h*dq2, h*(dq1+dq2)],
            [-h*dq1, 0]
        ])
        
        G = np.array([
            (m1 + m2)*g*l1*np.cos(q1) + m2*g*l2*np.cos(q1+q2),
            m2*g*l2*np.cos(q1+q2)
        ])
        
        tau = M @ ddq + C @ dq + G
        return tau

class FlightEstimator:
    # Estimate parabolic flight parameters using least squares
    
    def __init__(self):
        self.measurements = []
        self.g = 9.81
        
    def add_measurement(self, pos: np.ndarray, t: float):
        # Add a position measurement at time t
        self.measurements.append((pos.copy(), t))
        
    def estimate(self) -> FlightParameters:
        # Estimate flight parameters from measurements
        if len(self.measurements) < 3:
            return FlightParameters(valid=False)
        
        # Extract measurements
        positions = np.array([m[0] for m in self.measurements])
        times = np.array([m[1] for m in self.measurements])
        
        # Shift time to start at 0
        t0 = times[0]
        times = times - t0
        
        # Least squares for x(t) = x0 + vx0*t
        A_x = np.column_stack([np.ones(len(times)), times])
        params_x = np.linalg.lstsq(A_x, positions[:, 0], rcond=None)[0]
        x0, vx0 = params_x
        
        # Least squares for z(t) = z0 + vz0*t - 0.5*g*t^2
        A_z = np.column_stack([np.ones(len(times)), times, -0.5*times**2])
        params_z = np.linalg.lstsq(A_z, positions[:, 1], rcond=None)[0]
        z0, vz0, g_est = params_z[0], params_z[1], params_z[2]*2
        
        return FlightParameters(
            x0=x0, z0=z0, vx0=vx0, vz0=vz0,
            g=self.g, valid=True
        )
    
    def predict_trajectory(self, flight: FlightParameters, t_future: float) -> np.ndarray:
        # Predict position at future time
        x = flight.x0 + flight.vx0 * t_future
        z = flight.z0 + flight.vz0 * t_future - 0.5 * flight.g * t_future**2
        return np.array([x, z])
    
    def clear(self):
        # Clear all measurements
        self.measurements = []

class MPCCatcher:
    # MPC-based catching controller
    
    def __init__(self, arm: TwoLinkArm):
        self.arm = arm
        self.R = np.eye(2) * 0.1  
        
    def terminal_constraints(self, q: np.ndarray, flight: FlightParameters) -> np.ndarray:
        
        pos_e, angle_e = self.arm.forward_kinematics(q)
        x_e, z_e = pos_e
        
        # Constraint 1: z position on parabola (using x to eliminate time)
        if abs(flight.vx0) > 1e-6:
            t = (x_e - flight.x0) / flight.vx0
            z_parabola = flight.z0 + flight.vz0*t - 0.5*flight.g*t**2
            g1 = z_e - z_parabola
        else:
            g1 = 0.0
        
        # Constraint 2: Orientation aligned with trajectory tangent
        # Tangent angle = atan2(vz, vx) where vz = vz0 - g*t
        if abs(flight.vx0) > 1e-6:
            t = (x_e - flight.x0) / flight.vx0
            vz = flight.vz0 - flight.g * t
            tangent_angle = np.arctan2(vz, flight.vx0)
            # Normalize angle difference to [-pi, pi]
            angle_diff = angle_e - tangent_angle
            g2 = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        else:
            g2 = 0.0
        
        return np.array([g1, g2])
    
    def solve_catching_pose(self, q0: np.ndarray, flight: FlightParameters) -> tuple:
        # For a 2-link planar arm we can do a deterministic search over
        # candidate interception times and use analytic IK to find joint
        # solutions that satisfy the terminal constraints. This is far more
        # reliable than relying on a local optimizer in joint space.

        best_q = None
        best_cost = np.inf
        best_ok = False

        # Time window to search (seconds) - tuneable
        t_min = 0.05
        t_max = 3.0
        samples = 300

        # If vx0 is near zero, we'll still sample times but orientation checks
        # will be relaxed.
        for t in np.linspace(t_min, t_max, samples):
            # Predict end-effector target in world coords (x,z)
            pos = self.arm.forward_kinematics(q0)[0]  # current ee pos (not used)
            target = FlightEstimator().predict_trajectory(flight, t)

            x_e, z_e = target[0], target[1]

            # Ignore points below ground or too close to base (safety)
            if z_e < 0.05:
                continue

            # Try IK for this target. Preferred orientation is the trajectory tangent
            desired_angle = None
            if abs(flight.vx0) > 1e-6:
                vz = flight.vz0 - flight.g * t
                desired_angle = np.arctan2(vz, flight.vx0)

            ik_sols = self.arm.inverse_kinematics(np.array([x_e, z_e]), angle=desired_angle)

            # If no solutions with orientation constraint, try without orientation
            if len(ik_sols) == 0:
                ik_sols = self.arm.inverse_kinematics(np.array([x_e, z_e]), angle=None)

            for q_candidate in ik_sols:
                # Evaluate terminal constraint error and workspace penalties
                g = self.terminal_constraints(q_candidate, flight)
                constraint_err = np.linalg.norm(g)

                pos_ee, _ = self.arm.forward_kinematics(q_candidate)
                x_c, z_c = pos_ee

                workspace_penalty = 0.0
                if z_c < 0.1:
                    workspace_penalty += 1000 * (0.1 - z_c)**2
                if x_c < 0.05:
                    workspace_penalty += 1000 * (0.05 - x_c)**2

                distance_cost = 0.1 * np.sum((q_candidate - q0)**2)

                total_cost = 1000.0 * (constraint_err**2) + distance_cost + workspace_penalty

                # Accept if constraint error within tolerance and better cost
                if constraint_err < 0.08 and total_cost < best_cost:
                    best_cost = total_cost
                    best_q = q_candidate
                    best_ok = True

        if best_q is None:
            # No IK-based solution found; fall back to returning q0 (no catch)
            return q0, False

        return best_q, best_ok

class PyBulletSimulation:
    # Pybullet sim environment
    
    def __init__(self, urdf_path: str = "urdf/2linkrobot.urdf", arm_params: ArmParameters = None):
        # Connect to pybullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Configure camera
        p.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=90,
            cameraPitch=-20,
            cameraTargetPosition=[1, 0, 1]
        )
        
        # Load ground plane
        self.plane = p.loadURDF("plane.urdf")
        
        # Check if URDF exists
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Load the arm (rotate to yz plane)
        # Rotation: -90 degrees around Y axis to go from xy to yz plane (right-side up)
        base_orientation = p.getQuaternionFromEuler([0, -np.pi/2, 0])
        self.arm_id = p.loadURDF(urdf_path, [0, 0, 0], 
                                 baseOrientation=base_orientation,
                                 useFixedBase=True)
        
        # Get joint info and find end effector link
        self.num_joints = p.getNumJoints(self.arm_id)
        print(f"Loaded arm with {self.num_joints} joints")
        
        # Find end effector link index
        # In PyBullet, link index corresponds to joint index
        # The ee_joint (fixed joint) will have the endEffector as its child link
        self.ee_link_idx = None
        for i in range(self.num_joints):
            info = p.getJointInfo(self.arm_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            print(f"Joint {i}: {joint_name}, type: {joint_type}")
            if joint_name == "ee_joint" or i == self.num_joints - 1:
                # Fixed joint (type 4) or last joint - this is the end effector
                self.ee_link_idx = i
        
        if self.ee_link_idx is None:
            print("Warning: Could not find end effector, using link 1")
            self.ee_link_idx = 1
        
        print(f"Using end effector link index: {self.ee_link_idx}")
        
        # Ball parameters
        self.ball_id = None
        self.catching_net_id = None

        # Keep a copy of arm parameters (joint limits, etc.) so the simulation
        # can perform joint-level control without needing a TwoLinkArm instance.
        if arm_params is None:
            self.arm_params = ArmParameters()
        else:
            self.arm_params = arm_params

        # Debug logging for ball positions (list of tuples (t, x, z, vx, vz))
        self.debug_log = False
        self.ball_history = []

        # Create catching net visualization
        self.create_catching_net()
        
    def create_catching_net(self):
        # Create a visual catching net at end effector
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.15,
            length=0.02,
            rgbaColor=[0.3, 0.3, 0.8, 0.5]
        )
        
        self.catching_net_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, -10]  # Hidden initially
        )
        
    def update_catching_net_position(self):
        # Update catching net to follow end effector
        try:
            # Get end effector link state
            ee_state = p.getLinkState(self.arm_id, self.ee_link_idx)
            
            if ee_state is not None and len(ee_state) > 1:
                ee_pos = ee_state[0]
                ee_orn = ee_state[1]
                
                # Update catching net position
                p.resetBasePositionAndOrientation(
                    self.catching_net_id,
                    ee_pos,
                    ee_orn
                )
        except Exception as e:
            # Silently handle errors - catching net is just visualization
            pass

    def get_end_effector_pose(self):
        """Return end-effector (x,z) in world coordinates or None if unavailable."""
        try:
            if self.ee_link_idx is None:
                return None
            ee_state = p.getLinkState(self.arm_id, self.ee_link_idx)
            if ee_state is None:
                return None
            ee_pos = ee_state[0]
            return np.array([ee_pos[0], ee_pos[2]])
        except Exception:
            return None

    def check_ball_contact(self, distance_threshold: float = 0.0):
        """Check for contacts between the ball and the arm links.

        Returns a list of link indices on the arm that are in contact with the ball.
        If there is no ball or no contacts, returns an empty list.
        """
        contacts = []
        try:
            if self.ball_id is None:
                return contacts
            # Use getContactPoints to find actual contacts (distance <= 0)
            pts = p.getContactPoints(bodyA=self.ball_id, bodyB=self.arm_id)
            for pt in pts:
                # pt[4] is linkIndexA, pt[3] is linkIndexB? Use the documented tuple fields.
                # Safer to use the dictionary-like access if available; otherwise index 4/3 vary.
                # Here we assume the returned tuple has link indices at these positions.
                # Use the explicit API fields via the contact tuple structure:
                try:
                    # In pybullet Python wrapper, ContactPoint has these indices:
                    # (contactFlag, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, ...)
                    link_idx = pt[4] if len(pt) > 4 else -1
                except Exception:
                    link_idx = -1
                # If link index invalid, try index 3
                if link_idx == -1 and len(pt) > 3:
                    link_idx = pt[3]
                if link_idx is not None and link_idx >= 0:
                    contacts.append(link_idx)
        except Exception:
            # If getContactPoints not available or fails, silently return empty
            return contacts

        # Remove duplicates
        return sorted(list(set(contacts)))
    
    def throw_ball(self, start_pos: np.ndarray, velocity: np.ndarray):
        # Create and throw a ball
        if self.ball_id is not None:
            p.removeBody(self.ball_id)
        
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.065)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.065,
            rgbaColor=[1, 0.8, 0, 1]
        )
        
        # Make sure ball starts above the plane by at least its radius
        radius = 0.065
        z0 = max(start_pos[1], radius + 0.01)

        self.ball_id = p.createMultiBody(
            baseMass=0.057,  # Tennis ball mass (kg)
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[start_pos[0], 0, z0]
        )

        p.resetBaseVelocity(
            self.ball_id,
            linearVelocity=[velocity[0], 0, velocity[1]]
        )

        print(f"Ball spawned at x={start_pos[0]:.2f}, z={z0:.2f} with vel vx={velocity[0]:.2f}, vz={velocity[1]:.2f}")
    
    def get_ball_state(self):
        """Get current ball position and velocity"""
        if self.ball_id is None:
            return None, None
        
        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        vel, _ = p.getBaseVelocity(self.ball_id)
        
        return np.array([pos[0], pos[2]]), np.array([vel[0], vel[2]])
    
    def set_joint_positions(self, q: np.ndarray):
        """Set arm joint positions (only revolute joints)"""
        joint_indices = [0, 1]  # baseHinge and interArm
        for i, joint_idx in enumerate(joint_indices):
            p.resetJointState(self.arm_id, joint_idx, q[i])
    
    def get_joint_positions(self):
        """Get current joint positions"""
        joint_indices = [0, 1]
        q = []
        for joint_idx in joint_indices:
            state = p.getJointState(self.arm_id, joint_idx)
            q.append(state[0])
        return np.array(q)
    
    def control_joints_pd(self, q_target: np.ndarray, kp=10.0, kd=1.0):
        """Simple PD control for joint positions"""
        # Use POSITION_CONTROL for stable joint motion in PyBullet.
        # Clamp targets to joint limits and apply position control with gains.
        q_target_clamped = q_target.copy()
        for i in range(2):
            qmin = self.arm_params.q_min[i]
            qmax = self.arm_params.q_max[i]
            q_target_clamped[i] = float(np.clip(q_target_clamped[i], qmin, qmax))

        max_forces = [200.0, 200.0]
        # Convert gains to reasonable position/velocity gains in PyBullet
        pos_gain = min(1.0, kp * 0.02)
        vel_gain = min(1.0, kd * 0.02)

        p.setJointMotorControlArray(
            self.arm_id,
            [0, 1],
            p.POSITION_CONTROL,
            targetPositions=[float(q_target_clamped[0]), float(q_target_clamped[1])],
            positionGains=[pos_gain, pos_gain],
            velocityGains=[vel_gain, vel_gain],
            forces=max_forces
        )

        # Diagnostic: print if joints hit limits
        q_curr = self.get_joint_positions()
        for i in range(2):
            if abs(q_curr[i] - self.arm_params.q_min[i]) < 1e-3 or abs(q_curr[i] - self.arm_params.q_max[i]) < 1e-3:
                print(f"Warning: joint {i} near limit: {q_curr[i]:.3f} rad")
    
    def step(self):
        p.stepSimulation()
        self.update_catching_net_position()
        time.sleep(1./240.)
    
    def cleanup(self):
        # If we collected ball history, save it for debugging
        try:
            if len(self.ball_history) > 0:
                import csv
                out_path = os.path.join(os.getcwd(), 'ball_trajectory.csv')
                with open(out_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Header: time, ball(x,z), ball(vx,vz), ee(x,z)
                    writer.writerow(['t', 'ball_x', 'ball_z', 'ball_vx', 'ball_vz', 'ee_x', 'ee_z'])
                    for row in self.ball_history:
                        # Ensure rows with older format are padded
                        if len(row) == 5:
                            row = list(row) + [float('nan'), float('nan')]
                        writer.writerow(row)
                print(f"Wrote ball trajectory to {out_path}")
        except Exception:
            pass

        p.disconnect()

def main():
    """Main catching demonstration"""
    print("=" * 60)
    print("2-Link Arm Ball Catching with MPC")
    print("=" * 60)
    
    # Initialize components
    # Create arm parameters and pass them to the simulation so the sim can
    # use joint limits for safe position-control even before a TwoLinkArm
    # instance is constructed.
    arm_params = ArmParameters()
    sim = PyBulletSimulation(urdf_path="urdf/2linkrobot.urdf", arm_params=arm_params)
    arm = TwoLinkArm(arm_params)
    estimator = FlightEstimator()
    mpc = MPCCatcher(arm)
    
    # Initial arm configuration (pointing upward-right)
    q_init = np.array([np.pi/3, np.pi/6])
    sim.set_joint_positions(q_init)

    # Immediately enable position control to HOLD the initial pose until
    # a catching target is computed. Without this, joints are passive and
    # the arm can fall under gravity into a resting ("stuck") pose.
    sim.control_joints_pd(q_init, kp=50.0, kd=5.0)
    
    print("\nPress ENTER to throw ball...")
    input()
    
    # Throw ball (adjusted to be within reach of 2-link arm)
    # Use a closer, slightly faster throw so the ball crosses the arm plane
    # at a higher z and earlier time (reduces gravity drop).
    # If this still barely reaches the base, try changing start_x closer to 1.0
    start_pos = np.array([1.0, 2.5])
    velocity = np.array([-1.8, 2.0])
    sim.throw_ball(start_pos, velocity)
    
    print(f"\n✓ Ball thrown from ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    print(f"  Initial velocity: ({velocity[0]:.2f}, {velocity[1]:.2f}) m/s")
    
    # Simulation loop
    t = 0
    dt = 1./240.
    catching_started = False
    catching_initiated = False
    q_target = q_init.copy()
    
    measurement_count = 0
    last_print_time = 0
    
    # Run until the pybullet GUI is closed by the user. This lets you
    # continue inspecting after contacts instead of the script exiting
    # immediately. We also handle KeyboardInterrupt so Ctrl+C stops cleanly.
    contact_logged = False
    try:
        while p.isConnected():
            # Get ball state
            ball_pos, ball_vel = sim.get_ball_state()

            if ball_pos is not None and ball_pos[1] > 0:  # Ball in air
                # Add measurement every 10 steps (more realistic)
                if int(t * 240) % 10 == 0:
                    estimator.add_measurement(ball_pos, t)
                    measurement_count += 1

                # Debug: log ball positions and end-effector for first 2 seconds to help diagnose trajectory
                if t < 2.0:
                    try:
                        ee = sim.get_end_effector_pose()
                        ee_x = float(ee[0]) if ee is not None else float('nan')
                        ee_z = float(ee[1]) if ee is not None else float('nan')
                        sim.ball_history.append((t, float(ball_pos[0]), float(ball_pos[1]), float(ball_vel[0]), float(ball_vel[1]), ee_x, ee_z))
                    except Exception:
                        pass
                    if int(t * 240) % 5 == 0:
                        print(f"[DBG] t={t:.3f}s ball ({ball_pos[0]:.3f},{ball_pos[1]:.3f}) vel ({ball_vel[0]:.3f},{ball_vel[1]:.3f}) | ee ({ee_x:.3f},{ee_z:.3f})")
                
                # Estimate flight parameters after enough measurements
                if measurement_count >= 5 and not catching_initiated:
                    flight = estimator.estimate()
                    
                    if flight.valid:
                        print(f"\n Flight trajectory estimated (t={t:.2f}s, {measurement_count} measurements)")
                        print(f"  Estimated initial position: ({flight.x0:.2f}, {flight.z0:.2f})")
                        print(f"  Estimated initial velocity: ({flight.vx0:.2f}, {flight.vz0:.2f})")
                        
                        # Solve for catching pose
                        print("\n⚙ Computing optimal catching pose...")
                        q_target, success = mpc.solve_catching_pose(q_init, flight)
                        
                        if success:
                            pos_target, angle_target = arm.forward_kinematics(q_target)
                            
                            print(f"  Catching solution found:")
                            print(f"  Joint angles: [{np.rad2deg(q_target[0]):.1f}°, {np.rad2deg(q_target[1]):.1f}°]")
                            print(f"  End-effector position: ({pos_target[0]:.2f}, {pos_target[1]:.2f})")
                            print(f"  End-effector angle: {np.rad2deg(angle_target):.1f}°")
                            
                            constraints = mpc.terminal_constraints(q_target, flight)
                            print(f"  Constraint error: {np.linalg.norm(constraints):.4f}")
                            
                            catching_initiated = True
                            catching_started = True
                        else:
                            print("Failed to find valid catching pose (ball out of reach)")
                            # Keep the arm in a safe holding pose (initial pose)
                            # so it doesn't become passive and fall to an arbitrary pose.
                            q_target = q_init.copy()
                            sim.control_joints_pd(q_target, kp=50.0, kd=5.0)
                            catching_initiated = True

            # Control arm to move to target
            if catching_started:
                sim.control_joints_pd(q_target, kp=50.0, kd=5.0)

            # Print status every 1s (always) to help diagnose stuck joints
            if t - last_print_time > 1.0:
                q_current = sim.get_joint_positions()
                try:
                    error = np.linalg.norm(q_target - q_current)
                except Exception:
                    error = 0.0
                if ball_pos is not None:
                    print(f"  t={t:.1f}s | Ball: ({ball_pos[0]:.2f}, {ball_pos[1]:.2f}) | Joint error: {np.rad2deg(error):.1f}° | q_current: [{np.rad2deg(q_current[0]):.1f}°, {np.rad2deg(q_current[1]):.1f}°]")
                else:
                    print(f"  t={t:.1f}s | q_current: [{np.rad2deg(q_current[0]):.1f}°, {np.rad2deg(q_current[1]):.1f}°]")
                last_print_time = t

            sim.step()
            # After stepping, check for contacts between ball and arm
            contacts = sim.check_ball_contact()
            if len(contacts) > 0 and not contact_logged:
                # Report which links contacted the ball (log once)
                print(f"Contact detected between ball and arm links: {contacts}")
                if sim.ee_link_idx in contacts:
                    print("Result: CAUGHT by end-effector (link index {}).".format(sim.ee_link_idx))
                else:
                    print("Result: CONTACT on non-end-effector link(s) - NOT counted as catch.")
                # Don't break; keep running so the user can inspect the GUI.
                contact_logged = True

            # Allow a graceful finish if user closes the GUI: loop ends when p.isConnected() becomes False
            # Advance time
            t += dt
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt - exiting simulation loop")

        t += dt
    
    print("\n" + "=" * 60)
    print("Simulation complete")
    print("=" * 60)
    
    time.sleep(1)
    sim.cleanup()

if __name__ == "__main__":
    main()