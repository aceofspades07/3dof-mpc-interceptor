import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import os

class ArmParameters:
    # Parameters for 2-link planar arm
    l1: float = 1.0  # Length of link 1 (m)
    l2: float = 1.0  # Length of link 2 (m)
    m1: float = 1.0  # Mass of link 1 (kg)
    m2: float = 0.8  # Mass of link 2 (kg)
    start_pos : np.ndarray = np.array([0, 0, 0.1]) # Start 0.1 meters in the air 
    start_orientation : np.ndarray = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    q_min: np.ndarray = np.array([-np.pi, -np.pi])
    q_max: np.ndarray = np.array([np.pi, np.pi])
    dq_max: np.ndarray = np.array([5.0, 5.0])  # rad/s
    ddq_max: np.ndarray = np.array([5.0, 5.0])  # rad/s^2
    torque_limits: np.ndarray = np.array([100.0, 100.0])  # Nm
    config : str = "ELBOW_UP" # "ELBOW_UP" or "ELBOW_DOWN"
    control_mode: str = "POSITION"  # "POSITION" or "VELOCITY"

class ArmDynamics:
    def __init__(self, params: ArmParameters):
        self.params = params

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        l1, l2 = self.params.l1, self.params.l2
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        z = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        return np.array([x, z])
    
    def ik_solver(self, target_pos: np.ndarray) -> np.ndarray:
        l1, l2 = self.params.l1, self.params.l2
        max_reach = l1 + l2
        min_reach = abs(l1 - l2)

        base_height = self.params.start_pos[2]

        x, z = target_pos

        if z < base_height:
            z = base_height  # Restrict to upper half-plane

        dist = np.sqrt(x**2 + z**2)
        # If target is OUTSIDE max reach, pull it in to the max radius
        if dist + 1e-6 > max_reach:
            scale = max_reach / dist
            x = x * scale
            z = z * scale
        #If target is INSIDE min reach (too close to itself), push it out
        elif dist + 1e-6 < min_reach:
            scale = min_reach / dist
            x = x * scale
            z = z * scale
        
        if z < base_height:
            z = base_height  # Restrict to upper half-plane
        
        cos_q2 = (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)  # Numerical safety

        sin_q2 = np.sqrt(1 - cos_q2**2)
        sin_q2 = np.clip(sin_q2, -1.0, 1.0)  # Numerical safety

        q2_mag = np.arctan2(sin_q2, cos_q2)
        q2 = [-q2_mag, q2_mag]      

        cos_q2 = np.cos(q2)
        sin_q2 = np.sin(q2)

        k1 = l1 + l2 * cos_q2
        k2 = l2 * sin_q2
        q1 = np.arctan2(z, x) - np.arctan2(k2, k1)

        # do not let q1 go beyond 0 to pi
        q1 = np.clip(q1, 0, np.pi)

        return np.array([q1, q2])
    

arm_params = ArmParameters()

# 1. Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 2. Set up the environment
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# 3. Define the Spawn Orientation
start_pos = arm_params.start_pos
start_orientation = arm_params.start_orientation

urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/2linkarm.urdf'))
if not os.path.exists(urdf_path):
        urdf_path = os.path.abspath(os.path.join(os.getcwd(), 'urdf/2linkarm.urdf'))
if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

# 4. Load the Robot
robot_id = p.loadURDF(
    urdf_path,
    basePosition=start_pos,
    baseOrientation=start_orientation,
    useFixedBase=True  # Keeps the base link pinned in space
)

# 5. Setup User Debug Parameters (Sliders) to test the XZ motion
joint_ids = []
param_ids = []

print(p.getJointInfo(robot_id, 0))

# We only care about the revolute joints (indices 0 and 1 based on your URDF)
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    
    if joint_type == p.JOINT_REVOLUTE:
        joint_ids.append(j)

# Trajectory Definition
start_point = np.array([0.632, 0.589])
end_point   = np.array([-0.529, 1.345])

dynamics = ArmDynamics(arm_params)
filtered_angles = np.zeros(2) 
first_run = True
last_print_time = time.time()

start_time = time.time()
duration = 1.0 

print(f"Starting Cubic Trajectory: {start_point} -> {end_point}")

prev_ee_pos = None 

# 6. Simulation Loop
while True:
    now = time.time()
    elapsed = now - start_time
    
    if elapsed < duration:
        u = elapsed / duration
        ratio = 3 * (u**2) - 2 * (u**3)
        target_pos = start_point + (end_point - start_point) * ratio
    else:
        target_pos = end_point

    ik_solution = dynamics.ik_solver(target_pos)

    # FIX: Removed the if/else switch based on X.
    # Index 0 corresponds to "Elbow Up" (Mountain shape).
    # We use Index 0 for the entire path to ensure continuity.
    target_angles = np.array([ik_solution[0][0] , ik_solution[1][0]])

    if first_run:
        filtered_angles = target_angles
        first_run = False

    diff = np.abs(target_angles - filtered_angles)
    max_diff = np.max(diff)

    if max_diff > 0.5: 
        alpha = 0.02   
    else:
        alpha = 0.8
    
    filtered_angles = alpha * target_angles + (1 - alpha) * filtered_angles

    for i, joint_id in enumerate(joint_ids):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=filtered_angles[i],
            force=arm_params.torque_limits[i],    # Enforce Torque Limit
            maxVelocity=arm_params.dq_max[i]      # Enforce Velocity Limit
        )

    p.stepSimulation()

    # Trace Logic
    current_angles = [p.getJointState(robot_id, joint_id)[0] for joint_id in joint_ids]
    ee_pos_rel = dynamics.forward_kinematics(np.array(current_angles))
    curr_ee_3d = [ee_pos_rel[0], 0, ee_pos_rel[1] + arm_params.start_pos[2]]

    if prev_ee_pos is not None:
        p.addUserDebugLine(prev_ee_pos, curr_ee_3d, lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
    
    prev_ee_pos = curr_ee_3d

    if time.time() - last_print_time >= 0.1:
        ee_pos = dynamics.forward_kinematics(np.array(current_angles))
        ee_pos[1] += arm_params.start_pos[2]
        joint_speeds = [p.getJointState(robot_id, joint_id)[1] for joint_id in joint_ids]
        print("-----")
        print(f"Status: {'MOVING' if elapsed < duration else 'HOLDING'}")
        print(f"Elapsed time: {elapsed:.2f} s")
        print(f"Target position: {target_pos}")
        print(f"Current joint angles: {current_angles}")
        print(f"Current end-effector position: {ee_pos}")
        print(f"Joint speeds: {joint_speeds}")
        last_print_time = time.time()
    
    time.sleep(1./240.)