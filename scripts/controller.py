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
    dq_max: np.ndarray = np.array([2.0, 2.0])  # rad/s
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
        
        config = self.params.config

        # if x < 0:
        #     config = "ELBOW_DOWN"
        # elif x > 0:
        #     config = "ELBOW_UP"
        
        cos_q2 = (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)  # Numerical safety

        sin_q2 = np.sqrt(1 - cos_q2**2)
        sin_q2 = np.clip(sin_q2, -1.0, 1.0)  # Numerical safety

        q2_mag = np.arctan2(sin_q2, cos_q2)

        if config == "ELBOW_UP":
            q2 = -q2_mag # Elbow Up
        elif config == "ELBOW_DOWN":
            q2 = q2_mag # Elbow Down

        cos_q2 = np.cos(q2)
        sin_q2 = np.sin(q2)

        k1 = l1 + l2 * cos_q2
        k2 = l2 * sin_q2
        q1 = np.arctan2(z, x) - np.arctan2(k2, k1)

        # q1 = (q1 + np.pi) % (2 * np.pi) - np.pi # do not let q1 go beyond -pi to pi
        # if q1 < 0:
        #     q1 = 0

        return np.array([q1, q2])
    

arm_params = ArmParameters()

# 1. Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 2. Set up the environment
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# 3. Define the Spawn Orientation
# The URDF joints rotate around the Z-axis (0,0,1).
# To move in the World XZ plane, we rotate the robot -90 degrees around the X-axis.
# Euler angles are [Roll, Pitch, Yaw]
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
        # Add a slider for this joint
        #param_ids.append(p.addUserDebugParameter(joint_name, -3.14, 3.14, 0))

max_reach = arm_params.l1 + arm_params.l2
min_reach = abs(arm_params.l1 - arm_params.l2)
slid_target_x = p.addUserDebugParameter("Target X", -1*max_reach, max_reach, 1.8)
slid_target_z = p.addUserDebugParameter("Target Z", -1*max_reach, max_reach, 0.1)

# 6. Simulation Loop
last_print_time = time.time()
# Run forever (until user closes GUI / kills process)
while True:
    # Read slider values
    
    target_pos = p.readUserDebugParameter(slid_target_x) , p.readUserDebugParameter(slid_target_z) 

    dynamics = ArmDynamics(arm_params)
    target_angles =  dynamics.ik_solver(target_pos)

    if target_angles is not None:
        
        p.setJointMotorControlArray(
            robot_id,
            joint_ids,
            p.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=arm_params.torque_limits
        )
    else:
        print("Target position unreachable")
        pass

    # Apply position control

    if arm_params.control_mode == "POSITION":
        p.setJointMotorControlArray(
            robot_id,
            joint_ids,
            p.POSITION_CONTROL,
            targetPositions=target_angles
        )
    
    

    # Apply velocity control
    elif arm_params.control_mode == "VELOCITY":
        Kp = [5.0,5.0]  # Proportional gain
        p.setJointMotorControlArray(
            robot_id,
            joint_ids,
            p.VELOCITY_CONTROL,
            targetVelocities=[Kp[0] * (target_angles[0] - p.getJointState(robot_id, joint_ids[0])[0]),
                            Kp[1] * (target_angles[1] - p.getJointState(robot_id, joint_ids[1])[0])]
        )

    # Debugging
    current_angles = [p.getJointState(robot_id, joint_id)[0] for joint_id in joint_ids]
    # Print debug info at 1 Hz to reduce console spam
    if time.time() - last_print_time >= 0.3:
        ee_pos = dynamics.forward_kinematics(np.array(current_angles))
        print("-----")
        print(f"Target position: {target_pos}")
        print(f"Current joint angles: {current_angles}")
        print(f"Current end-effector position: {ee_pos}")
        last_print_time = time.time()
    
    p.stepSimulation()
    time.sleep(1./240.)




