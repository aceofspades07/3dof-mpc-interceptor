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

        q2 = np.arctan2(sin_q2, cos_q2)

        k1 = l1 + l2 * cos_q2
        k2 = l2 * sin_q2
        q1 = np.arctan2(z, x) - np.arctan2(k2, k1)

        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi # do not let q1 go beyond -pi to pi

        return np.array([q1, q2])
    

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
start_pos = [0, 0, 0.1] # Start 0.1 meters in the air 
start_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])

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


arm_params = ArmParameters()
max_reach = arm_params.l1 + arm_params.l2
min_reach = abs(arm_params.l1 - arm_params.l2)
slid_target_x = p.addUserDebugParameter("Target X", -1*max_reach, max_reach, max_reach/2)
slid_target_z = p.addUserDebugParameter("Target Z", -1*max_reach, max_reach, max_reach/2)

# 6. Simulation Loop
while True:
    # Read slider values
    """
    target_pos_1 = p.readUserDebugParameter(param_ids[0]) # Replace this with the calculated target from high-level controller
    target_pos_2 = p.readUserDebugParameter(param_ids[1]) # Replace this with the calculated target from high-level controller
    """
    tx = p.readUserDebugParameter(slid_target_x)
    tz = p.readUserDebugParameter(slid_target_z)

    params = ArmParameters()
    dynamics = ArmDynamics(params)
    ik_angles =  dynamics.ik_solver((tx,tz))

    if ik_angles is not None:
        target_pos_1 = ik_angles[0]
        target_pos_2 = ik_angles[1]

        p.setJointMotorControlArray(
            robot_id,
            joint_ids,
            p.POSITION_CONTROL,
            targetPositions=[target_pos_1, target_pos_2],
            forces=[100, 100] # Torque limit
        )
    else:
        print("Target position unreachable")
        pass

    # Apply position control
    p.setJointMotorControlArray(
        robot_id,
        joint_ids,
        p.POSITION_CONTROL,
        targetPositions=[target_pos_1, target_pos_2]
    )
    

    # Apply velocity control
    """
    Kp = [5.0,5.0]  # Proportional gain
    p.setJointMotorControlArray(
        robot_id,
        joint_ids,
        p.VELOCITY_CONTROL,
        targetVelocities=[Kp[0] * (target_pos_1 - p.getJointState(robot_id, joint_ids[0])[0]),
                          Kp[1] * (target_pos_2 - p.getJointState(robot_id, joint_ids[1])[0])]
    )
    """

    p.stepSimulation()
    time.sleep(1./240.)




