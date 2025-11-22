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
    q_min: np.ndarray = np.array([0, 0])
    q_max: np.ndarray = np.array([2*np.pi, 2*np.pi])
    dq_max: np.ndarray = np.array([2.0, 2.0])  # rad/s
    ddq_max: np.ndarray = np.array([5.0, 5.0])  # rad/s^2

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
        param_ids.append(p.addUserDebugParameter(joint_name, -3.14, 3.14, 0))

# 6. Simulation Loop
while True:
    # Read slider values
    target_pos_1 = p.readUserDebugParameter(param_ids[0]) # Replace this with the calculated target from high-level controller
    target_pos_2 = p.readUserDebugParameter(param_ids[1]) # Replace this with the calculated target from high-level controller

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

