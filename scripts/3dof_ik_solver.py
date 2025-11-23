import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import os

class ArmParameters:
    # Parameters for 3-DOF PRR (Slider + 2-Link)
    l1: float = 1.0  # Link 1 Length
    l2: float = 1.0  # Link 2 Length
    base_z_offset: float = 0.06 # Height of shoulder relative to rail
    
    start_pos : np.ndarray = np.array([0, 0, 0.1]) 
    start_orientation : np.ndarray = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    
    # Rail Constraints
    rail_min: float = -2.0
    rail_max: float = 2.0
    ideal_reach_x: float = 1.0 # Strategy: Keep arm 1.0m in front of slider
    
    # Limits [Slider, Shoulder, Elbow]
    dq_max: np.ndarray = np.array([4.0, 5.0, 5.0])  
    torque_limits: np.ndarray = np.array([500.0, 200.0, 200.0])  
    
    control_mode: str = "POSITION"

class ArmDynamics:
    def __init__(self, params: ArmParameters):
        self.params = params

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        # q = [slider_pos, q1, q2]
        slider_pos = q[0]
        q1 = q[1]
        q2 = q[2]
        
        l1, l2 = self.params.l1, self.params.l2
        
        # Relative to slider
        x_rel = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        z_rel = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        
        # World Frame (Relative to Rail Origin)
        x = slider_pos + x_rel
        z = z_rel + self.params.base_z_offset
        
        return np.array([x, z])
    
    def ik_solver(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Returns [slider_pos, q1, q2]
        """
        l1, l2 = self.params.l1, self.params.l2
        target_x, target_z = target_pos

        # --- 1. RESOLVE REDUNDANCY (Calculate Slider) ---
        desired_slider = target_x - self.params.ideal_reach_x
        slider_pos = np.clip(desired_slider, self.params.rail_min, self.params.rail_max)
        
        # --- 2. TRANSFORM TO SHOULDER FRAME ---
        x_rel = target_x - slider_pos
        z_rel = target_z - self.params.base_z_offset
        
        # --- 3. STANDARD 2-LINK IK ---
        dist_sq = x_rel**2 + z_rel**2
        dist = np.sqrt(dist_sq)
        max_reach = l1 + l2
        
        # Reach Safety
        if dist > max_reach:
            scale = max_reach / dist
            x_rel *= scale
            z_rel *= scale
        
        # Prevent singularity
        if dist < 1e-6:
            return np.array([slider_pos, 0.0, 0.0])

        # Law of Cosines for Elbow (q2)
        cos_q2 = (x_rel**2 + z_rel**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)
        
        q2_mag = np.arccos(cos_q2)
        q2 = -q2_mag # Elbow Up Config
        
        # Shoulder (q1)
        k1 = l1 + l2 * np.cos(q2)
        k2 = l2 * np.sin(q2)
        
        q1 = np.arctan2(z_rel, x_rel) - np.arctan2(k2, k1)
        
        # Normalize
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([slider_pos, q1, q2])

# --- SETUP ---
arm_params = ArmParameters()

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# Robust Path Loading for 3-DOF URDF
urdf_filename = "3dof_planar_slider.urdf"
urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/', urdf_filename))
if not os.path.exists(urdf_path):
    urdf_path = os.path.abspath(os.path.join(os.getcwd(), urdf_filename))
if not os.path.exists(urdf_path):
    urdf_path = urdf_filename 

if not os.path.exists(urdf_path):
    raise FileNotFoundError(f"URDF file not found: {urdf_path}")

print(f"Loading URDF: {urdf_path}")
robot_id = p.loadURDF(
    urdf_path,
    basePosition=arm_params.start_pos,
    baseOrientation=arm_params.start_orientation,
    useFixedBase=True
)

# Identify Joints
joint_map = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    name = info[1].decode('utf-8')
    joint_map[name] = j

try:
    # Order must match IK output: [Slider, Shoulder, Elbow]
    joint_ids = [
        joint_map['slider_joint'], 
        joint_map['baseHinge'], 
        joint_map['interArm']
    ]
except KeyError as e:
    print(f"Error: Joint {e} not found in URDF.")
    exit()

# --- TRAJECTORY DEFINITION ---
start_point = np.array([1.221, 1.568])
end_point   = np.array([-2.8, 0.579])

# Draw trajectory line (World Z is Up)
p.addUserDebugLine(
    [start_point[0], 0, start_point[1]], 
    [end_point[0], 0, end_point[1]], 
    lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0
)

dynamics = ArmDynamics(arm_params)

# --- INITIALIZATION ---
# Calculate initial joint positions for the start point
initial_joints = dynamics.ik_solver(start_point)

# Teleport robot to start configuration
for i, joint_id in enumerate(joint_ids):
    p.resetJointState(robot_id, joint_id, initial_joints[i])

# Initialize smoothing filter with the correct starting values
filtered_cmds = initial_joints.copy()

last_print_time = time.time()
start_time = time.time()
duration = 2.0 

print(f"Starting 3-DOF Cubic Trajectory: {start_point} -> {end_point}")

prev_ee_pos = None 

# --- SIMULATION LOOP ---
while True:
    now = time.time()
    elapsed = now - start_time
    
    # 1. Cubic Spline Generation
    if elapsed < duration:
        u = elapsed / duration
        # Smoothstep: 3u^2 - 2u^3
        ratio = 3 * (u**2) - 2 * (u**3)
        target_pos = start_point + (end_point - start_point) * ratio
    else:
        target_pos = end_point

    # 2. IK Solver
    # Returns [slider, q1, q2]
    target_cmds = dynamics.ik_solver(target_pos)

    # 3. Smoothing (Low Pass Filter)
    diff = np.abs(target_cmds - filtered_cmds)
    max_diff = np.max(diff)

    if max_diff > 0.5: 
        alpha = 0.02   
    else:
        alpha = 0.8
    
    filtered_cmds = alpha * target_cmds + (1 - alpha) * filtered_cmds

    # 4. Send Commands
    for i, joint_id in enumerate(joint_ids):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=filtered_cmds[i],
            force=arm_params.torque_limits[i],
            maxVelocity=arm_params.dq_max[i]
        )

    p.stepSimulation()

    # 5. Visual Tracing (Green Line)
    current_states = [p.getJointState(robot_id, j)[0] for j in joint_ids]
    ee_pos_rel = dynamics.forward_kinematics(np.array(current_states))
    
    # Map to World Frame for Drawing (X, 0, Z)
    # Note: base_z_offset was already added in forward_kinematics relative to rail.
    # We just need to add the Rail's World Z position.
    curr_ee_3d = [ee_pos_rel[0], 0, ee_pos_rel[1] + arm_params.start_pos[2]]

    if prev_ee_pos is not None:
        # Draw a persistent line segment
        p.addUserDebugLine(prev_ee_pos, curr_ee_3d, lineColorRGB=[0, 1, 0], lineWidth=4, lifeTime=0)
    
    prev_ee_pos = curr_ee_3d

    if time.time() - last_print_time >= 0.1:
        joint_speeds = [p.getJointState(robot_id, j)[1] for j in joint_ids]
        print("-----")
        print(f"Status: {'MOVING' if elapsed < duration else 'HOLDING'}")
        print(f"Elapsed time: {elapsed:.2f} s")
        print(f"Target: {target_pos}")
        print(f"Current EE (Rel): {ee_pos_rel}")
        print(f"Slider: {current_states[0]:.2f}")
        last_print_time = time.time()
    
    time.sleep(1./240.)