import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import os

class ArmParameters:
    # 3-DOF Simulation Parameters
    l1: float = 1.0       # Length of Link 1
    l2: float = 1.0       # Length of Link 2
    base_z_offset: float = 0.06 # Height of shoulder joint relative to rail (from URDF)
    
    start_pos : np.ndarray = np.array([0, 0, 0.1]) 
    start_orientation : np.ndarray = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    
    # Rail Limits (from URDF)
    rail_min: float = -2.0
    rail_max: float = 2.0
    
    # Redundancy Strategy: 
    # How far in front of the slider should the target ideally be?
    ideal_reach_x: float = 1.0 
    
    # Motor Limits
    torque_limits: np.ndarray = np.array([500.0, 200.0, 200.0]) # [Slider, Shoulder, Elbow]
    dq_max: np.ndarray = np.array([2.0, 5.0, 5.0]) 
    
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
        
        # Calculate relative to the slider
        x_rel = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        z_rel = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        
        # Add Slider offset and Base Height
        x_world = slider_pos + x_rel
        z_world = z_rel + self.params.base_z_offset
        
        return np.array([x_world, z_world])
    
    def ik_solver(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Returns [slider_pos, q1, q2]
        """
        l1, l2 = self.params.l1, self.params.l2
        target_x, target_z = target_pos

        base_height = self.params.base_z_offset

        # --- 1. RESOLVE REDUNDANCY (Calculate Slider) ---
        # We attempt to place the slider such that the target is at 'ideal_reach_x'
        desired_slider = target_x - self.params.ideal_reach_x
        
        # Constraint: Clamp slider to physical rail limits
        slider_pos = np.clip(desired_slider, self.params.rail_min, self.params.rail_max)

        slider_pos = [slider_pos,slider_pos]
        
        # --- 2. TRANSFORM TO SHOULDER FRAME ---
        # Now that slider is fixed, find target relative to the shoulder joint
        x_rel = target_x - slider_pos[0]
        z_rel = target_z - base_height

        if z_rel < 0:
            z_rel = 0  # Restrict to upper half-plane

        # --- 3. STANDARD 2-LINK IK ---
        
        # Reach Safety
        dist_sq = x_rel**2 + z_rel**2
        dist = np.sqrt(dist_sq)
        max_reach = l1 + l2
        
        # If target is out of reach, stretch arm towards it
        if dist > max_reach:
            scale = max_reach / dist
            x_rel *= scale
            z_rel *= scale
        
        # Prevent singularity at shoulder (0,0)
        if dist < 1e-6:
            return np.array([slider_pos, 0.0, 0.0])

        if z_rel < 0:
            z_rel = 0  # Restrict to upper half-plane

        # Law of Cosines for Elbow (q2)
        cos_q2 = (x_rel**2 + z_rel**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0) # Numerical safety
        
        # Calculate q2 magnitude
        q2_mag = np.arccos(cos_q2)
        
        # CONFIGURATION: Elbow Up (Negative q2)
        # This keeps the arm "above" the target line, avoiding the rail
        q2 = [-q2_mag,q2_mag]
        
        # Calculate Shoulder (q1)
        k1 = l1 + l2 * np.cos(q2)
        k2 = l2 * np.sin(q2)
        
        q1 = np.arctan2(z_rel, x_rel) - np.arctan2(k2, k1)
        
        # Normalize q1 to [-PI, PI]
        q1 = np.clip(q1, 0, np.pi)
        
        return np.array([slider_pos, q1, q2])

# --- MAIN SIMULATION ---
if __name__ == "__main__":
    arm_params = ArmParameters()
    
    # 1. Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")

    # 2. Load URDF (Robust Path Finding)
    urdf_filename = "3dof_planar_slider.urdf" # Make sure this matches your file name
    urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/', urdf_filename))
    if not os.path.exists(urdf_path):
        urdf_path = os.path.abspath(os.path.join(os.getcwd(), urdf_filename))
    if not os.path.exists(urdf_path):
        # Fallback
        urdf_path = urdf_filename 

    robot_id = p.loadURDF(
        urdf_path,
        basePosition=arm_params.start_pos,
        baseOrientation=arm_params.start_orientation,
        useFixedBase=True # 3DOF Rail is fixed to world
    )

    # 3. Identify Joints
    joint_map = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[1].decode('utf-8')
        joint_map[name] = j
        
    # Ensure we found the right joints
    try:
        target_joint_ids = [
            joint_map['slider_joint'], 
            joint_map['baseHinge'], 
            joint_map['interArm']
        ]
    except KeyError as e:
        print(f"Error: Could not find joint {e}. Check your URDF joint names.")
        exit()

    # 4. Setup Target Sliders
    # Rail is +/- 2m. Arm reaches ~2m. Total X range approx +/- 4.0m
    slid_target_x = p.addUserDebugParameter("Target X", -4.0, 4.0, 1.5)
    slid_target_z = p.addUserDebugParameter("Target Z", 0.0, 2.0, 0.5)

    dynamics = ArmDynamics(arm_params)
    
    # Smoothing Variables
    filtered_cmds = np.zeros(3) 
    first_run = True
    prev_ee_pos = None
    last_print_time = time.time()

    print("Starting 3-DOF Accurate IK Solver...")

    while True:
        # 1. Read Inputs
        tx = p.readUserDebugParameter(slid_target_x)
        tz = p.readUserDebugParameter(slid_target_z)
        target_pos = np.array([tx, tz])
        current_pos = p.getLinkState(robot_id, target_joint_ids[-1])[0][:2]  # X,Z of end-effector
        # 2. Solve Inverse Kinematics
        # Returns [slider_pos, q1, q2]
        ik_solution = dynamics.ik_solver(target_pos)
        # if target_pos[0] >= current_pos[0]:
        target_cmds = np.array([ik_solution[0][0] , ik_solution[1][0] , ik_solution[2][0] ])  # [slider, shoulder, elbow]
        # else:
          #  target_cmds = np.array([ik_solution[0][1] , ik_solution[1][1] , ik_solution[2][1] ])  # [slider, shoulder, elbow]

        # 3. Apply Low-Pass Filter (Smoothing)
        if first_run:
            filtered_cmds = target_cmds
            first_run = False

        diff = np.abs(target_cmds - filtered_cmds)
        max_diff = np.max(diff)

        # Dynamic smoothing factor
        if max_diff > 0.5: 
            alpha = 0.02   # Move slow if big jump (e.g. wrapping or rail switch)
        else:
            alpha = 0.8    # Move fast for small adjustments
        
        filtered_cmds = alpha * target_cmds + (1 - alpha) * filtered_cmds

        # 4. Send Motor Commands
        for i, j_id in enumerate(target_joint_ids):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=j_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=filtered_cmds[i],
                force=arm_params.torque_limits[i],
                maxVelocity=arm_params.dq_max[i]
            )

        p.stepSimulation()

        # 5. Visual Debugging (Green Line Trace)
        current_states = [p.getJointState(robot_id, j)[0] for j in target_joint_ids]
        actual_pos = np.array(current_states)
        ee_pos = dynamics.forward_kinematics(actual_pos) 
        
        # Draw line (World Z is Up)
        curr_draw_pos = [ee_pos[0], 0, ee_pos[1]] 

        if prev_ee_pos is not None:
            p.addUserDebugLine(prev_ee_pos, curr_draw_pos, lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
        prev_ee_pos = curr_draw_pos

        if time.time() - last_print_time >= 0.2:
            # Verify accuracy
            error = np.linalg.norm(target_pos - ee_pos)
            print(f"Tgt: {target_pos} | Act: {ee_pos} | Err: {error:.4f} | Slider: {current_states[0]:.2f}")
            last_print_time = time.time()
        
        time.sleep(1./240.)