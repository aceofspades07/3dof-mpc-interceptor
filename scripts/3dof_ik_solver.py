import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import os
import matplotlib.pyplot as plt

class ArmParameters:
    # Parameters for 3-DOF PRR (Slider + 2-Link)
    l1: float = 1.0  
    l2: float = 1.0  
    base_z_offset: float = 0.06
    
    start_pos : np.ndarray = np.array([0, 0, 0.1]) 
    start_orientation : np.ndarray = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    
    # Rail Constraints
    rail_min: float = -2.0
    rail_max: float = 2.0
    ideal_reach_x: float = 1.0 
    
    # Limits [Slider, Shoulder, Elbow]
    dq_max: np.ndarray = np.array([2.0, 5.0, 5.0])  
    torque_limits: np.ndarray = np.array([500.0, 200.0, 200.0])  
    
    control_mode: str = "POSITION"

class ArmDynamics:
    def __init__(self, params: ArmParameters):
        self.params = params

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        slider_pos = q[0]
        q1 = q[1]
        q2 = q[2]
        
        l1, l2 = self.params.l1, self.params.l2
        
        x_rel = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        z_rel = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        
        x = slider_pos + x_rel
        z = z_rel + self.params.base_z_offset
        
        return np.array([x, z])
    
    def ik_solver(self, target_pos: np.ndarray) -> np.ndarray:
        l1, l2 = self.params.l1, self.params.l2
        target_x, target_z = target_pos

        desired_slider = target_x - self.params.ideal_reach_x
        slider_pos = np.clip(desired_slider, self.params.rail_min, self.params.rail_max)
        
        x_rel = target_x - slider_pos
        z_rel = target_z - self.params.base_z_offset
        
        dist_sq = x_rel**2 + z_rel**2
        dist = np.sqrt(dist_sq)
        max_reach = l1 + l2
        
        if dist > max_reach:
            scale = max_reach / dist
            x_rel *= scale
            z_rel *= scale
        
        if dist < 1e-6:
            return np.array([slider_pos, 0.0, 0.0])

        cos_q2 = (x_rel**2 + z_rel**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)
        
        q2_mag = np.arccos(cos_q2)
        q2 = -q2_mag # Elbow Up Config
        
        k1 = l1 + l2 * np.cos(q2)
        k2 = l2 * np.sin(q2)
        
        q1 = np.arctan2(z_rel, x_rel) - np.arctan2(k2, k1)
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([slider_pos, q1, q2])

# --- SETUP ---
arm_params = ArmParameters()

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# Robust Path Loading
urdf_filename = "3dof_planar_slider.urdf"
urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/', urdf_filename))
if not os.path.exists(urdf_path):
    urdf_path = os.path.abspath(os.path.join(os.getcwd(), urdf_filename))
if not os.path.exists(urdf_path):
    urdf_path = urdf_filename 

if not os.path.exists(urdf_path):
    raise FileNotFoundError(f"URDF file not found: {urdf_path}")

robot_id = p.loadURDF(
    urdf_path,
    basePosition=arm_params.start_pos,
    baseOrientation=arm_params.start_orientation,
    useFixedBase=True
)

joint_map = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    name = info[1].decode('utf-8')
    joint_map[name] = j

try:
    joint_ids = [
        joint_map['slider_joint'], 
        joint_map['baseHinge'], 
        joint_map['interArm']
    ]
except KeyError as e:
    print(f"Error: Joint {e} not found.")
    exit()

# --- TRAJECTORY DEFINITION ---
start_point = np.array([1.221, 1.568])
end_point   = np.array([-0.421, 0.579])

dynamics = ArmDynamics(arm_params)

# Initialization
initial_joints = dynamics.ik_solver(start_point)
for i, joint_id in enumerate(joint_ids):
    p.resetJointState(robot_id, joint_id, initial_joints[i])

filtered_cmds = initial_joints.copy()
last_print_time = time.time()
start_time = time.time()
duration = 1.5 

print(f"Starting Trajectory for Metrics Collection...")

prev_ee_pos = None 

# DATA LOGGING ARRAYS
log_time = []
log_target_x = []
log_target_z = []
log_actual_x = []
log_actual_z = []
log_slider = []
log_velocity_shoulder = []
log_velocity_elbow = []
log_error = []

# --- SIMULATION LOOP ---
while True:
    now = time.time()
    elapsed = now - start_time
    
    # Auto-stop after trajectory + 1.5 second hold
    if elapsed > duration + 1.5:
        print("Trajectory Complete. Generating Report Graphs...")
        break

    # 1. Generate Target (Cubic Spline)
    if elapsed < duration:
        u = elapsed / duration
        ratio = 3 * (u**2) - 2 * (u**3)
        target_pos = start_point + (end_point - start_point) * ratio
    else:
        target_pos = end_point

    # 2. IK & Control
    target_cmds = dynamics.ik_solver(target_pos)
    
    # Smoothing
    diff = np.abs(target_cmds - filtered_cmds)
    max_diff = np.max(diff)
    alpha = 0.02 if max_diff > 0.5 else 0.8
    filtered_cmds = alpha * target_cmds + (1 - alpha) * filtered_cmds

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

    # 3. Get Real State
    current_states = [p.getJointState(robot_id, j)[0] for j in joint_ids]
    joint_vels = [p.getJointState(robot_id, j)[1] for j in joint_ids]
    ee_pos_rel = dynamics.forward_kinematics(np.array(current_states))
    curr_ee_3d = [ee_pos_rel[0], 0, ee_pos_rel[1]]

    # 4. VISUAL: Draw Green Trail (Persistent)
    if prev_ee_pos is not None:
        p.addUserDebugLine(prev_ee_pos, curr_ee_3d, lineColorRGB=[0, 1, 0], lineWidth=3, lifeTime=0)
    prev_ee_pos = curr_ee_3d

    # 5. LOG DATA (For Plots)
    log_time.append(elapsed)
    log_target_x.append(target_pos[0])
    log_target_z.append(target_pos[1])
    log_actual_x.append(ee_pos_rel[0])
    log_actual_z.append(ee_pos_rel[1])
    log_slider.append(current_states[0])
    log_velocity_shoulder.append(joint_vels[1])
    log_velocity_elbow.append(joint_vels[2])
    
    # Calc error vs FINAL endpoint (to detect when we arrive)
    err_to_final = np.linalg.norm(end_point - ee_pos_rel)
    log_error.append(err_to_final)

    time.sleep(1./240.)



# --- METRICS CALCULATION ---
TARGET_TOLERANCE = 0.005 # 5mm tolerance zone

# Find first time error dropped below tolerance
time_to_target = None
for t, err in zip(log_time, log_error):
    if err < TARGET_TOLERANCE:
        time_to_target = t
        break

rmse = np.sqrt(np.mean(np.array(log_error)**2))

print(f"\n--- FINAL METRICS ---")
print(f"Total Duration Command: {duration} s")
if time_to_target:
    print(f"Time to Reach Target (<5mm): {time_to_target:.3f} s")
else:
    print("Target not reached within tolerance.")
print(f"Final Error: {log_error[-1]*1000:.2f} mm")
print(f"Peak Shoulder Vel: {np.max(np.abs(log_velocity_shoulder)):.2f} rad/s")
'''
# --- PLOTTING GRAPHS FOR REPORT ---
plt.figure(figsize=(12, 10))

# Graph 1: Trajectory Path (X-Z)
plt.subplot(2, 2, 1)
plt.plot(log_target_x, log_target_z, 'r--', label='Spline Path')
plt.plot(log_actual_x, log_actual_z, 'g-', label='Robot Path')
plt.scatter([end_point[0]], [end_point[1]], c='blue', marker='x', s=100, label='Target')
plt.title('Trajectory Tracking')
plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.legend()
plt.grid(True)

# Graph 2: Distance to Final Target
plt.subplot(2, 2, 2)
plt.plot(log_time, np.array(log_error)*1000, 'b-')
plt.axhline(y=TARGET_TOLERANCE*1000, color='g', linestyle=':', label='5mm Zone')
if time_to_target:
    plt.axvline(x=time_to_target, color='r', linestyle='--', label=f'Reached: {time_to_target:.2f}s')
plt.title('Distance to Final Target')
plt.xlabel('Time (s)')
plt.ylabel('Distance (mm)')
plt.legend()
plt.grid(True)

# Graph 3: Joint Velocities
plt.subplot(2, 2, 3)
plt.plot(log_time, log_velocity_shoulder, label='Shoulder')
plt.plot(log_time, log_velocity_elbow, label='Elbow')
plt.title('Joint Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (rad/s)')
plt.legend()
plt.grid(True)

# Graph 4: Slider Usage
plt.subplot(2, 2, 4)
plt.plot(log_time, log_slider, 'm-')
plt.title('Slider Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

plt.tight_layout()
plt.show()'''