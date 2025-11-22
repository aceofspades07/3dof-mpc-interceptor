import pybullet as p
import numpy as np
import pybullet_data
import time
import os
import random

def main():
	p.connect(p.GUI)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	p.resetSimulation()
	p.setGravity(0, 0, -9.81)

	urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/2linkarm.urdf'))
	if not os.path.exists(urdf_path):
		urdf_path = os.path.abspath(os.path.join(os.getcwd(), 'urdf/2linkarm.urdf'))
	if not os.path.exists(urdf_path):
		raise FileNotFoundError(f"URDF file not found: {urdf_path}")

	# Spawn the robot arm in the yz plane upright by rotating -90 degrees about the y-axis
	# Quaternion for -90 deg about y-axis: [0, -0.7071, 0, 0.7071]
	base_orientation = [0, -0.7071, 0, 0.7071]
	robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], baseOrientation=base_orientation, useFixedBase=True)

	print(f"Spawned robot with ID: {robot_id}")

	# Set random joint angles for continuous joints
	num_joints = p.getNumJoints(robot_id)
	JOINT_REVOLUTE = p.JOINT_REVOLUTE
	JOINT_CONTINUOUS = 4  # PyBullet does not define this constant, but 4 is the value for continuous joints
	for joint_idx in range(num_joints):
		joint_info = p.getJointInfo(robot_id, joint_idx)
		joint_type = joint_info[2]
		# Only set for continuous or revolute joints
		if joint_type in [JOINT_REVOLUTE, JOINT_CONTINUOUS]:
			lower_limit = joint_info[8]
			upper_limit = joint_info[9]
			if joint_type == JOINT_CONTINUOUS:
				angle = random.uniform(-np.pi, np.pi)
			else:
				angle = random.uniform(lower_limit, upper_limit)
			p.resetJointState(robot_id, joint_idx, angle)
			print(f"Set joint {joint_info[1].decode()} to {angle:.2f} radians")

	# Run simulation for a while
	for _ in range(10000):
		p.stepSimulation()
		time.sleep(1./240.)

	p.disconnect()

if __name__ == "__main__":
	main()
