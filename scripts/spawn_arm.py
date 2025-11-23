import pybullet as p
import numpy as np
import pybullet_data
import time
import os
import random
import math

def main():
	p.connect(p.GUI)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	p.resetSimulation()
	p.setGravity(0, 0, -9.81)

	# Use same spawn position and orientation as 2link_ik_solver.py
	
	start_pos = [0, 0, 0.1]
	start_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])

	urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../urdf/2linkarm.urdf'))
	if not os.path.exists(urdf_path):
		urdf_path = os.path.abspath(os.path.join(os.getcwd(), 'urdf/2linkarm.urdf'))
	if not os.path.exists(urdf_path):
		raise FileNotFoundError(f"URDF file not found: {urdf_path}")

	robot_id = p.loadURDF(
		urdf_path,
		basePosition=start_pos,
		baseOrientation=start_orientation,
		useFixedBase=True
	)

	print(f"Spawned robot with ID: {robot_id}")

	# Setup sliders for joint control
	joint_ids = []
	slider_ids = []
	for j in range(p.getNumJoints(robot_id)):
		info = p.getJointInfo(robot_id, j)
		joint_type = info[2]
		joint_name = info[1].decode("utf-8")
		if joint_type == p.JOINT_REVOLUTE:
			joint_ids.append(j)
			lower = info[8]
			upper = info[9]
			slider_id = p.addUserDebugParameter(f"{joint_name}", lower, upper, 0.0)
			slider_ids.append(slider_id)

	# Simulation loop with slider control
	while True:
		target_angles = [p.readUserDebugParameter(slider_id) for slider_id in slider_ids]
		p.setJointMotorControlArray(
			robot_id,
			joint_ids,
			p.POSITION_CONTROL,
			targetPositions=target_angles
		)
		p.stepSimulation()
		time.sleep(1./240.)

if __name__ == "__main__":
	main()
