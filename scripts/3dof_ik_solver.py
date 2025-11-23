import numpy as np

class IKParameters:
    # Physical Dimensions (Match your URDF)
    l1: float = 1.0       # Length of Link 1
    l2: float = 1.0       # Length of Link 2
    base_height: float = 0.06  # Z-height of the shoulder relative to rail
    
    # Rail Constraints
    rail_min: float = -2.0
    rail_max: float = 2.0
    
    # Strategy: How far in front of the slider do we want to reach?
    # Setting this to 1.0 means the slider tries to stay 1 meter behind the target X.
    ideal_reach_x: float = 1.0 

def solve_3dof_ik(target_x, target_z, params=IKParameters()):
    """
    Calculates joint positions [slider, shoulder, elbow] for a given (x, z) target.
    Resolves redundancy by attempting to keep the arm at 'ideal_reach_x'.
    """
    
    # --- STEP 1: RESOLVE REDUNDANCY (Calculate Slider) ---
    # We want: Target_X = Slider_Pos + Ideal_Reach
    # Therefore: Desired_Slider = Target_X - Ideal_Reach
    desired_slider = target_x - params.ideal_reach_x
    
    # Constraint: The slider cannot leave the rail.
    # If target is too far left/right, the slider hits the limit and stops.
    slider_pos = np.clip(desired_slider, params.rail_min, params.rail_max)
    
    # --- STEP 2: TRANSFORM TO SHOULDER FRAME ---
    # Now that the base is fixed at 'slider_pos', calculate the target
    # coordinates relative to the shoulder joint.
    x_rel = target_x - slider_pos
    z_rel = target_z - params.base_height
    
    # --- STEP 3: STANDARD 2-LINK IK ---
    
    # A. Check Reachability
    dist_sq = x_rel**2 + z_rel**2
    dist = np.sqrt(dist_sq)
    max_reach = params.l1 + params.l2
    
    # Safety: If target is out of reach, stretch arm towards it
    if dist > max_reach:
        scale = max_reach / dist
        x_rel *= scale
        z_rel *= scale
    
    # B. Calculate Elbow Angle (q2) using Law of Cosines
    # D^2 = L1^2 + L2^2 - 2*L1*L2*cos(pi - q2)
    # cos(q2) = (D^2 - L1^2 - L2^2) / (2*L1*L2)
    cos_q2 = (x_rel**2 + z_rel**2 - params.l1**2 - params.l2**2) / (2 * params.l1 * params.l2)
    
    # Numerical safety clip (floating point errors can make cos_q2 > 1.0)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    
    # Calculate q2 Magnitude (0 to PI)
    q2_mag = np.arccos(cos_q2)
    
    # Configuration Choice: ELBOW UP
    # For standard planar arms, negative q2 usually corresponds to "Elbow Up"
    q2 = -q2_mag
    
    # C. Calculate Shoulder Angle (q1)
    # q1 = angle_to_target - angle_inside_triangle
    k1 = params.l1 + params.l2 * np.cos(q2)
    k2 = params.l2 * np.sin(q2)
    
    q1 = np.arctan2(z_rel, x_rel) - np.arctan2(k2, k1)
    
    # Normalize q1 to [-PI, PI] to stay within standard limits
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([slider_pos, q1, q2])

# --- Usage Example ---
if __name__ == "__main__":
    # Example 1: Target is comfortably in the middle
    # Target X=1.5. Ideal reach is 1.0. Slider should go to 0.5.
    tgt = (1.5, 0.5)
    joints = solve_3dof_ik(tgt[0], tgt[1])
    print(f"Target: {tgt}")
    print(f"  Slider:   {joints[0]:.3f} m")
    print(f"  Shoulder: {joints[1]:.3f} rad")
    print(f"  Elbow:    {joints[2]:.3f} rad")

    print("-" * 30)

    # Example 2: Target is far to the left (beyond rail limit)
    # Target X=-4.0. Slider limit is -2.0.
    # Slider should stay at -2.0, and arm should reach backward.
    tgt = (-4.0, 0.5)
    joints = solve_3dof_ik(tgt[0], tgt[1])
    print(f"Target: {tgt}")
    print(f"  Slider:   {joints[0]:.3f} m (Hit Limit)")
    print(f"  Shoulder: {joints[1]:.3f} rad")
    print(f"  Elbow:    {joints[2]:.3f} rad")