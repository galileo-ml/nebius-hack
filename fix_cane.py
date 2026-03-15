import numpy as np

# Hand position
H = np.array([0.1, -0.18, 0.98])
# Floor point
F = np.array([1.0, -0.18, 0.0])

V = F - H
length = np.linalg.norm(V)
half_length = length / 2
dir_vec = V / length

# We want local +Z to align with dir_vec
# dir_vec = (sin(theta), 0, cos(theta))
theta = np.arctan2(dir_vec[0], dir_vec[2])
theta_deg = np.degrees(theta)

center = (H + F) / 2

# Two halves
# Top half (white) is closer to hand
# Bottom half (red) is closer to floor
# Since dir_vec points from Hand to Floor, Hand is at center - half_length * dir_vec
# Floor is at center + half_length * dir_vec

# So top half center is center - (half_length / 2) * dir_vec
top_center = center - (half_length / 2) * dir_vec

# Bottom half center is center + (half_length / 2) * dir_vec
bottom_center = center + (half_length / 2) * dir_vec

capsule_half_length = half_length / 2

print(f"Total length: {length:.3f}")
print(f"Capsule half length: {capsule_half_length:.3f}")
print(f"Euler Y (pitch): {theta_deg:.3f}")
print(f"Top center: {top_center}")
print(f"Bottom center: {bottom_center}")

xml_top = f'<geom type="capsule" size="0.012 {capsule_half_length:.3f}" pos="{top_center[0]:.3f} {top_center[1]:.3f} {top_center[2]:.3f}" euler="0 {theta_deg:.3f} 0" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>'
xml_bottom = f'<geom type="capsule" size="0.012 {capsule_half_length:.3f}" pos="{bottom_center[0]:.3f} {bottom_center[1]:.3f} {bottom_center[2]:.3f}" euler="0 {theta_deg:.3f} 0" rgba="0.8 0.1 0.1 1" contype="0" conaffinity="0"/>'

print("XML:")
print("      <!-- white cane (in right hand) - top half white, bottom half red -->")
print("      " + xml_top)
print("      " + xml_bottom)
