"""
Print where the SO-101 gripper tip is for any set of joint angles.

Usage:
  python tip_check.py                    # q = 0 (all joints at 0)
  python tip_check.py 0 0 0 0 1.57       # 5 angles in radians:
                                         # shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).resolve().parent / "mujoco_menagerie" / "robotstudio_so101" / "scene.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

q = np.array([float(x) for x in sys.argv[1:6]]) if len(sys.argv) > 1 else np.zeros(5)
if len(q) != 5:
    sys.exit("Give exactly 5 angles (radians), or none for q = 0.")

for name, angle in zip(JOINTS, q):
    data.qpos[model.joint(name).qposadr[0]] = angle
mujoco.mj_kinematics(model, data)  # compute positions from angles only, no physics

x, y, z = data.site("gripperframe").xpos * 1000  # metres -> mm
print("q (deg):", "  ".join(f"{n}={np.degrees(a):.0f}" for n, a in zip(JOINTS, q)))
print(f"tip (mm): x={x:.1f}  y={y:.1f}  z={z:.1f}")
