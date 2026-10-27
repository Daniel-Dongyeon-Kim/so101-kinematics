"""
Print the kinematic chain (CHAIN + TIP_OFFSET) for ANY MuJoCo robot model.

Usage:
  python extract_chain.py <path/to/model.xml> <tip site name, or a body name if there's no site>
  python extract_chain.py mujoco_menagerie/robotstudio_so101/so101.xml gripperframe

How it works:
  In an MJCF file, bodies are nested: each <body> has a pos/quat that says where
  its frame sits relative to its PARENT body. A <joint> inside a body lets that body
  (and everything nested inside it) move relative to the parent.
  So to go from the base to the tip, start at the tip's body, follow parent links
  up to the world, then reverse the list.

  It reads the COMPILED model (MjModel), not the raw XML text, because MuJoCo has
  already normalized quaternions, converted euler/axis-angle to quaternions,
  applied <default> classes and handled degrees vs radians.
"""
import sys

import mujoco
import numpy as np

if len(sys.argv) != 3:
    sys.exit(__doc__)
xml_path, site_name = sys.argv[1], sys.argv[2]

model = mujoco.MjModel.from_xml_path(xml_path)
try:
    site_id = model.site(site_name).id
    tip_body = model.site_bodyid[site_id]
except KeyError:  # no site with that name: treat it as a body, tip = that body's origin
    site_id = None
    tip_body = model.body(site_name).id

# Walk from the tip's body up to the world (body 0), then reverse.
bodies = []
b = tip_body
while b != 0:
    bodies.append(b)
    b = model.body_parentid[b]
bodies.reverse()

fmt = lambda v: "[" + ", ".join(f"{x:.8g}" for x in v) + "]"
warnings = []
print("CHAIN = [")
print("    # joint name     position offset (m)     orientation offset (quat w,x,y,z)")
for b in bodies:
    body = model.body(b)
    joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
    pos, quat = model.body_pos[b], model.body_quat[b]

    if not joints:
        label = None  # fixed link: no motor, just an offset
        if np.allclose(pos, 0) and np.allclose(quat, [1, 0, 0, 0]):
            print(f"    # (body '{body.name}' has no joint and no offset: skip it)")
            continue
        warnings.append(f"body '{body.name}' has no joint: it's a FIXED offset (row with joint None)")
    else:
        if len(joints) > 1:
            warnings.append(f"body '{body.name}' has {len(joints)} joints: needs one rotation per joint")
        j = joints[0]
        label = model.joint(j).name
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            warnings.append(f"joint '{label}' is not a hinge (rotation): rot_z won't describe it")
        if not np.allclose(model.jnt_axis[j], [0, 0, 1]):
            warnings.append(f"joint '{label}' turns about {model.jnt_axis[j]}, not z: need a general axis rotation")
        if not np.allclose(model.jnt_pos[j], 0):
            warnings.append(f"joint '{label}' is not at its body's origin (pos {model.jnt_pos[j]})")

    print(f"    ({label!r:<16}, {fmt(pos):<34}, {fmt(quat)}),   # body '{body.name}'")
print("]")
if site_id is not None:
    print(f"TIP_OFFSET = ({fmt(model.site_pos[site_id])}, {fmt(model.site_quat[site_id])})"
          f"   # site '{site_name}' in body '{model.body(tip_body).name}'")
else:
    print(f"TIP_OFFSET = ([0, 0, 0], [1, 0, 0, 0])   # no site: tip = origin of body '{site_name}'")

print("\nChecks:")
print("\n".join("  WARNING: " + w for w in warnings) if warnings else
      "  OK: every joint is a hinge about its body's own z axis, at the body origin.\n"
      "  So: offset = make_T(quat_to_rot(quat), pos), joint = make_T(rot_z(q), 0).")
