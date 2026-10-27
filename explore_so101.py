"""
Project 1, Session 1: get to know the SO-101 model in MuJoCo.

What this does (no viewer needed, runs in ~2 s on an M1):
  1. Loads the Menagerie SO-101 and prints every joint, its limits and its actuator.
  2. Sweeps each arm joint across its range (others held at 0) and records where
     the gripper tip ("gripperframe" site) goes. Saves a plot: joint_sweeps.png
  3. Saves 500 random configurations with MuJoCo's tip position + orientation to
     fk_ground_truth.npz. Session 2 checks YOUR forward kinematics against this file.

Usage:
  python explore_so101.py                      # expects ./mujoco_menagerie next to this file
  MENAGERIE_DIR=/path/to/mujoco_menagerie python explore_so101.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MENAGERIE = Path(os.environ.get("MENAGERIE_DIR", HERE / "mujoco_menagerie"))
XML = MENAGERIE / "robotstudio_so101" / "scene.xml"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP = "gripperframe"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
tip_id = model.site(TIP).id


def tip_pose(q_arm):
    """Set the 5 arm joints, run kinematics only (no physics), return tip pos (m) and rotation."""
    data.qpos[:] = 0
    for name, val in zip(ARM_JOINTS, q_arm):
        data.qpos[model.joint(name).qposadr[0]] = val
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[tip_id].copy(), data.site_xmat[tip_id].reshape(3, 3).copy()


# 1. Inventory -------------------------------------------------------------
print(f"MuJoCo {mujoco.__version__} | timestep {model.opt.timestep*1000:.0f} ms | "
      f"{model.njnt} joints, {model.nu} actuators\n")
print(f"{'joint':<14}{'min deg':>9}{'max deg':>9}   actuator (kp, max torque N·m)")
for j in range(model.njnt):
    lo, hi = np.degrees(model.jnt_range[j])
    a = model.actuator(model.joint(j).name)
    print(f"{model.joint(j).name:<14}{lo:9.1f}{hi:9.1f}   kp={model.actuator_gainprm[a.id, 0]:.0f}, "
          f"±{model.actuator_forcerange[a.id, 1]:.2f}")

p0, _ = tip_pose(np.zeros(5))
print(f"\nTip position at all-zeros: x={p0[0]*1000:.1f}  y={p0[1]*1000:.1f}  z={p0[2]*1000:.1f} mm")
print("Q to answer in your notes: what pose is the arm in at q=0? (Open the viewer to check.)")

# 2. One-joint-at-a-time sweeps -------------------------------------------
fig, axes = plt.subplots(1, len(ARM_JOINTS), figsize=(18, 3.6), sharey=True)
for k, (name, ax) in enumerate(zip(ARM_JOINTS, axes)):
    lo, hi = model.jnt_range[model.joint(name).id]
    angles = np.linspace(lo, hi, 60)
    pts = []
    for a in angles:
        q = np.zeros(5)
        q[k] = a
        pts.append(tip_pose(q)[0])
    pts = np.array(pts) * 1000
    for i, axis in enumerate("xyz"):
        ax.plot(np.degrees(angles), pts[:, i], label=axis)
    ax.set_title(name)
    ax.set_xlabel("joint angle (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("tip position (mm)")
axes[0].legend()
fig.suptitle("SO-101 tip position vs. each joint (others at 0)")
fig.tight_layout()
fig.savefig(HERE / "joint_sweeps.png", dpi=120)
print("\nSaved joint_sweeps.png")

# 3. Ground truth for checking your own FK next session --------------------
rng = np.random.default_rng(0)
ranges = np.array([model.jnt_range[model.joint(n).id] for n in ARM_JOINTS])
Q = rng.uniform(ranges[:, 0], ranges[:, 1], size=(500, 5))
P, R = zip(*(tip_pose(q) for q in Q))
np.savez(HERE / "fk_ground_truth.npz", q=Q, pos=np.array(P), rot=np.array(R), joints=ARM_JOINTS)
print("Saved fk_ground_truth.npz (500 random configs: q [rad], tip pos [m], tip rotation 3x3)")
