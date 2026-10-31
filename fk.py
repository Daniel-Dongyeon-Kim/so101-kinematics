"""
Session 2: forward kinematics (FK) for the SO-101, written from scratch in NumPy.

FK answers: "given the joint angles q, where is the gripper tip, and which way is it facing?"

The idea in one line:
    tip pose = (offset to joint 1)(turn joint 1) (offset to joint 2)(turn joint 2) ... (offset to tip)

Every piece is a 4x4 "homogeneous transform" T = [[R, p], [0 0 0 1]]:
    R (3x3) = a rotation, p (3,) = a translation.
Multiplying transforms left to right walks down the arm, one link at a time.

Run this file to check each part as you finish it:
    python fk.py
Then run the full 500-pose test:
    python test_fk.py
"""
import numpy as np

# ---------------------------------------------------------------------------
# The arm's geometry, copied from mujoco_menagerie/robotstudio_so101/so101.xml
# (MuJoCo's compiled values: lengths in metres, quaternions as w, x, y, z).
#
# Each entry is the FIXED offset from the previous joint's frame to this joint's
# frame (where the motor sits and how it's tilted). In this model every joint
# turns about its OWN local z axis, and the joint sits at its frame's origin.
# ---------------------------------------------------------------------------
CHAIN = [
    # joint name       position offset (m)                    orientation offset (quaternion w,x,y,z)
    ("shoulder_pan",  [0.0388353, 0.0, 0.0624],             [0.0, 0.0, -1.0, 0.0]),
    ("shoulder_lift", [-0.0303992, -0.0182778, -0.0542],    [0.5, -0.5, -0.5, -0.5]),
    ("elbow_flex",    [-0.11257, -0.028, 0.0],              [0.70710678, 0.0, 0.0, 0.70710678]),
    ("wrist_flex",    [-0.1349, 0.0052, 0.0],               [0.70710678, 0.0, 0.0, -0.70710678]),
    ("wrist_roll",    [0.0, -0.0611, 0.0181],               [0.01720911, -0.01720911, 0.70689734, 0.70689734]),
]
# Fixed offset from the wrist_roll frame to the tip ("gripperframe" site).
TIP_OFFSET = ([0.012, -0.000218, -0.098127], [0.70710678, 0.0, 0.70710678, 0.0])


def quat_to_rot(quat):
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix. Given to you: it's a standard
    formula (Modern Robotics appendix B), not the point of this session."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def rot_z(theta):
    """TODO 1: 3x3 rotation matrix for turning by angle theta (radians) about the z axis.

    Hint: a point on the x axis, (1, 0, 0), should end up at (cos θ, sin θ, 0).
          A point on the y axis, (0, 1, 0), should end up at (-sin θ, cos θ, 0).
          z doesn't change. Those three results are the three COLUMNS of the matrix.
    Reading: Modern Robotics section 3.2.1.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def make_T(R, p):
    """TODO 2: build the 4x4 transform [[R, p], [0, 0, 0, 1]].

    Hint: start from np.eye(4), put R in the top-left 3x3 block and p in the
          top 3 entries of the last column. (Modern Robotics section 3.3.1.)
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    
    return T


def fk(q):
    """TODO 3: forward kinematics. q = 5 joint angles (radians), in CHAIN order.
    Returns the 4x4 pose of the tip in the robot's base frame.

    Recipe:
        T = identity
        for each joint i:
            T = T @ (fixed offset for joint i) @ (rotation about z by q[i])
        T = T @ (fixed tip offset)
    where a fixed offset is make_T(quat_to_rot(quat), pos)
    and   a joint rotation is make_T(rot_z(angle), zero vector).

    The ORDER matters: offset first ("move to where the motor is"),
    then rotate ("turn the motor").
    """

    T = np.eye(4)
    for i in range(len(CHAIN)):
        T = T @ make_T(quat_to_rot(CHAIN[i][2]),CHAIN[i][1]) @ make_T(rot_z(q[i]),np.zeros(3))
    T = T @ make_T(quat_to_rot(TIP_OFFSET[1]),TIP_OFFSET[0])
    
    return T


# ---------------------------------------------------------------------------
# Quick checks. Each one tells you which TODO is wrong if it fails.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Check 1: rot_z
    R = rot_z(np.pi / 2)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0]), f"TODO 1: x axis turned 90° should become y.\n{R}"
    assert np.allclose(R @ [0, 0, 1], [0, 0, 1]), "TODO 1: z should not change."
    print("Check 1 passed: rot_z")

    # Check 2: make_T
    T = make_T(np.eye(3), [1, 2, 3])
    assert T.shape == (4, 4), "TODO 2: should be 4x4."
    assert np.allclose(T @ [0, 0, 0, 1], [1, 2, 3, 1]), "TODO 2: origin should move to p."
    print("Check 2 passed: make_T")

    # Check 3: fk at q = 0 should match MuJoCo (and tip_check.py): (391.4, -1.0, 246.3) mm
    tip = fk(np.zeros(5))[:3, 3] * 1000
    print(f"fk(q=0) tip (mm): {tip}")
    assert np.allclose(tip, [391.4, -1.0, 246.3], atol=0.1), "TODO 3: q=0 tip doesn't match MuJoCo."
    print("Check 3 passed: fk at q = 0. Now run: python test_fk.py")
