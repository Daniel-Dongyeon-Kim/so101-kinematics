# Session 1 notes

## Conventions
- q is each joint angle in radians: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
- Frames: x forward (red), y left (green), z up (blue)

## 1. The q = 0 pose
At q=0, the arm forms an L-shape with the upper arm vertical, and the forearm and gripper pointing forward horizontally. 
Tip at q=0: (391, -1, 246) mm
This is important because my FK has to give exactly (391, -1, 246) mm at q = 0. It’s the first test it has to pass.

*insert annotated gripper pic*

## 2. Does wrist_roll move the tip?
wrist_roll moves the tip frame in a small circle because the tip point isn’t exactly on the roll axis. This is important in order to not assume that roll can’t change the tip’s position and leave it out of my FK.

## 3. Which joints bend vs swing?
shoulder_pan is the only joint that swings left or right, acting as a turntable. The other three arm joints (shoulder_lift, elbow_flex, wrist_flex) only bend the arm up and down. This helps visualize the arm as “a turntable, then a 3-joint arm bending in one plane, then a wrist roll.” 

*insert joint_sweeps.png*

joint_sweeps.png helps visualize where the gripper tip goes as each joint moves on its own.

shoulder_pan shows z is constant and the joint acts as a turntable.
shoulder_lift, elbow_flex, wrist_flex shows y is constant, and these joints only bend the arm up and down.
wrist_roll shows a slight wiggle in y and z. This tells us the tip frame isn't exactly on the roll axis.


# Session 2 notes: Forward Kinematics

Forward kinematics takes q and returns where the gripper tip is and where it's facing.

so101.xml (arm's geometry found in official files online):
<body name="shoulder" pos="0.0388353 0 0.0624" quat="0 0 -1 0">
  <joint axis="0 0 1" name="shoulder_pan" .../>
  <body name="upper_arm" pos="-0.0303992 -0.0182778 -0.0542" quat="1 -1 -1 -1">
    <joint axis="0 0 1" name="shoulder_lift" .../>
1. Bodies are nested. Each body's position and quaternion are measured from its parent's frame
2. Joints live inside the body it moves. shoulder_pan turns shoulder and everything nested inside of it.
3. pos: where the motor sits relative to the parent joint, quat: how that frame is tilted. Tilts are chosen so every motor turns about its own z-axis. This is why it's axis="0 0 1" for every joint, and why rot_z function works for all 5 joints.

fk.py:
CHAIN = [
    # joint name       position offset (m)                   orientation offset (quaternion w,x,y,z)
    ("shoulder_pan",  [0.0388353, 0.0, 0.0624],             [0.0, 0.0, -1.0, 0.0]), # 180° about y
    ("shoulder_lift", [-0.0303992, -0.0182778, -0.0542],    [0.5, -0.5, -0.5, -0.5]), # 120° about (-1,-1,-1)
    ("elbow_flex",    [-0.11257, -0.028, 0.0],              [0.70710678, 0.0, 0.0, 0.70710678]), # 90° about z
    ("wrist_flex",    [-0.1349, 0.0052, 0.0],               [0.70710678, 0.0, 0.0, -0.70710678]), # 90° about z
    ("wrist_roll",    [0.0, -0.0611, 0.0181],               [0.01720911, -0.01720911, 0.70689734, 0.70689734]), # 178° about (-0.017,0.707,0.707) with a 2.8° spin about that axis
]
TIP_OFFSET = ([0.012, -0.000218, -0.098127], [0.70710678, 0.0, 0.70710678, 0.0]) # orientation offset to align tip frame with the world

Forward Kinematics recipe:
Initialize T with 4x4 identity
for loop through each joint i,
    T = T @ CHAIN offset @ ROTATION about z from q
T = T @ TIP_OFFSET

When doing forward kinematics, order matters.
Offset first from CHAIN values to move to where the motor is, then rotate using the 5 joint angles in q to turn the motor. Do gripper tip offset last after looping through each joint.

RESULTS:
Tested 500 random poses against MuJoCo
  position error (mm):     max 0.000001   mean 0.000001
  orientation error (deg): max 0.000000   mean 0.000000