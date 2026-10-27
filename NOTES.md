# Session 1 notes

## Conventions
- q = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll], radians
- Frames: x forward (red), y left (green), z up (blue)

## 1. The q = 0 pose
At q=0, the arm forms an L-shape with the upper arm vertical, and the forearm and gripper pointing forward horizontally. 
Tip at q=0: (391, -1, 246) mm
This is important because my FK has to give exactly (391, -1, 246) mm at q = 0. It’s the first test it has to pass.

## 2. Does wrist_roll move the tip?
wrist_roll moves the tip frame in a small circle because the tip point isn’t exactly on the roll axis. This is important in order to not assume that roll can’t change the tip’s position and leave it out of my FK.

## 3. Which joints bend vs swing?
shoulder_pan is the only joint that swings left or right, acting as a turntable. The other three arm joints (shoulder_lift, elbow_flex, wrist_flex) only bend the arm up and down. This helps visualize the arm as “a turntable, then a 3-joint arm bending in one plane, then a wrist roll.” 