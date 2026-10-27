"""
Test YOUR fk() against MuJoCo on 500 random poses (fk_ground_truth.npz from Session 1).

    python test_fk.py

Reports position error (mm) and orientation error (degrees), and saves fk_error.png
for your README. Target from the plan: every position error under 1 mm.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fk import fk

HERE = Path(__file__).resolve().parent
gt = np.load(HERE / "fk_ground_truth.npz")
Q, P_true, R_true = gt["q"], gt["pos"], gt["rot"]

pos_err_mm, rot_err_deg = [], []
for q, p, R in zip(Q, P_true, R_true):
    T = fk(q)
    pos_err_mm.append(np.linalg.norm(T[:3, 3] - p) * 1000)
    # Angle of the rotation that takes MuJoCo's orientation to yours (0 = identical).
    cos_angle = (np.trace(R.T @ T[:3, :3]) - 1) / 2
    rot_err_deg.append(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
pos_err_mm, rot_err_deg = np.array(pos_err_mm), np.array(rot_err_deg)

print(f"Tested {len(Q)} random poses against MuJoCo")
print(f"  position error (mm):     max {pos_err_mm.max():.6f}   mean {pos_err_mm.mean():.6f}")
print(f"  orientation error (deg): max {rot_err_deg.max():.6f}   mean {rot_err_deg.mean():.6f}")

worst = pos_err_mm.argmax()
if pos_err_mm.max() < 1.0:
    print("PASS: every pose within 1 mm.")
else:
    print(f"FAIL: worst pose #{worst}, q (deg) = {np.degrees(Q[worst]).round(1)}, "
          f"error {pos_err_mm[worst]:.1f} mm")
    print("Tip: try q with only ONE joint non-zero to find which joint is wrong.")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].hist(pos_err_mm, bins=40)
ax[0].set_xlabel("position error (mm)")
ax[0].set_ylabel("number of poses")
ax[1].hist(rot_err_deg, bins=40, color="tab:orange")
ax[1].set_xlabel("orientation error (deg)")
fig.suptitle(f"My FK vs MuJoCo, {len(Q)} random poses")
fig.tight_layout()
fig.savefig(HERE / "fk_error.png", dpi=120)
print("Saved fk_error.png")
