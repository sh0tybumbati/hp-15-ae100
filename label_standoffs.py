#!/usr/bin/env python3
"""Draw a labelled, to-scale map of the standoffs (board frame, mm) over the
straightened photo, so individual standoffs can be pointed at and nudged."""
import importlib.util, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PHOTO = sys.argv[1] if len(sys.argv) > 1 else "/root/.claude/uploads/3ad5407b-716d-5b3a-96ee-e3c53d526fc6/28cc9a32-image.jpg"
spec = importlib.util.spec_from_file_location("mc", "make_case.py")
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)

uv = mc._uv
u_left, u_tip = uv(*mc.LM_LEFT)[0], uv(*mc.LM_TIP)[0]
v_top = min(uv(x, y)[1] for x, y in [mc.LM_TOP, (370, 233), (610, 232)])
v_bot = uv(*mc.LM_LOBE)[1]
sx = mc.REF_WIDTH_MM / (u_tip - u_left)
sy = mc.REF_HEIGHT_MM / (uv(*mc.LM_BOTTOM)[1] - uv(*mc.LM_TOP)[1])
to_mm = lambda x, y: ((uv(x, y)[0] - u_left) * sx, (v_bot - uv(x, y)[1]) * sy)
W, H = (u_tip - u_left) * sx, (v_bot - v_top) * sy

im = Image.open(PHOTO).convert("RGB")
im = im.rotate(mc.TILT_DEG, center=(0, 0), resample=Image.BICUBIC, fillcolor=(255, 255, 255))
iw, ih = im.size
ext = [(0 - u_left) * sx, (iw - u_left) * sx, (v_bot - ih) * sy, v_bot * sy]

fig, ax = plt.subplots(figsize=(17, 10.8), dpi=100)
ax.imshow(np.asarray(im), extent=ext, alpha=0.5, zorder=0)
ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, ec="#0080c0", lw=1.5, ls="--", zorder=1))
for gx in range(0, int(W) + 10, 10):
    ax.axvline(gx, color="k", lw=0.25 if gx % 50 else 0.7, alpha=0.25, zorder=1)
for gy in range(0, int(H) + 10, 10):
    ax.axhline(gy, color="k", lw=0.25 if gy % 50 else 0.7, alpha=0.25, zorder=1)

rows = []
for i, p in enumerate(mc.MOUNT_HOLES_PX, 1):
    x, y = to_mm(*p); rows.append((i, x, y))
    ax.add_patch(plt.Circle((x, y), 3.0, fc="#ff3030", ec="k", lw=1, alpha=0.8, zorder=3))
    ax.add_patch(plt.Circle((x, y), 1.0, fc="#ffe000", ec="k", lw=0.6, zorder=4))
    ax.annotate(str(i), (x, y), xytext=(x + 5, y + 5), fontsize=17, fontweight="bold", color="#b00000",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#b00000", lw=1.5),
                arrowprops=dict(arrowstyle="-", color="#b00000", lw=1.2))
ax.set_xlim(-6, W + 6); ax.set_ylim(-6, H + 8); ax.set_aspect("equal")
ax.set_xticks(range(0, int(W) + 10, 20)); ax.set_yticks(range(0, int(H) + 10, 20))
ax.set_xlabel("x (mm), along the board:  0 = SATA end   →  toward the ports / right edge")
ax.set_ylabel("y (mm):  0 = bottom edge  ↑  toward the fan / back")
ax.set_title("Standoff map (1 grid square = 10 mm; red = 6 mm post, yellow = 2 mm pilot hole)\n"
             "Nudge directions:  +x = right (ports),  -x = left (SATA),  +y = up (back / fan),  -y = down (front)")
tbl = "#    x      y\n" + "\n".join(f"{i:>2} {x:6.1f} {y:6.1f}" for i, x, y in rows)
ax.text(0.01, 0.99, tbl, transform=ax.transAxes, va="top", family="monospace", fontsize=11,
        bbox=dict(fc="white", ec="k", alpha=0.9), zorder=6)
fig.tight_layout(); fig.savefig("standoff_labels.png")
print(tbl)
