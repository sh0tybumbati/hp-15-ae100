#!/usr/bin/env python3
"""
Parametric 3D-printable case generator for the HP 15-ae100 motherboard
(board code ABW50 LA-C501P, Rev 1.0).

Outputs (binary STL, millimetres):
    case_base.stl    tray with standoffs, port cut-outs, corner screw ears
    case_lid.stl     lid with locating lip, vents over CPU/heatsink, screw holes
    board_ghost.stl  (optional, --ghost) stand-in for the PCB to check fit in a slicer

SLIDE LAYOUT (default) - rounded ziggurat tiers + chevron bands, three printed parts + 2 screws
    1. base_plate.stl  standoffs + power-button tab. Print flat as-is.
    2. top_case.stl    ONE piece: rounded ziggurat sides + back, flat front, slotted top with engraved chevrons.
                       Already upside-down (top on bed).
    3. io_shield.stl   the front panel: half-round chevron bands + the port cut-outs. Print as-is (bands up).
    Assembly: mount the board on the plate standoffs (M2.5) -> slide plate + board into the top case from the
    front along the grooves until it stops at the back wall -> slide the shield up from below into its two vertical
    grooves (it stops under the top block) -> one M2.5 x 8-10 mm screw on each side of the shield face, into
    the header under the top plate. Those two screws are all there is to take apart.
    Use `--layout classic` for the older single tray + lid.

IMPORTANT - the feature positions below were measured off a product photo, not
a drawing. Print the base only (--only base) as a quick test fit first and tweak
BOARD_WIDTH_MM / the lists below with calipers.

    pip install trimesh manifold3d numpy
    python3 make_case.py --ref-height 142 --ref-width 242 --out out/
"""
import argparse
import os

import numpy as np
import trimesh
from trimesh import creation, transformations

# --------------------------------------------------------------------------
# Layout taken from the photo (pixel coordinates, origin top-left of photo).
# Everything is scaled by (board width in mm / photo board width in px).
# --------------------------------------------------------------------------
# The photo is not square to the board: the bottom edge of the PCB drops ~0.64 deg
# towards the right (measured on the long bottom edge, 830 px, y 778 -> 786). All
# photo pixels are rotated by TILT_DEG into the board's own frame before scaling.
TILT_DEG = 0.64

# Landmarks in the photo (px) used to calibrate against real measurements:
LM_LEFT = (23.5, 740)     # outer end of the SATA connector
LM_TIP = (972, 770)       # tip of the orange tab at the bottom-right corner
LM_TOP = (520, 233.5)     # top PCB edge, above the display connector
LM_BOTTOM = (520, 782)    # bottom PCB edge, below the CPU
LM_LOBE = (80, 790)       # lowest point of the bottom-left lobe (outline bottom)
# Real-world measurements:
REF_WIDTH_MM = 242.0      # LM_LEFT -> LM_TIP, along the board's long axis
REF_HEIGHT_MM = 142.0     # LM_TOP -> LM_BOTTOM, across the board
# where along the board each photo-pixel lookup is made (the tilt makes y depend on x)
RIGHT_X, LEFT_X, TOP_Y = 960, 40, 234


def _uv(x, y):
    """photo pixel -> (u along the board's long axis, v across it, down), in px."""
    t = np.radians(TILT_DEG)
    return x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)


# PCB mounting holes (x, y) in photo pixels -> become screw standoffs.
# Delete/add entries to match your board. (Heatsink/fan screws are omitted.)
MOUNT_HOLES_PX = [
    (370, 242), (455, 247), (793, 308), (905, 458),
    (821, 309), (103, 595), (238, 635), (150, 697), (85, 767),
    (383, 768), (417, 716), (588, 719),
    (961, 773),   # bottom-right corner tab
]

# Hand-tuning after test prints, keyed by standoff number (1-based position in
# MOUNT_HOLES_PX, as labelled in standoff_labels.png).
# Offsets are in mm on the board: +x = right (ports), +y = up (back / fan side).
MOUNT_ADJUST_MM = {
    1: (-1.0, 0.0),
    3: (+1.0, -1.0),
    5: (+1.0, -1.0),
    6: (-1.5, 0.0),
    7: (-1.0, 0.0),
    9: (-1.5, +1.0),
    10: (+0.5, 0.0),
    13: (-1.0, +1.0),
}
MOUNT_REMOVED = {8, 11, 12}      # standoffs dropped after the test fit


def mount_points_mm(px):
    """[(number, x_mm, y_mm)] for the standoffs that are kept, after adjustments."""
    out = []
    for n, p in enumerate(MOUNT_HOLES_PX, 1):
        if n in MOUNT_REMOVED:
            continue
        x, y = px(*p)
        dx, dy = MOUNT_ADJUST_MM.get(n, (0.0, 0.0))
        out.append((n, x + dx, y + dy))
    return out


# Rear-panel I/O on the right edge: (name, y_top_px, y_bottom_px, height_mm)
# height_mm = how far the connector rises above the PCB surface.
RIGHT_PORTS_PX = [
    ("Ethernet (RJ45)",     308, 380, 12.5),
    ("HDMI",                395, 462, 7.0),
    ("USB-1",               475, 532, 7.5),
    ("USB-2",               548, 606, 7.5),
    ("USB-3",               620, 680, 7.5),
    ("audio jack",          685, 732, 7.0),
]

# Detached DC power jack: it screws into the wall and connects by cable, so it can go
# anywhere. Round hole in the BACK wall (the right wall is full with the I/O row).
# (x_px: position along the back wall, d_mm: hole dia, z_mm: centre above PCB top)
DC_JACK = {"x_px": 700, "d_mm": 11.5, "z_mm": 5.0}

# Power/on-off switch SW1 on the BACK (underside) of the PCB, just inboard of the
# Ethernet jack. Located by registering the back photo to the front photo (holes +
# port rows, ~1.5 mm uncertainty). A compliant cantilever button is cut into the floor
# below it: a tab of the floor, free on three sides, with a post that presses SW1.
SWITCH = {
    "x_mm": 214.0, "y_mm": 115.6,   # SW1 plunger centre, board mm
    "tab_len": 24.0,                # root of the cantilever (inboard, -x) -> post (mm); sets the spring rate
    "tab_w": 9.0,                   # width of the tab
    "end_margin": 3.0,              # free end sits this far beyond the post (towards the Ethernet jack)
    "slot": 1.0,                    # gap cut around the tab
    "post_d": 4.0,                  # post diameter: bigger than the plunger so a ~1 mm
                                    # position error still hits it; the switch body is the stop
    "sw_h": 2.0,                    # plunger tip height below the PCB underside (guess; was 3.0, post made 1 mm taller)
    "gap": 0.4,                     # rest clearance post tip <-> plunger
}

# Left-edge connectors: (name, y_top_px, y_bottom_px, height_mm)
LEFT_PORTS_PX = [
    ("SATA/HDD connector", 690, 790, 8.0),
]

# Fan exhaust: the blower sits in the L-shaped notch at the top-left of the PCB
# (confirmed from the teardown photo) and blows towards the back edge (top of photo). (name, x_left_px, x_right_px, height_mm)
BACK_VENTS_PX = [
    ("fan exhaust", 80, 352, 10.0),
]

# Lid vents: (x0, y0, x1, y1, slot_w_mm, pitch_mm) in photo pixels
VENT_AREAS_PX = [
    (105, 385, 285, 555, 3.0, 5.0),   # fan intake (blower sits in the PCB's L-notch)
    (335, 380, 700, 690, 2.0, 4.0),   # heatsink / heat-pipe area
]


BUILD_VOLUME = (256, 256, 256)   # Elegoo Centauri Carbon 2 (mm); edit for other printers


def tube(r_out, r_in, h, center_xy, z0=0.0, sections=48):
    """Hollow cylinder standing on z0."""
    c = creation.cylinder(radius=r_out, height=h, sections=sections)
    if r_in > 0:
        hole = creation.cylinder(radius=r_in, height=h + 2, sections=sections)
        c = c.difference(hole, engine="manifold")
    c.apply_translation([center_xy[0], center_xy[1], z0 + h / 2])
    return c


def box(x0, y0, z0, x1, y1, z1):
    b = creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def button_parts(sw, floor_t, z_pcb_bot, x_limit):
    """(slot cutter, post) for the compliant floor button, in board mm."""
    cx, cy = sw["x_mm"], sw["y_mm"]
    xr = cx - sw["tab_len"]                           # root edge of the cantilever (inboard)
    xf = cx + sw["end_margin"]                        # free end (stays clear of the Ethernet jack)
    hw, g = sw["tab_w"] / 2, sw["slot"]
    z0, z1 = -1.0, floor_t + 0.5
    cut = union([box(xr, cy - hw - g, z0, xf + g, cy - hw, z1),          # side slots
                 box(xr, cy + hw, z0, xf + g, cy + hw + g, z1),
                 box(xf, cy - hw - g, z0, xf + g, cy + hw + g, z1)])     # free-end slot
    z_tip = z_pcb_bot - sw["sw_h"] - sw["gap"]
    post = creation.cylinder(radius=sw["post_d"] / 2, height=z_tip - (floor_t - 0.01), sections=40)
    post.apply_translation([cx, cy, (floor_t - 0.01) + (z_tip - (floor_t - 0.01)) / 2])
    return cut, post


def union(parts):
    return trimesh.boolean.union(parts, engine="manifold")


RIB_W, RIB_PITCH, RIB_OUT = 2.4, 5.0, 1.2     # wall fluting (mm)
FOOT_H = 1.6                                  # height of each foot tier (mm)
STEP = 3.0                                    # height of each lid crown step (mm)


def lid_deco(ox0, oy0, ox1, oy1, H, px, top, stepA, frame_only=False):
    """Engraved pattern for the lid shoulder (its top face is at z=top):
    fluted columns wherever it is free, plus a recessed sunburst plaque."""
    d = 0.8
    z0, z1 = top - d, top + 1
    cut = []
    fw, fp, guard = 1.6, 3.2, 4.0
    blocked = []                                       # (x0, y0, x1, y1) mm
    for ax0, ay0, ax1, ay1, *_ in VENT_AREAS_PX:
        (bx0, by0), (bx1, by1) = px(ax0, ay1), px(ax1, ay0)
        blocked.append((bx0, by0, bx1, by1))
    blocked.append(stepA)
    # recessed plaque to the right of the crown
    pw, ph = 30.0, 26.0
    pcx, pcy = (stepA[2] + (ox1 - 7)) / 2, H / 2
    qx0, qy0, qx1, qy1 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
    if frame_only:     # narrow grooves only (a wide pocket can't be printed on the bed face)
        fwid = 1.2
        for bx in (box(qx0, qy0, z0, qx1, qy0 + fwid, z1), box(qx0, qy1 - fwid, z0, qx1, qy1, z1),
                   box(qx0, qy0, z0, qx0 + fwid, qy1, z1), box(qx1 - fwid, qy0, z0, qx1, qy1, z1)):
            cut.append(bx)
    else:
        cut.append(box(qx0, qy0, z0, qx1, qy1, z1))
    blocked.append((qx0, qy0, qx1, qy1))
    ylo, yhi = oy0 + 10, oy1 - 10
    x = ox0 + 10
    while x + fw <= ox1 - 10:
        spans = sorted((r[1] - guard, r[3] + guard) for r in blocked
                       if r[0] - guard < x + fw and x < r[2] + guard)
        seg = ylo
        for b0, b1 in spans:
            if b0 - seg >= 8:
                cut.append(box(x, seg, z0, x + fw, b0, z1))
            seg = max(seg, b1)
        if yhi - seg >= 8:
            cut.append(box(x, seg, z0, x + fw, yhi, z1))
        x += fp
    # sunburst inside the plaque
    inner = box(qx0 + 2, qy0 + 2, z0 - 1, qx1 - 2, qy1 - 2, z1)
    for ang in range(-60, 61, 20):
        ray = box(-0.5, 0, z0 - d, 0.5, 40.0, z1)
        ray.apply_transform(transformations.rotation_matrix(np.radians(ang), [0, 0, 1]))
        ray.apply_translation([pcx, qy0 + 2, 0])
        cut.append(ray.intersection(inner, engine="manifold"))
    return union(cut)


def _hull(points):
    return trimesh.convex.convex_hull(np.asarray(points, float))


def _band_along_x(x0, x1, y_wall, sgn, z0, z1, rb):
    """Rail band on a side wall: rb proud of the wall, chamfered 45 deg on the ceiling side."""
    pts = []
    for x in (x0, x1):
        pts += [(x, y_wall, z0), (x, y_wall + sgn * rb, z0), (x, y_wall + sgn * rb, z1 - rb), (x, y_wall, z1)]
    return _hull(pts)


def _band_along_y(y0, y1, x_wall, sgn, z0, z1, rb):
    pts = []
    for y in (y0, y1):
        pts += [(x_wall, y, z0), (x_wall + sgn * rb, y, z0), (x_wall + sgn * rb, y, z1 - rb), (x_wall, y, z1)]
    return _hull(pts)


def _smooth_stair(z_top, r_conv, r_conc, n_arc=9):
    """STAIR polyline with every corner rounded (quadratic-Bezier fillets): convex step lips get r_conv,
    inner corners r_conc. All turns are 45 deg, so nothing overhangs more than 45 deg when printed upside-down.
    Returns an (n, 2) array of (outward offset, z) points, z strictly increasing."""
    P = [tuple(map(float, p)) for p in STAIR] + [(0.0, float(z_top))]
    pts = [P[0]]
    for i in range(1, len(P) - 1):
        a_, b_, c_ = (np.array(P[i - 1]), np.array(P[i]), np.array(P[i + 1]))
        d1, d2 = (b_ - a_) / np.linalg.norm(b_ - a_), (c_ - b_) / np.linalg.norm(c_ - b_)
        ang = float(np.arccos(np.clip(d1 @ d2, -1, 1)))
        if ang < 1e-6:
            pts.append(tuple(b_)); continue
        R = r_conv if (d1[0] * d2[1] - d1[1] * d2[0]) > 0 else r_conc
        t = min(R * np.tan(ang / 2), 0.45 * np.linalg.norm(b_ - a_), 0.45 * np.linalg.norm(c_ - b_))
        S_, E_ = b_ - d1 * t, b_ + d2 * t
        for u in np.linspace(0, 1, n_arc):
            pts.append(tuple((1 - u) ** 2 * S_ + 2 * (1 - u) * u * b_ + u ** 2 * E_))
    pts.append(P[-1])
    return np.array(pts)


def _arm_prism(p, q, width, z0, z1):
    p, q = np.array(p, float), np.array(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    n = np.array([-u[1], u[0]]) * width / 2
    return _hull([(*(p + n), z0), (*(p - n), z0), (*(q + n), z0), (*(q - n), z0),
                  (*(p + n), z1), (*(p - n), z1), (*(q + n), z1), (*(q - n), z1)])


def chevron_top(ox0, oy0, ix1, oy1, H, z_top, keepouts, d=0.8, fw=1.6, pitch=7.0, s=0.55, margin=9.0):
    """Engraved chevron grooves (V pointing towards the front, +x) across the top face."""
    z0, z1 = z_top - d, z_top + 1
    ym, L = H / 2, H / 2 + margin + 12
    arms, xk = [], ox0 + margin
    while xk <= ix1 - margin + s * L:
        arms.append(_arm_prism((xk, ym), (xk - s * L, ym - L), fw, z0, z1))
        arms.append(_arm_prism((xk, ym), (xk - s * L, ym + L), fw, z0, z1))
        xk += pitch
    cut = union(arms).intersection(box(ox0 + margin, oy0 + margin, z0 - 1, ix1 - margin, oy1 - margin, z1 + 1), engine="manifold")
    for (x0_, y0_, x1_, y1_) in keepouts:
        cut = cut.difference(box(x0_ - 3, y0_ - 3, z0 - 2, x1_ + 3, y1_ + 3, z1 + 2), engine="manifold")
    return cut


def _rod(p, q, r):
    p, q = np.array(p, float), np.array(q, float)
    L = float(np.linalg.norm(q - p))
    cyl = creation.cylinder(radius=r, height=L, sections=24)
    cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], (q - p) / L))
    cyl.apply_translation((p + q) / 2)
    return cyl


def _groove(x0, x1, y_back, y_in, sgn, gz0, gz1, flat=1.5):
    """Plate groove cutter. The ledge under the plate is chamfered at 45 deg beyond `flat` mm so the
    groove roof is self-supporting when the top case is printed upside-down."""
    box_part = box(x0, min(y_back, y_in), gz0, x1, max(y_back, y_in), gz1)
    yc = y_back + sgn * flat
    drop = abs(y_in - yc)
    wedge = _hull([(x, y, z) for x in (x0, x1) for (y, z) in [(yc, gz0), (y_in, gz0), (y_in, gz0 - drop)]])
    return union([box_part, wedge])


# Stepped "ziggurat" body profile, like the radio: (outward offset on the long sides mm, height mm).
# Treads are 45 deg (offset change == height change) so the shell prints upside-down without supports.
STAIR = [(15, 0), (15, 4), (10, 9), (10, 13), (5, 18), (5, 22), (0, 27)]
STAIR_BACK = 0.24            # the back (left end) steps are this fraction of the side steps (bed length!)


def build_slide(a, px, W, H, mount_pts):
    """Radio-style sliding-tray layout. Returns ({name: mesh in print orientation}, assembled {name: mesh}, info).

    base_plate  standoffs + button; slides in from the front along grooves in the side walls
    top_case    ONE piece: stepped (ziggurat) sides + back, flat front with an open slot for the shield,
                slotted top; printed upside-down (top on the bed)
    io_shield   the slatted front panel with the port cut-outs; slides up into vertical grooves, held by 2 screws
    """
    cl, tw = a.clearance, a.wall
    ts, eng, chd, lip_t = 2.4, 3.0, 2.0, 1.6          # shield thickness, plate engagement, shield groove depth, front lip
    lift, fl = a.lift, a.floor
    ix0, iy0, ix1, iy1 = -cl, -cl, W + cl, H + cl
    ox0, oy0, oy1 = ix0 - tw, iy0 - tw, iy1 + tw
    x_ch1 = ix1 + ts + 0.25                            # end of the shield groove
    x_front = x_ch1 + lip_t                            # the front plane of the shell
    z_p1 = lift + fl
    z_pcb_bot = z_p1 + a.standoff
    z_pcb_top = z_pcb_bot + a.pcb
    z_in = z_pcb_top + a.comp_height                   # ceiling underside
    tc = a.ceiling
    z_top = z_in + tc
    assert z_top > STAIR[-1][1] + 3, "raise --comp-height: the top block would vanish"
    zb_blk = STAIR[-1][1]                              # the top block starts here
    z_ch_top = 24.0                                    # shield grooves stop here (the rounded stair is thick enough below)
    hdr_len, hdr_h = 8.0, 3.0
    m = 0.8
    gz0, gz1 = lift - 0.2, lift + fl + 0.2
    # ------------------------------------------------------------------ top case: stepped solid
    def rect(os_, z):
        ob_ = os_ * STAIR_BACK
        return [(ox0 - ob_, oy0 - os_, z), (x_front, oy0 - os_, z), (x_front, oy1 + os_, z), (ox0 - ob_, oy1 + os_, z)]
    prof = _smooth_stair(z_top, a.tier_radius, a.tier_radius * 0.8)
    zs = np.append(np.arange(0.0, z_top - 1e-6, 0.5), z_top)
    frames = [rect(float(o), float(z)) for o, z in zip(np.interp(zs, prof[:, 1], prof[:, 0]), zs)]
    solid = union([_hull(frames[i] + frames[i + 1]) for i in range(len(frames) - 1)])
    # cavity (open underneath), the shield slot with its two vertical grooves, and the front window
    cuts = [box(ix0, iy0, -1, ix1, iy1, z_in),
            box(ix1, iy0, -1, x_ch1, iy1, z_top + 1),
            box(ix1, iy0 - chd, -1, x_ch1, iy0, z_ch_top), box(ix1, iy1, -1, x_ch1, iy1 + chd, z_ch_top),
            box(x_ch1 - 0.01, iy0, -1, x_front + 1, iy1, z_top + 1)]
    for c_ in cuts:
        solid = solid.difference(c_, engine="manifold")
    # plate grooves (45 deg ledge keeps the roof printable), running through the front lips
    solid = solid.difference(_groove(ix0, x_front + 1, iy0 - eng, iy0 + 0.5, +1, gz0, gz1), engine="manifold")
    solid = solid.difference(_groove(ix0, x_front + 1, iy1 + eng, iy1 - 0.5, -1, gz0, gz1), engine="manifold")
    # screw header under the ceiling at the front
    top = union([solid, box(ix1 - hdr_len, iy0, z_in - hdr_h, ix1, iy1, z_in + 0.01)])
    # SATA opening (left end)
    for name, ya, yb, hh in LEFT_PORTS_PX:
        y_hi, y_lo = px(LEFT_X, ya)[1] + m, px(LEFT_X, yb)[1] - m
        top = top.difference(box(ox0 - 12, y_lo, z_pcb_bot, ix0 + 1, y_hi, min(z_pcb_top + hh + m, z_in - 1)), engine="manifold")
    # fan exhaust grille + DC jack: each in a wedge-shaped bay cut into the back stair (45 deg floor, printable)
    def bay(x0_, x1_):
        pts = [(x, oy1 + d, z) for x in (x0_, x1_) for d, z in [(0, 9), (28, -19), (28, z_top + 1), (0, z_top + 1)]]
        return _hull(pts)
    jx = px(DC_JACK["x_px"], TOP_Y)[0]
    for name, xa, xb, hh in BACK_VENTS_PX:
        x0v, x1v = px(xa, TOP_Y)[0], px(xb, TOP_Y)[0]
        top = top.difference(bay(x0v - 3, x1v + 3), engine="manifold")
        slot_w, pitch = 3.5, 5.5
        x = x0v
        while x + slot_w <= x1v:
            top = top.difference(box(x, iy1 - 1, z_pcb_bot + 1.0, x + slot_w, oy1 + 1, min(z_pcb_top + hh, z_in - 1)), engine="manifold")
            x += pitch
    top = top.difference(bay(jx - 9, jx + 9), engine="manifold")
    jhole = creation.cylinder(radius=DC_JACK["d_mm"] / 2, height=(oy1 + 1) - (iy1 - 1), sections=48)
    jhole.apply_transform(transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    jhole.apply_translation([jx, (iy1 - 1 + oy1 + 1) / 2, z_pcb_top + DC_JACK["z_mm"]])
    top = top.difference(jhole, engine="manifold")
    # slotted top (the radio's grille) + engraved flutes and plaque
    hx0 = px(VENT_AREAS_PX[-1][0], VENT_AREAS_PX[-1][1])[0] - 2
    hx1 = px(VENT_AREAS_PX[-1][2], VENT_AREAS_PX[-1][1])[0] + 2
    if not a.no_vents:
        for ax0, ay0, ax1, ay1, slot_w, pitch in VENT_AREAS_PX:
            vx0, vy0 = px(ax0, ay1); vx1, vy1 = px(ax1, ay0)
            x = vx0
            while x + slot_w <= vx1:
                top = top.difference(box(x, vy0, z_in - 1, x + slot_w, vy1, z_top + 1), engine="manifold"); x += pitch
    if not a.no_deco:
        keep = []
        for ax0, ay0, ax1, ay1, *_ in VENT_AREAS_PX:
            (bx0, by0), (bx1, by1) = px(ax0, ay1), px(ax1, ay0)
            keep.append((bx0, by0, bx1, by1))
        top = top.difference(chevron_top(ox0, oy0, ix1, oy1, H, z_top, keep), engine="manifold")
    # one pilot hole per side in the header (the shield screws)
    z_scr = z_in + (tc - hdr_h) / 2
    y_scr = [iy0 + 8.0, iy1 - 8.0]
    for y_ in y_scr:
        pil = creation.cylinder(radius=a.pilot_d / 2 - 0.1, height=10.0, sections=32)
        pil.apply_transform(transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        pil.apply_translation([ix1 - 5.0 + 0.01, y_, z_scr]); top = top.difference(pil, engine="manifold")
    # ------------------------------------------------------------------ base plate (local z = 0 at its underside)
    plate = box(ix0 + 0.3, iy0 - (eng - 0.25), 0, ix1, iy1 + (eng - 0.25), fl)
    extra = []
    if not a.no_button:
        c_, p_ = button_parts(SWITCH, fl, fl + a.standoff, ix1)
        plate = plate.difference(c_, engine="manifold"); extra.append(p_)
    plate = union([plate] + extra + [tube(a.standoff_d / 2, a.hole_d / 2, a.standoff + 0.5, (x, y), z0=fl - 0.5)
                                     for _, x, y in mount_pts])
    # ------------------------------------------------------------------ I/O shield = the slatted front panel
    x_out = ix1 + ts
    parts_s = [box(ix1, iy0 + 0.25, 0, x_out, iy1 - 0.25, z_top),                       # centre panel, flush with the top
               box(ix1, iy0 - chd + 0.25, 0, x_out, iy0 + 0.3, z_ch_top - 0.2),           # tabs that ride in the grooves
               box(ix1, iy1 - 0.3, 0, x_out, iy1 + chd - 0.25, z_ch_top - 0.2)]
    if not a.no_deco:                                                                    # chevron bands (V pointing down)
        ym, slope, pitch_z = (iy0 + iy1) / 2, 0.38, 7.5
        Lh = (iy1 - iy0) / 2 + 6
        rods, zk = [], -slope * Lh - pitch_z
        while zk <= z_top + 1:
            rods.append(_rod((x_out, ym, zk), (x_out, ym - Lh, zk + slope * Lh), x_front - x_out))
            rods.append(_rod((x_out, ym, zk), (x_out, ym + Lh, zk + slope * Lh), x_front - x_out))
            zk += pitch_z
        bands = union(rods).intersection(box(ix1 + 0.2, iy0 + 3.0, 0, x_front + 0.01, iy1 - 3.0, z_top), engine="manifold")
        for y_ in y_scr:        # keep the screw heads clear of the bands
            kc = creation.cylinder(radius=4.5, height=8.0, sections=32)
            kc.apply_transform(transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            kc.apply_translation([x_out, y_, z_scr]); bands = bands.difference(kc, engine="manifold")
        parts_s.append(bands)
    shield = union(parts_s)
    for name, ya, yb, hh in RIGHT_PORTS_PX:
        y_hi, y_lo = px(RIGHT_X, ya)[1] + m, px(RIGHT_X, yb)[1] - m
        shield = shield.difference(box(ix1 - 1, y_lo, z_pcb_bot, x_front + 1, y_hi,
                                       min(z_pcb_top + hh + m, z_in - hdr_h - 0.5)), engine="manifold")
    for y_ in y_scr:
        hole = creation.cylinder(radius=1.45, height=ts + 2, sections=32)
        hole.apply_transform(transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        hole.apply_translation([(ix1 + x_out) / 2, y_, z_scr]); shield = shield.difference(hole, engine="manifold")
    # ------------------------------------------------------------------ orientations
    top_p = top.copy(); top_p.apply_transform(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, z_top], [0, 0, 0, 1.0]]))
    sh_p = shield.copy()   # slats up: inner face on the bed
    sh_p.apply_transform(np.array([[0, 0, -1, z_top], [0, 1, 0, 0], [1, 0, 0, -ix1], [0, 0, 0, 1.0]]))
    asm = {"top_case": top, "base_plate": plate.copy(), "io_shield": shield}
    asm["base_plate"].apply_translation([0, 0, lift])
    info = dict(z_top=z_top, z_in=z_in, z_pcb_bot=z_pcb_bot, z_pcb_top=z_pcb_top, x_end=x_front, lift=lift, x_front=x_front)
    return {"top_case.stl": top_p, "base_plate.stl": plate, "io_shield.stl": sh_p}, asm, info


_LAST = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-height", type=float, default=REF_HEIGHT_MM,
                    help="measured PCB height between the display connector edge and the "
                         "edge below the CPU (mm) [142]; sets the scale of the whole layout")
    ap.add_argument("--ref-width", type=float, default=REF_WIDTH_MM,
                    help="measured PCB length from the SATA connector end to the tip of the "
                         "bottom-right orange tab (mm) [242]; sets the x scale")
    ap.add_argument("--clearance", type=float, default=0.8, help="gap PCB<->wall (mm)")
    ap.add_argument("--wall", type=float, default=2.0, help="wall thickness (mm)")
    ap.add_argument("--floor", type=float, default=2.0, help="floor thickness (mm)")
    ap.add_argument("--standoff", type=float, default=6.0, help="standoff height (mm)")
    ap.add_argument("--pcb", type=float, default=1.2, help="PCB thickness (mm)")
    ap.add_argument("--comp-height", type=float, default=17.0,
                    help="clearance above the PCB under the ceiling (mm); the I/O-end header hangs 3 mm of it")
    ap.add_argument("--lid", type=float, default=2.4, help="lid plate thickness (mm)")
    ap.add_argument("--hole-d", type=float, default=2.4,
                    help="standoff pilot hole diameter (mm). 2.4 = M2.5 screw tapping into "
                         "printed plastic (FDM holes print ~0.2 small); use ~2.9 for a loose "
                         "clearance fit, ~2.2 for a tighter bite")
    ap.add_argument("--standoff-d", type=float, default=6.0, help="standoff outer dia (mm)")
    ap.add_argument("--screw-d", type=float, default=3.2, help="lid screw clearance dia (mm)")
    ap.add_argument("--pilot-d", type=float, default=2.6, help="base ear pilot dia for M3 (mm)")
    ap.add_argument("--no-vents", action="store_true")
    ap.add_argument("--dc-jack-d", type=float, default=None, help="DC jack hole dia (mm) [11.5]")
    ap.add_argument("--no-deco", action="store_true", help="plain case, no Art Deco styling")
    ap.add_argument("--no-button", action="store_true", help="omit the compliant power-button tab")
    ap.add_argument("--sw-height", type=float, default=None,
                    help="height of the switch plunger tip below the PCB underside (mm) [2.0]")
    ap.add_argument("--no-feet", action="store_true", help="omit the stick-on rubber foot recesses (classic layout)")
    ap.add_argument("--layout", choices=["slide", "classic"], default="slide",
                    help="slide = base plate slides into a top case + I/O shield + crown cap (default); "
                         "classic = single tray + lid")
    ap.add_argument("--lift", type=float, default=3.0, help="slide layout: gap under the plate (mm)")
    ap.add_argument("--ceiling", type=float, default=3.6, help="slide layout: top plate thickness (mm)")
    ap.add_argument("--tier-radius", type=float, default=5.0,
                    help="slide layout: rounding of the tier lips (mm); inner corners use 80%%. 0 = hard 45 deg steps. "
                         "Keep it under ~6 or the 4 mm risers get swallowed")
    ap.add_argument("--ghost", action="store_true", help="also write board_ghost.stl")
    ap.add_argument("--only", choices=["base", "lid", "plate", "chassis", "button", "slide"],
                    help="write just one part; 'plate' = standoff test-fit plate, "
                         "'chassis' = standoff plate + I/O shield in one piece, "
                         "'button' = small coupon to test the power-button alignment")
    ap.add_argument("--out", default="out", help="output directory")
    a = ap.parse_args()
    if a.sw_height is not None:
        SWITCH["sw_h"] = a.sw_height
    if a.dc_jack_d:
        DC_JACK["d_mm"] = a.dc_jack_d

    u_left, u_tip = _uv(*LM_LEFT)[0], _uv(*LM_TIP)[0]
    v_top = min(_uv(x, y)[1] for x, y in [LM_TOP, (370, 233), (610, 232)])
    v_bot = _uv(*LM_LOBE)[1]
    sx = a.ref_width / (u_tip - u_left)
    sy = a.ref_height / (_uv(*LM_BOTTOM)[1] - _uv(*LM_TOP)[1])
    W = (u_tip - u_left) * sx
    H = (v_bot - v_top) * sy

    def px(x, y):          # photo px -> board mm (origin bottom-left, y up), tilt-corrected
        u, v = _uv(x, y)
        return (u - u_left) * sx, (v_bot - v) * sy

    ix0, iy0 = -a.clearance, -a.clearance                 # cavity
    ix1, iy1 = W + a.clearance, H + a.clearance
    ox0, oy0 = ix0 - a.wall, iy0 - a.wall                 # outer
    ox1, oy1 = ix1 + a.wall, iy1 + a.wall

    z_pcb_bot = a.floor + a.standoff
    z_pcb_top = z_pcb_bot + a.pcb
    z_top = z_pcb_top + a.comp_height                     # rim of base

    ear_r, ear_off = 5.0, 1.5
    # lid-screw ears sit flush with the left/right ends and stick out front/back only
    ear_centres = [(ox0 + ear_r, oy0 - ear_off), (ox1 - ear_r, oy0 - ear_off),
                   (ox0 + ear_r, oy1 + ear_off), (ox1 - ear_r, oy1 + ear_off)]

    # ---------------------------------------------------------------- base
    deco = not a.no_deco
    shell = box(ox0, oy0, 0, ox1, oy1, z_top)
    ears = [tube(ear_r, 0, z_top, c) for c in ear_centres]
    extras = []
    if deco:
        # ziggurat foot: two stepped tiers like the radio's stacked base bands
        for i in range(3):      # 3 tiers, widest at the bed (prints without supports)
            o = RIB_OUT + (2 - i) * 0.3
            extras.append(box(ox0 - o, oy0 - o, i * FOOT_H, ox1 + o, oy1 + o, (i + 1) * FOOT_H))
        # vertical fluting on all four walls, stopping short of the rim
        zr0, zr1 = 3 * FOOT_H, z_top - 2.0
        lo = 2 * ear_r + 1
        def hit(v, spans):
            return any(v + RIB_W > lo_ and v < hi_ for lo_, hi_ in spans)
        k = 1.5   # keep ribs this far from any opening
        right_skip = [(px(RIGHT_X, yb)[1] - k, px(RIGHT_X, ya)[1] + k) for _, ya, yb, _ in RIGHT_PORTS_PX]
        left_skip = [(px(LEFT_X, yb)[1] - k, px(LEFT_X, ya)[1] + k) for _, ya, yb, _ in LEFT_PORTS_PX]
        back_skip = [(px(xa, TOP_Y)[0] - k, px(xb, TOP_Y)[0] + k) for _, xa, xb, _ in BACK_VENTS_PX]
        jx = px(DC_JACK['x_px'], TOP_Y)[0]
        back_skip.append((jx - DC_JACK['d_mm'] / 2 - k, jx + DC_JACK['d_mm'] / 2 + k))
        x = ox0 + lo
        while x + RIB_W <= ox1 - lo:
            extras.append(box(x, oy0 - RIB_OUT, zr0, x + RIB_W, oy0 + 0.5, zr1))
            if not hit(x, back_skip):
                extras.append(box(x, oy1 - 0.5, zr0, x + RIB_W, oy1 + RIB_OUT, zr1))
            x += RIB_PITCH
        y = oy0 + lo
        while y + RIB_W <= oy1 - lo:
            if not hit(y, left_skip):
                extras.append(box(ox0 - RIB_OUT, y, zr0, ox0 + 0.5, y + RIB_W, zr1))
            if not hit(y, right_skip):
                extras.append(box(ox1 - 0.5, y, zr0, ox1 + RIB_OUT, y + RIB_W, zr1))
            y += RIB_PITCH
    body = union([shell] + ears + extras)
    body = body.difference(box(ix0, iy0, a.floor, ix1, iy1, z_top + 1), engine="manifold")

    use_button = not a.no_button
    if use_button:
        b_cut, b_post = button_parts(SWITCH, a.floor, z_pcb_bot, ix1)
        body = union([body.difference(b_cut, engine="manifold"), b_post])
    if not a.no_feet:     # shallow recesses for stick-on rubber feet (lifts the case so the button can be pressed)
        for fx, fy in [(24, 24), (W - 24, 24), (24, H - 22), (W - 70, H - 22)]:
            rec = creation.cylinder(radius=5.0, height=1.6, sections=48)
            rec.apply_translation([fx, fy, -0.2])
            body = body.difference(rec, engine="manifold")
    standoffs = [tube(a.standoff_d / 2, a.hole_d / 2, a.standoff + 0.5,
                      (x, y), z0=a.floor - 0.5) for _, x, y in mount_points_mm(px)]
    body = union([body] + standoffs)

    # pilot holes in ears (open at top, blind at the bottom)
    for c in ear_centres:
        pilot = creation.cylinder(radius=a.pilot_d / 2, height=z_top - 2.0, sections=32)
        pilot.apply_translation([c[0], c[1], 2.0 + (z_top - 2.0) / 2 + 0.01])
        body = body.difference(pilot, engine="manifold")

    # port cut-outs (right wall). 0.8 mm margin on every side.
    m = 0.8
    for name, ya, yb, hh in RIGHT_PORTS_PX:
        y_hi = px(RIGHT_X, ya)[1] + m
        y_lo = px(RIGHT_X, yb)[1] - m
        cut = box(ix1 - 1, y_lo, z_pcb_bot, ox1 + 6, y_hi, min(z_pcb_top + hh + m, z_top - 1.0))
        body = body.difference(cut, engine="manifold")

    # back wall exhaust slots (fan blows out towards the screen hinge)
    for name, xa, xb, hh in BACK_VENTS_PX:
        slot_w, pitch = 3.5, 5.5
        x0v, x1v = px(xa, TOP_Y)[0], px(xb, TOP_Y)[0]
        x = x0v
        while x + slot_w <= x1v:
            cut = box(x, iy1 - 1, z_pcb_bot + 1.0, x + slot_w, oy1 + 6, z_pcb_top + hh)
            body = body.difference(cut, engine="manifold")
            x += pitch

    # left wall cut-outs (SATA connector)
    for name, ya, yb, hh in LEFT_PORTS_PX:
        y_hi = px(LEFT_X, ya)[1] + m
        y_lo = px(LEFT_X, yb)[1] - m
        cut = box(ox0 - 6, y_lo, z_pcb_bot, ix0 + 1, y_hi, z_pcb_top + hh + m)
        body = body.difference(cut, engine="manifold")

    # DC jack hole in the back wall
    jx = px(DC_JACK["x_px"], TOP_Y)[0]
    hole = creation.cylinder(radius=DC_JACK["d_mm"] / 2, height=(oy1 + 6) - (iy1 - 1), sections=48)
    hole.apply_transform(transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    hole.apply_translation([jx, (iy1 - 1 + oy1 + 6) / 2, z_pcb_top + DC_JACK["z_mm"]])
    body = body.difference(hole, engine="manifold")

    # locating tongue on the inner half of the wall top
    tongue_h, tg = 2.0, 0.25
    tongue = box(ix0 - 1.2, iy0 - 1.2, z_top - 0.01, ix1 + 1.2, iy1 + 1.2, z_top + tongue_h).difference(
        box(ix0, iy0, z_top - 1, ix1, iy1, z_top + tongue_h + 1), engine="manifold")
    body = union([body, tongue])

    # ---------------------------------------------------------------- lid
    # Stepped ziggurat crown like the radio's silhouette: perimeter plate, a
    # shoulder, a step and a tall central spine over the heatsink vents.
    # Printed outer-face UP on its flat underside; every step narrows as it
    # rises, so no supports are needed.
    t1 = a.lid + 1.2
    hx0 = px(VENT_AREAS_PX[-1][0], VENT_AREAS_PX[-1][1])[0] - 2        # spine x-range = heatsink vent area
    hx1 = px(VENT_AREAS_PX[-1][2], VENT_AREAS_PX[-1][1])[0] + 2
    parts = [box(ox0, oy0, 0, ox1, oy1, t1)] + [tube(ear_r, 0, t1, c) for c in ear_centres]
    lid_top, stepA = t1, None
    if deco:
        tiers = [(ox0 + 7, oy0 + 7, ox1 - 7, oy1 - 7),
                 (hx0 - 4, oy0 + 12, hx1 + 4, oy1 - 12),
                 (hx0, oy0 + 17, hx1, oy1 - 17)]
        stepA = tiers[1]
        z = t1
        for (x0, y0, x1, y1) in tiers:
            parts.append(box(x0, y0, z - 0.01, x1, y1, z + STEP))
            z += STEP
        lid_top = z
    lid = union(parts)

    # groove on the underside that receives the base's tongue
    groove = box(ix0 - 1.2 - tg, iy0 - 1.2 - tg, -1, ix1 + 1.2 + tg, iy1 + 1.2 + tg, tongue_h + 0.4).difference(
        box(ix0 + tg, iy0 + tg, -2, ix1 - tg, iy1 - tg, tongue_h + 2), engine="manifold")
    lid = lid.difference(groove, engine="manifold")

    for c in ear_centres:
        hole = creation.cylinder(radius=a.screw_d / 2, height=t1 + 2, sections=32)
        hole.apply_translation([c[0], c[1], t1 / 2])
        lid = lid.difference(hole, engine="manifold")

    if not a.no_vents:
        for ax0, ay0, ax1, ay1, slot_w, pitch in VENT_AREAS_PX:
            vx0, vy0 = px(ax0, ay1)
            vx1, vy1 = px(ax1, ay0)
            x = vx0
            while x + slot_w <= vx1:
                lid = lid.difference(box(x, vy0, -1, x + slot_w, vy1, lid_top + 1),
                                     engine="manifold")
                x += pitch

    if deco:
        lid = lid.difference(lid_deco(ox0, oy0, ox1, oy1, H, px, t1 + STEP, stepA),
                             engine="manifold")
    # lid is exported in print orientation (underside on z=0, crown pointing up)

    # ---------------------------------------------------------------- write
    os.makedirs(a.out, exist_ok=True)
    parts = {}
    if a.only in (None, "plate"):
        # quick test-fit: flat plate + the same standoffs, no walls
        mg = 3.0
        base_plate = box(-mg, -mg, 0, W + mg, H + mg, a.floor)
        extra = []
        if use_button:
            c_, p_ = button_parts(SWITCH, a.floor, z_pcb_bot, ix1)
            base_plate = base_plate.difference(c_, engine="manifold"); extra.append(p_)
        plate = union([base_plate] + extra +
                      [tube(a.standoff_d / 2, a.hole_d / 2, a.standoff + 0.5, (x, y), z0=a.floor - 0.5)
                       for _, x, y in mount_points_mm(px)])
        parts["test_standoff_plate.stl"] = plate
    if a.only in (None, "button"):
        # small coupon: just the corner around SW1 (standoffs 3, 4, 5 + the button) for a quick alignment test
        cx0, cy0 = 180.0, 80.0
        cp = box(cx0, cy0, 0, ix1 + a.wall, H + 3.0, a.floor)
        extra = []
        if use_button:
            c_, p_ = button_parts(SWITCH, a.floor, z_pcb_bot, ix1)
            cp = cp.difference(c_, engine="manifold"); extra.append(p_)
        parts["button_test_coupon.stl"] = union(
            [cp] + extra + [tube(a.standoff_d / 2, a.hole_d / 2, a.standoff + 0.5, (x, y), z0=a.floor - 0.5)
                            for n, x, y in mount_points_mm(px) if x >= cx0 and y >= cy0])
    if a.only in (None, "chassis"):
        # standoff plate + I/O shield (right-edge wall with the port cut-outs) in one piece, to build your own enclosure around.
        # Origin = bottom-left corner of the board outline, +x along the board, +y up
        # the photo, z up from the bottom of the plate.
        mg2 = 3.0
        shield = box(ix1, -mg2, 0, ox1, H + mg2, z_top)
        for name, ya, yb, hh in RIGHT_PORTS_PX:
            y_hi, y_lo = px(RIGHT_X, ya)[1] + m, px(RIGHT_X, yb)[1] - m
            shield = shield.difference(
                box(ix1 - 1, y_lo, z_pcb_bot, ox1 + 1, y_hi, min(z_pcb_top + hh + m, z_top - 1.0)), engine="manifold")
        c_plate = box(-mg2, -mg2, 0, ox1, H + mg2, a.floor)
        c_extra = []
        if use_button:
            c_, p_ = button_parts(SWITCH, a.floor, z_pcb_bot, ix1)
            c_plate = c_plate.difference(c_, engine="manifold"); c_extra.append(p_)
        chassis = union([c_plate, shield] + c_extra +
                        [tube(a.standoff_d / 2, a.hole_d / 2, a.standoff + 0.5, (x, y), z0=a.floor - 0.5)
                         for _, x, y in mount_points_mm(px)])
        parts["standoff_plate_io_shield.stl"] = chassis
    if a.layout == "classic" and a.only in (None, "base"):
        parts["case_base.stl"] = body
    if a.layout == "classic" and a.only in (None, "lid"):
        parts["case_lid.stl"] = lid
    if a.layout == "slide" and a.only in (None, "slide"):
        slide_parts, slide_asm, slide_info = build_slide(a, px, W, H, mount_points_mm(px))
        parts.update(slide_parts)
        _LAST["slide"] = (slide_asm, slide_info)
    if a.ghost:
        g = [box(0, 0, z_pcb_bot, W, H, z_pcb_top)]
        for _, x, y in mount_points_mm(px):
            g.append(tube(2.5, 0, 2.0, (x, y), z0=z_pcb_top))
        parts["board_ghost.stl"] = union(g)

    print(f"board {W:.1f} x {H:.1f} mm | case outer {ox1-ox0:.1f} x {oy1-oy0:.1f} x {z_top:.1f} mm")
    for fn, mesh in parts.items():
        path = os.path.join(a.out, fn)
        mesh.export(path)
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f"{path}: watertight={mesh.is_watertight} size={ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f} mm")
        if any(e > b for e, b in zip(ext, BUILD_VOLUME)):
            print(f"  WARNING: {fn} exceeds the {BUILD_VOLUME} mm build volume")


if __name__ == "__main__":
    main()
