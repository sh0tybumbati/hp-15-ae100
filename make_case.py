#!/usr/bin/env python3
"""
Parametric 3D-printable case generator for the HP 15-ae100 motherboard
(board code ABW50 LA-C501P, Rev 1.0).

Outputs (binary STL, millimetres):
    case_base.stl    tray with standoffs, port cut-outs, corner screw ears
    case_lid.stl     lid with locating lip, vents over CPU/heatsink, screw holes
    board_ghost.stl  (optional, --ghost) stand-in for the PCB to check fit in a slicer

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
    ("HDMI",                395, 462, 7.0),
    ("USB-1",               475, 532, 7.5),
    ("USB-2",               548, 606, 7.5),
    ("USB-3",               620, 680, 7.5),
    ("audio jack",          685, 732, 7.0),
]

# Detached DC power jack: the board only has a cable header at the top right;
# the jack itself screws into the wall. Round hole in the right wall.
# (y_top_px: height along the edge, d_mm: hole dia, z_mm: centre above PCB top)
DC_JACK = {"y_px": 318, "d_mm": 11.5, "z_mm": 5.0}

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


def union(parts):
    return trimesh.boolean.union(parts, engine="manifold")


RIB_W, RIB_PITCH, RIB_OUT = 2.4, 5.0, 1.2     # wall fluting (mm)
FOOT_H = 1.6                                  # height of each foot tier (mm)
STEP = 3.0                                    # height of each lid crown step (mm)


def lid_deco(ox0, oy0, ox1, oy1, H, px, top, stepA):
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
    ap.add_argument("--comp-height", type=float, default=14.0,
                    help="tallest part above PCB (RAM slot, heatsink...) (mm)")
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
    ap.add_argument("--ghost", action="store_true", help="also write board_ghost.stl")
    ap.add_argument("--only", choices=["base", "lid", "plate", "chassis"],
                    help="write just one part; 'plate' = standoff test-fit plate, "
                         "'chassis' = standoff plate + I/O shield in one piece")
    ap.add_argument("--out", default="out", help="output directory")
    a = ap.parse_args()
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
        jy = px(RIGHT_X, DC_JACK['y_px'])[1]
        right_skip.append((jy - DC_JACK['d_mm'] / 2 - k, jy + DC_JACK['d_mm'] / 2 + k))
        left_skip = [(px(LEFT_X, yb)[1] - k, px(LEFT_X, ya)[1] + k) for _, ya, yb, _ in LEFT_PORTS_PX]
        back_skip = [(px(xa, TOP_Y)[0] - k, px(xb, TOP_Y)[0] + k) for _, xa, xb, _ in BACK_VENTS_PX]
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
        cut = box(ix1 - 1, y_lo, z_pcb_bot, ox1 + 6, y_hi, z_pcb_top + hh + m)
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

    # DC jack hole in the right wall
    jy = px(RIGHT_X, DC_JACK["y_px"])[1]
    hole = creation.cylinder(radius=DC_JACK["d_mm"] / 2, height=ox1 - ix1 + 8, sections=48)
    hole.apply_transform(transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    hole.apply_translation([(ix1 - 1 + ox1 + 6) / 2, jy, z_pcb_top + DC_JACK["z_mm"]])
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
        plate = union([box(-mg, -mg, 0, W + mg, H + mg, a.floor)] +
                      [tube(a.standoff_d / 2, a.hole_d / 2, a.standoff + 0.5, (x, y), z0=a.floor - 0.5)
                       for _, x, y in mount_points_mm(px)])
        parts["test_standoff_plate.stl"] = plate
    if a.only in (None, "chassis"):
        # standoff plate + I/O shield (right-edge wall with the port cut-outs, DC jack
        # hole) in one piece, to build your own enclosure around.
        # Origin = bottom-left corner of the board outline, +x along the board, +y up
        # the photo, z up from the bottom of the plate.
        mg2 = 3.0
        shield = box(ix1, -mg2, 0, ox1, H + mg2, z_top)
        for name, ya, yb, hh in RIGHT_PORTS_PX:
            y_hi, y_lo = px(RIGHT_X, ya)[1] + m, px(RIGHT_X, yb)[1] - m
            shield = shield.difference(
                box(ix1 - 1, y_lo, z_pcb_bot, ox1 + 1, y_hi, z_pcb_top + hh + m), engine="manifold")
        jhole = creation.cylinder(radius=DC_JACK["d_mm"] / 2, height=ox1 - ix1 + 4, sections=48)
        jhole.apply_transform(transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        jhole.apply_translation([(ix1 + ox1) / 2, px(RIGHT_X, DC_JACK["y_px"])[1],
                                 z_pcb_top + DC_JACK["z_mm"]])
        shield = shield.difference(jhole, engine="manifold")
        chassis = union([box(-mg2, -mg2, 0, ox1, H + mg2, a.floor), shield] +
                        [tube(a.standoff_d / 2, a.hole_d / 2, a.standoff + 0.5, (x, y), z0=a.floor - 0.5)
                         for _, x, y in mount_points_mm(px)])
        parts["standoff_plate_io_shield.stl"] = chassis
    if a.only in (None, "base"):
        parts["case_base.stl"] = body
    if a.only in (None, "lid"):
        parts["case_lid.stl"] = lid
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
