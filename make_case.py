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
    python3 make_case.py --board-width 203 --out out/
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
PX_X0, PX_X1 = 22, 975      # left / right edge of PCB in the photo
PX_Y0, PX_Y1 = 233, 795     # top / bottom edge of PCB in the photo

# PCB mounting holes (x, y) in photo pixels -> become screw standoffs.
# Delete/add entries to match your board. (Heatsink/fan screws are omitted.)
MOUNT_HOLES_PX = [
    (370, 242), (455, 247), (793, 308), (905, 458),
    (103, 595), (238, 635), (150, 697), (85, 767),
    (383, 768), (417, 716), (588, 719),
]

# Rear-panel I/O on the right edge: (name, y_top_px, y_bottom_px, height_mm)
# height_mm = how far the connector rises above the PCB surface.
RIGHT_PORTS_PX = [
    ("power/top connector", 285, 347, 8.0),
    ("HDMI",                395, 462, 7.0),
    ("USB-1",               475, 532, 7.5),
    ("USB-2",               548, 606, 7.5),
    ("USB-3",               620, 680, 7.5),
    ("audio jack",          685, 732, 7.0),
]

# Left-edge connectors: (name, y_top_px, y_bottom_px, height_mm)
LEFT_PORTS_PX = [
    ("SATA/HDD connector", 690, 790, 8.0),
]

# Fan exhaust: the blower sits at the back-left over the heat shield and blows
# towards the back edge (top of the photo). (name, x_left_px, x_right_px, height_mm)
BACK_VENTS_PX = [
    ("fan exhaust", 345, 550, 9.0),
]

# Lid vents: (x0, y0, x1, y1, slot_w_mm, pitch_mm) in photo pixels
VENT_AREAS_PX = [
    (340, 300, 548, 600, 3.0, 5.0),   # fan intake (blower draws air from the lid)
    (548, 380, 700, 690, 2.0, 4.0),   # heatsink / heat-pipe area
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


def lid_deco(ox0, oy0, ox1, oy1, W, H, px):
    """Engraved Art Deco pattern for the lid's outer face (z=0 is the outer face,
    so every groove is open to the print bed: no supports, only short bridges)."""
    d = 0.8
    cut = []
    # stepped frame: two concentric outlines with notched (ziggurat) corners
    for inset in (5.0, 8.0):
        x0, y0, x1, y1 = ox0 + inset, oy0 + inset, ox1 - inset, oy1 - inset
        n = 6.0   # corner notch size
        pts = [(x0 + n, y0), (x1 - n, y0), (x1 - n, y0 + n / 2), (x1, y0 + n / 2), (x1, y1 - n / 2),
               (x1 - n, y1 - n / 2), (x1 - n, y1), (x0 + n, y1), (x0 + n, y1 - n / 2),
               (x0, y1 - n / 2), (x0, y0 + n / 2), (x0 + n, y0 + n / 2)]
        w = 1.0
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            cut.append(box(min(ax, bx) - w / 2, min(ay, by) - w / 2, -1,
                           max(ax, bx) + w / 2, max(ay, by) + w / 2, d))
    # fluted columns either side of the vents
    fw, fp = 1.6, 3.2
    vx0 = px(VENT_AREAS_PX[0][0], 0)[0] - 6
    vx1 = px(VENT_AREAS_PX[-1][2], 0)[0] + 6
    ylo, yhi = oy0 + 12, oy1 - 12
    x = ox0 + 12
    while x + fw <= vx0:
        cut.append(box(x, ylo, -1, x + fw, yhi, d)); x += fp
    # recessed plaque + sunburst on the right
    pw, ph = 46.0, 30.0
    pcx, pcy = (vx1 + (ox1 - 12)) / 2, H / 2
    px0, py0, px1, py1 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
    cut.append(box(px0, py0, -1, px1, py1, d))
    x = vx1
    while x + fw <= ox1 - 12:
        if not (px0 - 3 < x + fw and x < px1 + 3):
            pass
        cut.append(box(x, ylo, -1, x + fw, py0 - 4, d))
        cut.append(box(x, py1 + 4, -1, x + fw, yhi, d))
        x += fp
    inner = box(px0 + 2, py0 + 2, -1, px1 - 2, py1 - 2, d + 0.8)
    for ang in range(-60, 61, 20):
        ray = box(-0.5, 0, -1, 0.5, 40.0, d + 0.8)
        ray.apply_transform(transformations.rotation_matrix(np.radians(ang), [0, 0, 1]))
        ray.apply_translation([pcx, py0 + 2, 0])
        cut.append(ray.intersection(inner, engine="manifold"))
    return union(cut)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board-width", type=float, default=203.0,
                    help="measured PCB width, left-right in photo (mm) [203, estimate]")
    ap.add_argument("--board-height", type=float, default=None,
                    help="measured PCB height (mm); default keeps the photo aspect ratio")
    ap.add_argument("--clearance", type=float, default=1.0, help="gap PCB<->wall (mm)")
    ap.add_argument("--wall", type=float, default=2.4, help="wall thickness (mm)")
    ap.add_argument("--floor", type=float, default=2.0, help="floor thickness (mm)")
    ap.add_argument("--standoff", type=float, default=4.0, help="standoff height (mm)")
    ap.add_argument("--pcb", type=float, default=1.2, help="PCB thickness (mm)")
    ap.add_argument("--comp-height", type=float, default=14.0,
                    help="tallest part above PCB (RAM slot, heatsink...) (mm)")
    ap.add_argument("--lid", type=float, default=2.4, help="lid plate thickness (mm)")
    ap.add_argument("--hole-d", type=float, default=2.0,
                    help="standoff pilot hole diameter (mm) (M2.5 self-tap ~2.0)")
    ap.add_argument("--standoff-d", type=float, default=6.0, help="standoff outer dia (mm)")
    ap.add_argument("--screw-d", type=float, default=3.2, help="lid screw clearance dia (mm)")
    ap.add_argument("--pilot-d", type=float, default=2.6, help="base ear pilot dia for M3 (mm)")
    ap.add_argument("--no-vents", action="store_true")
    ap.add_argument("--no-deco", action="store_true", help="plain case, no Art Deco styling")
    ap.add_argument("--ghost", action="store_true", help="also write board_ghost.stl")
    ap.add_argument("--only", choices=["base", "lid"], help="write just one part")
    ap.add_argument("--out", default="out", help="output directory")
    a = ap.parse_args()

    sx = a.board_width / (PX_X1 - PX_X0)
    W = a.board_width
    H = a.board_height or (PX_Y1 - PX_Y0) * sx
    sy = H / (PX_Y1 - PX_Y0)

    def px(x, y):          # photo px -> board mm (origin bottom-left, y up)
        return (x - PX_X0) * sx, (PX_Y1 - y) * sy

    ix0, iy0 = -a.clearance, -a.clearance                 # cavity
    ix1, iy1 = W + a.clearance, H + a.clearance
    ox0, oy0 = ix0 - a.wall, iy0 - a.wall                 # outer
    ox1, oy1 = ix1 + a.wall, iy1 + a.wall

    z_pcb_bot = a.floor + a.standoff
    z_pcb_top = z_pcb_bot + a.pcb
    z_top = z_pcb_top + a.comp_height                     # rim of base

    ear_r = 5.0
    ear_centres = [(ox0, oy0), (ox1, oy0), (ox0, oy1), (ox1, oy1)]

    # ---------------------------------------------------------------- base
    deco = not a.no_deco
    shell = box(ox0, oy0, 0, ox1, oy1, z_top)
    ears = [tube(ear_r, 0, z_top, c) for c in ear_centres]
    extras = []
    if deco:
        # ziggurat foot: two stepped tiers like the radio's stacked base bands
        t1, t2 = 2 * RIB_OUT, RIB_OUT
        extras.append(box(ox0 - t1, oy0 - t1, 0, ox1 + t1, oy1 + t1, 1.2))
        extras.append(box(ox0 - t2, oy0 - t2, 1.2, ox1 + t2, oy1 + t2, 2.4))
        # vertical fluting on all four walls, stopping short of the rim
        zr0, zr1 = 2.4, z_top - 2.0
        lo = ear_r + 3
        def hit(v, spans):
            return any(v + RIB_W > lo_ and v < hi_ for lo_, hi_ in spans)
        k = 1.5   # keep ribs this far from any opening
        right_skip = [(px(0, yb)[1] - k, px(0, ya)[1] + k) for _, ya, yb, _ in RIGHT_PORTS_PX]
        left_skip = [(px(0, yb)[1] - k, px(0, ya)[1] + k) for _, ya, yb, _ in LEFT_PORTS_PX]
        back_skip = [(px(xa, 0)[0] - k, px(xb, 0)[0] + k) for _, xa, xb, _ in BACK_VENTS_PX]
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
                      px(*p), z0=a.floor - 0.5) for p in MOUNT_HOLES_PX]
    body = union([body] + standoffs)

    # pilot holes in ears (open at top, blind at the bottom)
    for c in ear_centres:
        pilot = creation.cylinder(radius=a.pilot_d / 2, height=z_top - 2.0, sections=32)
        pilot.apply_translation([c[0], c[1], 2.0 + (z_top - 2.0) / 2 + 0.01])
        body = body.difference(pilot, engine="manifold")

    # port cut-outs (right wall). 0.8 mm margin on every side.
    m = 0.8
    for name, ya, yb, hh in RIGHT_PORTS_PX:
        y_hi = px(0, ya)[1] + m
        y_lo = px(0, yb)[1] - m
        cut = box(ix1 - 1, y_lo, z_pcb_bot, ox1 + 6, y_hi, z_pcb_top + hh + m)
        body = body.difference(cut, engine="manifold")

    # back wall exhaust slots (fan blows out towards the screen hinge)
    for name, xa, xb, hh in BACK_VENTS_PX:
        slot_w, pitch = 3.5, 5.5
        x0v, x1v = px(xa, 0)[0], px(xb, 0)[0]
        x = x0v
        while x + slot_w <= x1v:
            cut = box(x, iy1 - 1, z_pcb_bot + 1.0, x + slot_w, oy1 + 6, z_pcb_top + hh)
            body = body.difference(cut, engine="manifold")
            x += pitch

    # left wall cut-outs (SATA connector)
    for name, ya, yb, hh in LEFT_PORTS_PX:
        y_hi = px(0, ya)[1] + m
        y_lo = px(0, yb)[1] - m
        cut = box(ox0 - 6, y_lo, z_pcb_bot, ix0 + 1, y_hi, z_pcb_top + hh + m)
        body = body.difference(cut, engine="manifold")

    # ---------------------------------------------------------------- lid
    lid_plate = union([box(ox0, oy0, 0, ox1, oy1, a.lid)] +
                      [tube(ear_r, 0, a.lid, c) for c in ear_centres])
    lip_t, lip_h, tol = 1.6, 2.5, 0.35
    lip_outer = box(ix0 + tol, iy0 + tol, a.lid - 0.01, ix1 - tol, iy1 - tol, a.lid + lip_h)
    lip_inner = box(ix0 + tol + lip_t, iy0 + tol + lip_t, a.lid - 1,
                    ix1 - tol - lip_t, iy1 - tol - lip_t, a.lid + lip_h + 1)
    lid = union([lid_plate, lip_outer.difference(lip_inner, engine="manifold")])

    for c in ear_centres:
        hole = creation.cylinder(radius=a.screw_d / 2, height=a.lid + 2, sections=32)
        hole.apply_translation([c[0], c[1], a.lid / 2])
        lid = lid.difference(hole, engine="manifold")

    if not a.no_vents:
        for ax0, ay0, ax1, ay1, slot_w, pitch in VENT_AREAS_PX:
            vx0, vy0 = px(ax0, ay1)
            vx1, vy1 = px(ax1, ay0)
            x = vx0
            while x + slot_w <= vx1:
                lid = lid.difference(box(x, vy0, -1, x + slot_w, vy1, a.lid + lip_h + 1),
                                     engine="manifold")
                x += pitch

    if deco:
        lid = lid.difference(lid_deco(ox0, oy0, ox1, oy1, W, H, px), engine="manifold")
    # NOTE: the lid is exported plate-down (outer face on z=0, lip pointing up),
    # which is the print orientation: no supports needed.

    # ---------------------------------------------------------------- write
    os.makedirs(a.out, exist_ok=True)
    parts = {}
    if a.only != "lid":
        parts["case_base.stl"] = body
    if a.only != "base":
        parts["case_lid.stl"] = lid
    if a.ghost:
        g = [box(0, 0, z_pcb_bot, W, H, z_pcb_top)]
        for p in MOUNT_HOLES_PX:
            x, y = px(*p)
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
