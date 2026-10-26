"""MSTNet architecture figure in the visual language of Xu et al. STNet (Fig. 2)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_MLP = "#A9C9E8"
C_MLP_E = "#4A7BB0"
C_ATT = "#F2B892"
C_ATT_E = "#C07840"
C_NORM = "#F3E59C"
C_NORM_E = "#B89B3A"
C_SEQ = "#C6DCC0"
C_SEQ_E = "#5A8A55"
C_HEAD = "#E9D6A8"
C_HEAD_E = "#9A7A3A"
C_SOIL = "#A8D4C4"
C_SOIL_E = "#3F8A6E"
C_EMB = "#B8D9C4"
C_EMB_E = "#3F8A6E"
C_SOFT = "#E8A0B8"
C_SOFT_E = "#A05070"
C_DASH = "#8A8A8A"
C_INK = "#2A2A2A"
C_ARROW = "#555555"
C_GRAY = "#EEEEEE"
C_GRAY_E = "#9A9A9A"


def rbox(ax, x, y, w, h, fc, ec, lw=1.1, rs=0.06, z=2, ls="-"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={rs}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            zorder=z,
        )
    )


def txt(ax, x, y, s, fs=8, c=C_INK, w="normal", ha="center", va="center", z=6):
    ax.text(x, y, s, fontsize=fs, color=c, fontweight=w, ha=ha, va=va, zorder=z, linespacing=1.15)


def arr(ax, x0, y0, x1, y1, rad=0.0, lw=1.05, ms=8):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=C_ARROW,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0.8,
            shrinkB=0.8,
            zorder=5,
        )
    )


def tab(ax, x, y, w, h, label, fc, ec, fs=8.5):
    rbox(ax, x, y, w, h, fc, ec, lw=1.0, rs=0.10, z=3)
    txt(ax, x + w / 2, y + h / 2, label, fs=fs, c=ec, w="bold")


def dashed_region(ax, x, y, w, h, fc="#FBFBFB", ec=C_DASH, lw=1.15, z=1):
    rbox(ax, x, y, w, h, fc, ec, lw=lw, rs=0.10, z=z, ls=(0, (3.2, 2.0)))


def sine_icon(ax, cx, cy, r=0.078, color="#666666"):
    """Xu DoY badge: circle containing a tilde-wave mark."""
    ax.add_patch(Circle((cx, cy), r, facecolor="#FFFFFF", edgecolor=color, lw=1.05, zorder=4))
    # Unicode tilde operator — cleaner than a freehand sine plot
    ax.text(
        cx,
        cy + 0.01 * r,
        "\u223C",  # ∼
        fontsize=13,
        color=color,
        ha="center",
        va="center",
        zorder=5,
        fontfamily="DejaVu Sans",
    )


def sum_node(ax, cx, cy, r=0.10):
    """Element-wise add: STIX ⊕ (arms meet the ring; DejaVu looks like a ball+plus)."""
    del r
    # Prefer the STIX unified ⊕ glyph over DejaVu's detached + in a circle
    ax.text(
        cx,
        cy,
        "\u2295",
        fontsize=18,
        color=C_INK,
        ha="center",
        va="center",
        zorder=6,
        fontfamily="STIXGeneral",
    )


def _v_arrow_to(ax, x, y0, y1, color=C_ARROW, lw=1.1, tip=0.07):
    """Upward arrow: shaft stops before the tip; tip apex lands on y1 (box edge)."""
    ax.plot([x, x], [y0, y1 - tip], color=color, lw=lw, solid_capstyle="butt", zorder=5)
    ax.fill(
        [x - 0.038, x + 0.038, x],
        [y1 - tip, y1 - tip, y1],
        color=color,
        zorder=6,
        linewidth=0,
    )


def _h_arrow_to(ax, x0, x1, y, color="#666666", lw=1.1, tip=0.07):
    """Rightward arrow: tip apex lands on x1 (box edge), does not enter the box."""
    ax.plot([x0, x1 - tip], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=5)
    ax.fill(
        [x1, x1 - tip, x1 - tip],
        [y, y - 0.032, y + 0.032],
        color=color,
        zorder=7,
        linewidth=0,
    )


def draw_encoder(ax, x, y, w, h, *, stack_w=None):
    """
    STNet encoder panel: Transformer stack (left) + room for Seq2Vec (right).

      E' ──► Multi-Head Attention ──► Add & Norm ──► Feed Forward ──► Add & Norm ──► O
       │                              ▲                │                              ▲
       └──────────────────────────────┘                └──────────────────────────────┘

    Left residual rail is inset inside the gray panel (not on the border).
    ``stack_w`` is the Transformer column width; the rest of ``w`` holds Seq2Vec.
    Returns (bus_y, bus_left_x, out_xy) where out_xy is the Transformer exit into Seq2Vec.
    """
    rbox(ax, x, y, w, h, C_GRAY, C_GRAY_E, lw=1.15, rs=0.08, z=2)
    # Left pad for residual rail; tight top pad (less empty headroom)
    pad_l, pad_r, pad_b, pad_t = 1.00, 0.18, 0.55, 0.14
    tf_w = stack_w if stack_w is not None else w
    # Tab centered on the Transformer column (not the full panel with Seq2Vec)
    tab(ax, x + (tf_w - 1.05) / 2, y + h - 0.04, 1.05, 0.24, "encoder", C_SEQ, C_SEQ_E, fs=8)
    bx = x + pad_l
    bw = tf_w - pad_l - pad_r
    rail_x = x + 0.32

    h_att, h_n, h_ff = 0.46, 0.30, 0.38
    gap = 0.48
    y_att = y + pad_b
    y_n1 = y_att + h_att + gap
    y_ff = y_n1 + h_n + gap
    y_n2 = y_ff + h_ff + gap

    rbox(ax, bx, y_att, bw, h_att, C_ATT, C_ATT_E, lw=1.05, z=3)
    txt(ax, bx + bw / 2, y_att + h_att / 2, "Multi-Head\nAttention", fs=7.6, c=C_ATT_E, w="bold")

    rbox(ax, bx, y_n1, bw, h_n, C_NORM, C_NORM_E, lw=1.0, z=3)
    txt(ax, bx + bw / 2, y_n1 + h_n / 2, "Add & Norm", fs=7.4, c=C_NORM_E, w="bold")

    rbox(ax, bx, y_ff, bw, h_ff, C_MLP, C_MLP_E, lw=1.05, z=3)
    txt(ax, bx + bw / 2, y_ff + h_ff / 2, "Feed Forward", fs=7.6, c=C_MLP_E, w="bold")

    rbox(ax, bx, y_n2, bw, h_n, C_NORM, C_NORM_E, lw=1.0, z=3)
    txt(ax, bx + bw / 2, y_n2 + h_n / 2, "Add & Norm", fs=7.4, c=C_NORM_E, w="bold")

    cx = bx + bw / 2

    # ---- Input: central E' trunk → residual branch → QKV trident into MHA ----
    bus_y = y_att - 0.34
    stem_xs = [bx + t * bw for t in (0.30, 0.50, 0.70)]
    branch_y = bus_y + 0.10  # residual peels off the center trunk
    fork_y = bus_y + 0.20  # trident cross-bar above the residual branch
    ax.plot([cx, cx], [bus_y, fork_y], color=C_ARROW, lw=1.3, solid_capstyle="butt", zorder=4)
    ax.plot(
        [stem_xs[0], stem_xs[-1]],
        [fork_y, fork_y],
        color=C_ARROW,
        lw=1.3,
        solid_capstyle="butt",
        zorder=4,
    )
    for xi in stem_xs:
        _v_arrow_to(ax, xi, fork_y, y_att, tip=0.07)

    # Main path: tips land on the *bottom edge* of the next block
    _v_arrow_to(ax, cx, y_att + h_att, y_n1, tip=0.08)
    j1_y = y_n1 + h_n + 0.16
    ax.plot([cx, cx], [y_n1 + h_n, j1_y], color=C_ARROW, lw=1.1, solid_capstyle="butt", zorder=4)
    _v_arrow_to(ax, cx, j1_y, y_ff, tip=0.08)
    _v_arrow_to(ax, cx, y_ff + h_ff, y_n2, tip=0.08)

    # ---- Residual 1: from center trunk → left rail → Add&Norm ----
    r1_y = y_n1 + 0.5 * h_n
    ax.plot(
        [cx, rail_x, rail_x],
        [branch_y, branch_y, r1_y],
        color="#666666",
        lw=1.15,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=5,
    )
    _h_arrow_to(ax, rail_x, bx, r1_y, tip=0.07)

    # ---- Residual 2 ----
    r2_y = y_n2 + 0.5 * h_n
    ax.plot(
        [cx, rail_x, rail_x],
        [j1_y, j1_y, r2_y],
        color="#666666",
        lw=1.15,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=5,
    )
    _h_arrow_to(ax, rail_x, bx, r2_y, tip=0.07)

    out_y = y_n2 + h_n / 2
    txt(ax, x + 0.12, bus_y - 0.02, r"$\mathbf{E}'$", fs=8.5, c=C_INK, ha="left", va="top")
    # Incoming rail joins the central trunk at (cx, bus_y); exit is the Transformer column edge
    return bus_y, cx, (x + tf_w, out_y)


def _feature_strip(ax, x, y, w, h, z=3):
    """One Xu-style multi-channel feature column (green/blue/yellow/gray bands)."""
    colors = ["#9BC99A", "#8FB8E0", "#E8D98A", "#D0D0D0"]
    seg_h = h / len(colors)
    for i, fc in enumerate(reversed(colors)):
        ax.add_patch(
            Rectangle(
                (x, y + i * seg_h),
                w,
                seg_h,
                facecolor=fc,
                edgecolor="#666666",
                linewidth=0.55,
                zorder=z,
            )
        )


def draw_seq2vec(ax, x, y, w, h, accent, cmap_name, note, *, framed=True):
    """Xu Fig. 2 Seq2Vec: staggered sequence strips → pooled vector, with T axis."""
    del accent, cmap_name  # palette is fixed to match Xu; note is the pooling label
    if framed:
        dashed_region(ax, x, y, w, h, fc="#FFFFFF")
    else:
        rbox(ax, x, y, w, h, "#FFFFFF", C_GRAY_E, lw=0.95, rs=0.06, z=2)
    tab(ax, x + 0.08, y + h - 0.04, 0.95, 0.22, "Seq2Vec", C_SEQ, C_SEQ_E, fs=7.8)

    # Content fills the box from the bottom up (no floating bars over empty space)
    n_seq = 4
    dx, dy = 0.085, 0.070
    strip_w, strip_h = 0.16, min(1.20, h - 0.90)
    note_h = 0.20
    note_y = y + 0.10
    # fan sits just above the weight chip
    y0 = note_y + note_h + 0.28
    x_front = x + 0.16
    for i in range(n_seq - 1, -1, -1):
        _feature_strip(
            ax,
            x_front + i * dx,
            y0 + i * dy,
            strip_w,
            strip_h,
            z=3 + (n_seq - i),
        )

    # Diagonal T axis under the fan, parallel to the depth stagger
    stack_right = x_front + (n_seq - 1) * dx + strip_w
    t0x, t0y = x_front + 0.02, y0 - 0.12
    t1x = stack_right - 0.04
    t1y = t0y + (n_seq - 1) * dy
    ax.annotate(
        "",
        xy=(t1x, t1y),
        xytext=(t0x, t0y),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=0.95),
        zorder=8,
    )
    txt(ax, t1x + 0.07, t1y, r"$T$", fs=8, c=C_INK, ha="left", va="center")

    out_x = x + w - 0.34
    out_y = y0 + 0.5 * (n_seq - 1) * dy
    _feature_strip(ax, out_x, out_y, strip_w + 0.02, strip_h, z=6)

    mid_y = out_y + strip_h / 2
    left_arrow = stack_right + 0.08
    wx = 0.5 * (left_arrow + out_x)
    ax.annotate(
        "",
        xy=(out_x - 0.04, mid_y),
        xytext=(left_arrow, mid_y),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.05),
        zorder=7,
    )
    note_w = min(1.15, w - 0.20)
    ax.annotate(
        "",
        xy=(wx, mid_y - 0.02),
        xytext=(wx, note_y + note_h),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=0.9, linestyle=(0, (2.5, 2.0))),
        zorder=7,
    )
    rbox(ax, x + (w - note_w) / 2, note_y, note_w, note_h, "#F4F4F4", C_DASH, lw=0.8, rs=0.06, z=6)
    txt(ax, x + w / 2, note_y + note_h / 2, note, fs=6.8, c="#444444")


def draw_branch(
    ax,
    *,
    ox,
    y_mid,
    branch_w,
    accent,
    branch_fc,
    tab_label,
    labs,
    e_labs,
    meta,
    cmap_name,
    s2v_note,
    mlp_label,
    o_label,
):
    # Uneven stack: DoY stacks need a full clear band between rows 1 and 2
    ys = [y_mid + 1.15, y_mid + 0.10, y_mid - 1.35]

    # Front-end
    x_in, x_mlp, x_sum = ox + 0.14, ox + 0.58, ox + 1.48
    mlp_w, mlp_h = 0.50, 0.30
    sum_r = 0.10

    # Encoder panel: Transformer stack + Seq2Vec inside the same gray box
    x_enc = ox + 2.00
    tf_w, s2v_gap, s2v_w = 3.45, 0.36, 1.55
    enc_w = tf_w + s2v_gap + s2v_w + 0.16  # inner pad after Seq2Vec
    x_s2v = x_enc + tf_w + s2v_gap
    # Tight to content: pad_b + blocks + 3*gaps + pad_t ≈ 3.65
    enc_h = 3.75
    enc_y = y_mid - enc_h / 2

    # Tight branch frame; extra bottom strip reserved for the meta caption
    top = max(ys[0] + 1.05, enc_y + enc_h) + 0.18
    content_bot = min(ys[-1] - 0.22, enc_y)
    bot = content_bot - 0.55
    dashed_region(ax, ox, bot, branch_w, top - bot, fc=branch_fc, ec=accent, lw=1.25, z=0)
    tab(ax, ox + 0.14, top - 0.08, min(2.3, branch_w * 0.38), 0.28, tab_label, C_SEQ, accent, fs=8.2)
    txt(ax, ox + branch_w / 2, (bot + content_bot) / 2, meta, fs=6.1, c="#555555")

    # Inputs → MLP → e_t → DoY⊕p → (gather)  — layout mirrors Xu Fig. 2
    for i, (yy, lab, e_lab) in enumerate(zip(ys, labs, e_labs)):
        rbox(ax, x_in, yy - 0.13, 0.32, 0.26, "#FFFFFF", accent, lw=0.95, rs=0.03, z=3)
        txt(ax, x_in + 0.16, yy, lab, fs=7.4, c=accent, w="bold")

        rbox(ax, x_mlp, yy - mlp_h / 2, mlp_w, mlp_h, C_MLP, C_MLP_E, lw=1.0, z=3)
        txt(ax, x_mlp + mlp_w / 2, yy, mlp_label, fs=6.6, c=C_MLP_E, w="bold")
        arr(ax, x_in + 0.32, yy, x_mlp, yy, lw=0.85, ms=6)

        sum_x = x_sum + 0.18
        # e_t sits on the MLP→⊕ arrow (Xu: above the shaft)
        txt(ax, x_mlp + mlp_w + 0.24, yy + 0.14, e_lab, fs=6.8, c=C_INK, w="bold")

        # DoY → p_t on the vertical stem → ⊕  (Xu: p is PE, NOT the trunk output)
        doy_r = 0.080
        doy_y = yy + 0.52
        sine_icon(ax, sum_x, doy_y, r=doy_r, color="#666666")
        txt(ax, sum_x, doy_y + doy_r + 0.17, rf"$DoY_{i+1 if i < 2 else 'T'}$", fs=5.8, c=C_INK)
        stem_top = doy_y - doy_r
        stem_bot = yy + sum_r
        # vertical connector (Xu uses a stem into ⊕; keep a light arrowhead)
        ax.annotate(
            "",
            xy=(sum_x, stem_bot),
            xytext=(sum_x, stem_top),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9, mutation_scale=7),
            zorder=5,
        )
        p_lab = rf"$\mathbf{{p}}_{i+1 if i < 2 else 'T'}$"
        # sit on the stem, clearly above ⊕ (never on the outgoing horizontal)
        txt(
            ax,
            sum_x + 0.10,
            stem_bot + 0.55 * (stem_top - stem_bot),
            p_lab,
            fs=6.6,
            c=C_INK,
            ha="left",
        )

        sum_node(ax, sum_x, yy, r=sum_r)
        ax.annotate(
            "",
            xy=(sum_x - sum_r, yy),
            xytext=(x_mlp + mlp_w, yy),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=0.95, mutation_scale=8, shrinkB=0.5),
            zorder=5,
        )

    # Xu-style: dashed spine in the small 1→2 gap; vertical label alone in the 2→T gap
    cx = x_mlp + mlp_w / 2
    gap_top = ys[1] - mlp_h / 2
    gap_bot = ys[2] + mlp_h / 2
    gap_y = 0.5 * (gap_top + gap_bot)
    ax.plot(
        [cx, cx],
        [ys[0] - mlp_h / 2, ys[1] + mlp_h / 2],
        color="#9BB4C8",
        lw=0.9,
        ls=(0, (1.6, 1.6)),
        zorder=2,
    )
    ax.text(
        cx,
        gap_y,
        "... weight-sharing ...",
        rotation=90,
        fontsize=6.2,
        color="#7A94A8",
        ha="center",
        va="center",
        zorder=6,
        clip_on=False,
    )
    txt(ax, x_in + 0.16, gap_y, r"$\vdots$", fs=10, c=accent)

    # Encoder (Transformer + Seq2Vec share one panel)
    bus_y, bus_left_x, (tf_right, out_y) = draw_encoder(
        ax, x_enc, enc_y, enc_w, enc_h, stack_w=tf_w
    )

    # Clean gather: stubs → vertical rail → land on the LEFT of the E' bus (no overshoot)
    sum_x = x_sum + 0.14
    rail_x = sum_x + sum_r + 0.14
    for yy in ys:
        ax.plot([sum_x + sum_r, rail_x], [yy, yy], color=C_ARROW, lw=0.95, zorder=4)
    ax.plot([rail_x, rail_x], [min(ys[-1], bus_y), ys[0]], color=C_ARROW, lw=1.15, zorder=4)
    ax.plot([rail_x, bus_left_x], [bus_y, bus_y], color=C_ARROW, lw=1.15, solid_capstyle="butt", zorder=4)

    # Seq2Vec inside the encoder; O flows horizontally into it, then o exits the panel
    s2v_h = 2.35
    s2v_y = enc_y + 0.5 * (enc_h - s2v_h)
    draw_seq2vec(ax, x_s2v, s2v_y, s2v_w, s2v_h, accent, cmap_name, s2v_note, framed=False)
    mid_x = tf_right + 0.5 * s2v_gap
    ax.plot([tf_right, mid_x], [out_y, out_y], color=C_ARROW, lw=1.0, zorder=4)
    ax.plot([mid_x, mid_x], [out_y, y_mid], color=C_ARROW, lw=1.0, zorder=4)
    txt(ax, mid_x + 0.04, out_y, r"$\mathbf{O}$", fs=9, c=C_INK, ha="left")
    arr(ax, mid_x, y_mid, x_s2v, y_mid, lw=1.0, ms=7)
    enc_right = x_enc + enc_w
    # Exit through the encoder pad; label sits just outside the panel
    ax.plot([x_s2v + s2v_w, enc_right], [y_mid, y_mid], color=C_ARROW, lw=1.0, zorder=4)
    txt(ax, enc_right + 0.14, y_mid, o_label, fs=10, c=accent, w="bold", ha="left")

    return enc_right + 0.55, y_mid, bot, top - bot


def main():
    fig_w, fig_h = 15.0, 9.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    branch_w = 8.55

    zs = draw_branch(
        ax,
        ox=0.18,
        y_mid=7.85,
        branch_w=branch_w,
        accent=C_MLP_E,
        branch_fc="#F4F8FC",
        tab_label="Spectral SITS",
        labs=[r"$\mathbf{x}_1$", r"$\mathbf{x}_2$", r"$\mathbf{x}_{T_s}$"],
        e_labs=[r"$\mathbf{e}_1$", r"$\mathbf{e}_2$", r"$\mathbf{e}_{T_s}$"],
        meta=r"Per soy pixel (train: municipal mean) · 10 bands · $T_s{=}45$",
        cmap_name="Blues",
        s2v_note=r"NDVI $\mathbf{w}$",
        mlp_label=r"$MLP_s$",
        o_label=r"$\mathbf{o}_s$",
    )
    zc = draw_branch(
        ax,
        ox=0.18,
        y_mid=1.95,
        branch_w=branch_w,
        accent=C_ATT_E,
        branch_fc="#FDF7F1",
        tab_label="Climate SITS",
        labs=[r"$\mathbf{c}_1$", r"$\mathbf{c}_2$", r"$\mathbf{c}_{T_c}$"],
        e_labs=[r"$\mathbf{e}_1$", r"$\mathbf{e}_2$", r"$\mathbf{e}_{T_c}$"],
        meta=r"Municipal daily Xavier · 5 vars · $T_c{=}183$",
        cmap_name="Oranges",
        s2v_note=r"uniform $\mathbf{w}$",
        mlp_label=r"$MLP_c$",
        o_label=r"$\mathbf{o}_c$",
    )

    # Compact output head, vertically centered between the two trunks
    y_src_s, y_src_c = zs[1], zc[1]
    # Center using the midlines of the two branch boxes
    y_box_s = zs[2] + 0.5 * zs[3]
    y_box_c = zc[2] + 0.5 * zc[3]
    y_mid = 0.5 * (y_box_s + y_box_c)
    in_w, in_h, gap = 0.85, 0.48, 0.10
    chip_span = 3 * in_h + 2 * gap
    y_os = y_mid + chip_span / 2 - in_h / 2
    y_s = y_mid
    y_oc = y_mid - chip_span / 2 + in_h / 2

    hx = max(zs[0], zc[0]) + 0.55
    in_x = hx + 0.18
    mlp_w, mlp_h = 0.82, chip_span
    mlp_x = in_x + in_w + 0.18
    out_w, out_h = 0.62, 0.62
    out_x = mlp_x + mlp_w + 0.16
    hw = (out_x + out_w) - hx + 0.18
    hy = y_mid - chip_span / 2 - 0.28
    hh = chip_span + 0.56

    dashed_region(ax, hx, hy, hw, hh, fc="#FFFCF6")
    tab(ax, hx + (hw - 1.25) / 2, hy + hh - 0.08, 1.25, 0.24, "output head", C_HEAD, C_HEAD_E, fs=8)

    rbox(ax, in_x, y_os - in_h / 2, in_w, in_h, C_EMB, C_EMB_E, lw=1.1, z=3)
    txt(ax, in_x + in_w / 2, y_os, r"$\mathbf{o}_s$", fs=10, c=C_EMB_E, w="bold")
    rbox(ax, in_x, y_s - in_h / 2, in_w, in_h, C_SOIL, C_SOIL_E, lw=1.1, z=3)
    txt(ax, in_x + in_w / 2, y_s, r"$\mathbf{s}$", fs=10, c=C_SOIL_E, w="bold")
    rbox(ax, in_x, y_oc - in_h / 2, in_w, in_h, C_EMB, C_EMB_E, lw=1.1, z=3)
    txt(ax, in_x + in_w / 2, y_oc, r"$\mathbf{o}_c$", fs=10, c=C_EMB_E, w="bold")

    # Orthogonal routes from trunk exits into the compact chips
    jx = hx - 0.12
    ax.plot([zs[0], jx], [y_src_s, y_src_s], color=C_ARROW, lw=1.05, zorder=4)
    ax.plot([jx, jx], [y_src_s, y_os], color=C_ARROW, lw=1.05, zorder=4)
    arr(ax, jx, y_os, in_x, y_os, rad=0.0, lw=1.05, ms=7)
    ax.plot([zc[0], jx], [y_src_c, y_src_c], color=C_ARROW, lw=1.05, zorder=4)
    ax.plot([jx, jx], [y_src_c, y_oc], color=C_ARROW, lw=1.05, zorder=4)
    arr(ax, jx, y_oc, in_x, y_oc, rad=0.0, lw=1.05, ms=7)

    rbox(ax, mlp_x, y_mid - mlp_h / 2, mlp_w, mlp_h, C_MLP, C_MLP_E, lw=1.15, z=3)
    txt(ax, mlp_x + mlp_w / 2, y_mid, r"$MLP_2$", fs=11, c=C_MLP_E, w="bold")
    arr(ax, in_x + in_w, y_os, mlp_x, y_os, rad=0.0, lw=0.95, ms=7)
    arr(ax, in_x + in_w, y_s, mlp_x, y_s, rad=0.0, lw=0.95, ms=7)
    arr(ax, in_x + in_w, y_oc, mlp_x, y_oc, rad=0.0, lw=0.95, ms=7)

    rbox(ax, out_x, y_mid - out_h / 2, out_w, out_h, C_SOFT, C_SOFT_E, lw=1.1, z=3)
    txt(ax, out_x + out_w / 2, y_mid, r"$\hat{y}$", fs=11, c=C_SOFT_E, w="bold")
    arr(ax, mlp_x + mlp_w, y_mid, out_x, y_mid, rad=0.0, lw=1.0, ms=7)

    # Equal margins around content (fixes left-heavy whitespace)
    x_left, x_right = 0.18, out_x + out_w
    y_bot = min(zs[2], zc[2], hy)
    y_top = max(zs[2] + zs[3], zc[2] + zc[3], hy + hh)
    pad_x, pad_y = 0.35, 0.30
    ax.set_xlim(x_left - pad_x, x_right + pad_x)
    ax.set_ylim(y_bot - pad_y, y_top + pad_y)
    w_data = (x_right - x_left) + 2 * pad_x
    h_data = (y_top - y_bot) + 2 * pad_y
    fig.set_size_inches(w_data * 0.95, h_data * 0.95, forward=True)

    fig.savefig(OUT / "mstnet_arch.pdf", bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(OUT / "mstnet_arch.png", dpi=260, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "mstnet_arch.pdf")


if __name__ == "__main__":
    main()
