"""Write the draw.io sources of Figures 1 and 2 of the 8-page paper:
    figures/exactfuse_overview.drawio  (Fig. 1: the unrolled network, one stage, one transformer block)
    figures/exactfuse_dstep.drawio     (Fig. 2: the closed-form data step)
Both open and edit in draw.io (app.diagrams.net); export them to PDF with draw.io (the paper includes the PDFs).
    python make_drawio_figures.py
"""
import base64
import math
import os
from xml.sax.saxutils import escape

FONT = "Helvetica"
# draw.io palette (fill, stroke)
BLUE = ("#dae8fc", "#6c8ebf")
GREEN = ("#d5e8d4", "#82b366")
ORANGE = ("#ffe6cc", "#d79b00")
PURPLE = ("#e1d5e7", "#9673a6")
YELLOW = ("#fff2cc", "#d6b656")
RED = ("#f8cecc", "#b85450")
GRAY = ("#f5f5f5", "#666666")
TEAL = ("#b0e3e6", "#0e8088")
CUBE_BLUE = ("#a9c4eb", "#6c8ebf")

# math in labels (HTML): italic letters with sub-/superscripts, script operators
YH = "<i>Y</i><sub>H</sub>"
YM = "<i>Y</i><sub>M</sub>"
D = r"\(\mathcal{D}\)"
R = r"\(\mathcal{R}\)"
DTH = r"\(\mathcal{D}_\theta\)"
RPH = r"\(\mathcal{R}_\phi\)"
DTHT = r"\(\alpha_k\mathcal{D}_\theta^{\top}\)"
RPHT = r"\(\beta_k\mathcal{R}_\phi^{\top}\)"
RPLUS = r"\(\mathcal{R}\)<sup>+</sup>"
PK = r"\(\mathcal{P}_k\)"


def X(sub=""):
    return "<b><i>X</i></b>" + (f"<sub>{sub}</sub>" if sub else "")


class Diagram:
    def __init__(self, name, fs):
        self.name, self.fs, self.cells, self.n = name, fs, [], 0

    def _id(self):
        self.n += 1
        return f"{self.name[:2]}{self.n}"

    def vertex(self, x, y, w, h, label="", style="", fs=None):
        i = self._id()
        st = f"html=1;whiteSpace=wrap;fontFamily={FONT};fontSize={fs or self.fs};" + style
        self.cells.append(f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{st}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def box(self, x, y, w, h, label, col, extra="", fs=None):
        return self.vertex(x, y, w, h, label, f"rounded=1;arcSize=12;fillColor={col[0]};strokeColor={col[1]};{extra}", fs)

    def group(self, x, y, w, h, title, col, fill=None, fs=None, title_align="left"):
        return self.vertex(x, y, w, h, title,
                           f"rounded=1;arcSize=4;dashed=1;dashPattern=6 3;fillColor={fill or 'none'};strokeColor={col};"
                           f"verticalAlign=top;align={title_align};spacingLeft=8;spacingRight=8;spacingTop=1;fontStyle=1;"
                           f"fontColor={col};container=0;", fs)

    def text(self, x, y, w, h, label, align="center", extra="", fs=None):
        return self.vertex(x, y, w, h, label, f"text;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;"
                                              f"spacing=0;{extra}", fs)

    def cube(self, x, y, w, h, col, size=8):
        return self.vertex(x, y, w, h, "", f"shape=cube;size={size};darkOpacity=0.08;darkOpacity2=0.2;fillColor={col[0]};"
                                           f"strokeColor={col[1]};")

    def op(self, x, y, sym="+", r=16):
        return self.vertex(x, y, r, r, sym, "ellipse;fillColor=#ffffff;strokeColor=#333333;fontStyle=1;spacing=0;"
                                            "verticalAlign=middle;spacingBottom=2;", fs=self.fs + 1)

    def badge(self, cx, cy, n, col="#4d4d4d", r=15):
        return self.vertex(cx - r / 2, cy - r / 2, r, r, f"<b>{n}</b>", f"ellipse;fillColor={col};strokeColor=#ffffff;"
                                                           "fontColor=#ffffff;spacing=0;verticalAlign=middle;", fs=9)

    def dot(self, cx, cy, col="#9673a6", r=7):
        return self.vertex(cx - r / 2, cy - r / 2, r, r, "", f"ellipse;fillColor={col};strokeColor={col};")

    def edge(self, src=None, tgt=None, label="", style="", pts=(), sp=None, tp=None, ex=None, en=None, fs=None):
        i = self._id()
        st = (f"html=1;edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=blockThin;endFill=1;endSize=5;"
              f"strokeColor=#333333;fontFamily={FONT};fontSize={fs or self.fs - 1};labelBackgroundColor=#ffffff;") + style
        if ex:
            st += f"exitX={ex[0]};exitY={ex[1]};exitDx=0;exitDy=0;"
        if en:
            st += f"entryX={en[0]};entryY={en[1]};entryDx=0;entryDy=0;"
        attrs = (f' source="{src}"' if src else "") + (f' target="{tgt}"' if tgt else "")
        geo = '<mxGeometry relative="1" as="geometry">'
        if sp:
            geo += f'<mxPoint x="{sp[0]}" y="{sp[1]}" as="sourcePoint"/>'
        if tp:
            geo += f'<mxPoint x="{tp[0]}" y="{tp[1]}" as="targetPoint"/>'
        if pts:
            geo += '<Array as="points">' + "".join(f'<mxPoint x="{a}" y="{b}"/>' for a, b in pts) + "</Array>"
        geo += "</mxGeometry>"
        self.cells.append(f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{st}" edge="1" parent="1"{attrs}>'
                          f"{geo}</mxCell>")
        return i

    def line(self, pts, style="endArrow=none;"):
        """a polyline through absolute points"""
        return self.edge(sp=pts[0], tp=pts[-1], pts=pts[1:-1], style="edgeStyle=none;" + style)

    def image(self, x, y, w, h, svg):
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return self.vertex(x, y, w, h, "", f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;"
                                           f"image=data:image/svg+xml,{b64};")

    def write(self, path, w, h):
        xml = (f'<mxfile host="drawio"><diagram id="{self.name}" name="{self.name}">'
               f'<mxGraphModel dx="{w}" dy="{h}" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
               f'fold="1" page="0" pageScale="1" pageWidth="{w}" pageHeight="{h}" math="1" shadow="0"><root>'
               '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) +
               "</root></mxGraphModel></diagram></mxfile>\n")
        open(path, "w", encoding="utf-8", newline="\n").write(xml)
        print("wrote", path, len(self.cells), "cells")


MEM = "dashed=1;strokeColor=#d79b00;fontColor=#b46504;"
GREYL = "strokeColor=#888888;"


def overview(path):
    d = Diagram("overview", 11)
    # ------------------------------------------------ (a) the unrolled iteration
    d.group(0, 0, 790, 222, "(a) Unrolled HQS iteration", "#666666", fill="#fcfcfc")
    lr = d.cube(26, 40, 40, 40, CUBE_BLUE, size=10)
    d.text(10, 82, 74, 26, f"{YH}: LR-HSI<br><i>B</i>×<i>h</i>×<i>w</i>", fs=10)
    msi = d.cube(20, 120, 54, 54, TEAL, size=5)
    d.text(10, 176, 74, 26, f"{YM}: HR-MSI<br><i>M</i>×<i>H</i>×<i>W</i>", fs=10)
    d.group(100, 24, 680, 76, "set up once per image", "#6c8ebf", fill="#f4f8fd", fs=10)
    b1 = d.box(114, 46, 238, 42, "estimate blur widths, grid offsets <i>θ</i>,<br>"
                                 "response shift <i>φ</i> (Thm. 3), noise levels", GREEN)
    b2 = d.box(380, 46, 178, 42, f"exact operators {D}, {R}<br>and their eigenbasis (Thm. 1)", GREEN)
    b3 = d.box(588, 46, 178, 42, "observation code <b><i>c</i></b><br>(blur, shift, noise)", GREEN)
    for bx, k in ((114, 1), (380, 2), (588, 3)):
        d.badge(bx + 2, 53, k)
    d.edge(lr, b1, ex=(1, 0.5), en=(0, 0.3))
    d.edge(msi, b1, ex=(1, 0.5), en=(0, 0.75), pts=[(92, 147), (92, 77.5)])
    d.edge(b1, b2)
    d.edge(b1, b3, ex=(0.85, 0), en=(0.5, 0), pts=[(316, 36), (677, 36)])
    # bus shared by every stage
    bus_y = 122
    d.line([(469, 88), (469, bus_y)], style="endArrow=none;" + GREYL)
    d.line([(677, 88), (677, bus_y), (284, bus_y)], style="endArrow=none;" + GREYL)
    d.text(110, 104, 250, 14, f"shared by every stage: {YH}, {YM}, {D}, {R}, <b><i>c</i></b>", align="left", fs=10)
    # stages
    bic = d.box(100, 152, 86, 40, "bicubic ×<i>s</i>", GRAY)
    xs = [224, 404, 584]
    st = []
    for k, x in enumerate(xs, 1):
        st.append(d.box(x, 148, 120, 48, f"<b>stage {k}</b><br>data step + prior", YELLOW))
        d.line([(x + 60, bus_y), (x + 60, 148)], style="endArrow=blockThin;endFill=1;endSize=5;" + GREYL)
    out = d.cube(738, 150, 44, 44, CUBE_BLUE, size=10)
    d.text(722, 196, 70, 24, f"<b><i>X̂</i></b>: HR-HSI<br><i>B</i>×<i>H</i>×<i>W</i>", fs=10)
    d.edge(lr, bic, ex=(0, 0.5), en=(0, 0.5), pts=[(10, 60), (10, 172)])
    d.edge(bic, st[0], label=X("0"))
    d.edge(st[0], st[1], label=X("1"))
    d.edge(st[1], st[2], label=X("2"))
    d.edge(st[2], out, en=(0, 0.6))
    for a, b in ((0, 1), (1, 2)):
        xa, xb = xs[a] + 60, xs[b] + 60
        d.edge(st[a], st[b], label="memory", ex=(0.5, 1), en=(0.5, 1), pts=[(xa, 212), (xb, 212)],
               style=MEM + "edgeStyle=orthogonalEdgeStyle;", fs=9)
    # ------------------------------------------------ (b) inside stage k
    top = 236
    d.group(0, top, 580, 220, "(b) Inside stage <i>k</i>", "#666666", fill="#fcfcfc")
    xin = d.box(10, top + 30, 56, 28, X("<i>k</i>−1"), GRAY)
    ds = d.box(92, top + 24, 140, 40, "<b>data step</b><br>closed form (Fig. 2)", PURPLE)
    d.edge(xin, ds)
    dsin = d.text(92, top + 82, 140, 16, f"{YH}, {YM}, {D}, {R}, <b><i>c</i></b>", fs=10)
    d.edge(dsin, ds, ex=(0.5, 0), en=(0.5, 1))
    # prior: a three-level U-Net
    d.vertex(246, top + 64, 314, 150, f"<b>prior</b> {PK}", "rounded=1;arcSize=4;dashed=1;dashPattern=6 3;fillColor=#fffaf3;"
             "strokeColor=#d79b00;fontColor=#b46504;verticalAlign=bottom;align=right;spacingRight=8;spacingBottom=2;", fs=10)
    add = d.op(506, top + 36)
    xo = d.box(534, top + 30, 40, 28, X("<i>k</i>"), GRAY)
    d.edge(ds, add, label=X("<i>k</i>").replace("X", "Z"), ex=(1, 0.5), en=(0, 0.5))
    ic = d.box(262, top + 92, 68, 22, "conv 3×3", GRAY, fs=10)
    e1 = d.box(258, top + 128, 76, 22, "2 blocks, 48", ORANGE, fs=10)
    e2 = d.box(312, top + 160, 76, 22, "3 blocks, 96", ORANGE, fs=10)
    m = d.box(362, top + 190, 84, 22, "4 blocks, 192", ORANGE, fs=10)
    d2 = d.box(420, top + 160, 76, 22, "3 blocks, 96", ORANGE, fs=10)
    d1 = d.box(474, top + 128, 76, 22, "4 blocks, 48", ORANGE, fs=10)
    oc = d.box(478, top + 92, 68, 22, "conv 3×3", GRAY, fs=10)
    d.line([(296, top + 44), (296, top + 92)], style="endArrow=blockThin;endFill=1;endSize=5;strokeColor=#333333;")
    d.text(302, top + 70, 160, 14, f"{RPLUS}{YM}, both residuals", align="left", fs=9)
    d.edge(ic, e1)
    d.edge(e1, e2, ex=(0.5, 1), en=(0, 0.5), pts=[(296, top + 171)])
    d.edge(e2, m, ex=(0.5, 1), en=(0, 0.5), pts=[(350, top + 201)])
    d.edge(m, d2, ex=(1, 0.5), en=(0.5, 1), pts=[(458, top + 201)])
    d.edge(d2, d1, ex=(1, 0.5), en=(0.5, 1), pts=[(512, top + 171)])
    d.edge(d1, oc)
    d.edge(oc, add, label="Δ<sub><i>k</i></sub>", ex=(0.5, 0), en=(0.5, 1))
    d.edge(e1, d1, label="skip", style=GREYL + "fontColor=#666666;", fs=9)
    d.edge(e2, d2, style=GREYL)
    d.edge(add, xo)
    d.edge(sp=(196, top + 139), tgt=e1, en=(0, 0.5), label="memory <b><i>F</i></b><sub><i>k</i>−1</sub>",
           style=MEM + "edgeStyle=none;", fs=9)
    d.edge(src=d1, ex=(1, 0.5), tp=(576, top + 139), style=MEM + "edgeStyle=none;")
    d.text(552, top + 142, 24, 14, "<b><i>F</i></b><sub><i>k</i></sub>", fs=9, extra="fontColor=#b46504;")
    for cx, cy in ((334, top + 128), (362, top + 190), (550, top + 128)):
        d.dot(cx, cy)
    d.dot(18, top + 196)
    d.text(26, top + 189, 200, 14, "FiLM: features modulated by <b><i>c</i></b>", align="left", fs=9)
    # ------------------------------------------------ (c) one transformer block
    cx0 = 594
    d.group(cx0, top, 196, 220, "(c) One transformer block", "#666666", fill="#fcfcfc")
    mid = cx0 + 98
    att = d.box(cx0 + 14, top + 40, 168, 40, "LN, transposed attention<br><b><i>A</i></b> = softmax(<i>τ</i><b><i>Q̂K̂</i></b>"
                                             "<sup>⊤</sup>), <i>C</i>′×<i>C</i>′", ORANGE, fs=10)
    r1 = d.op(mid - 8, top + 96)
    ffn = d.box(cx0 + 14, top + 126, 168, 40, "LN, gated feed-forward<br>GELU gate, depth-wise conv", ORANGE, fs=10)
    r2 = d.op(mid - 8, top + 182)
    d.edge(sp=(mid, top + 22), tgt=att)
    d.edge(att, r1)
    d.edge(r1, ffn)
    d.edge(ffn, r2)
    d.edge(src=r2, tp=(mid, top + 216))
    d.edge(sp=(mid, top + 30), tgt=r1, en=(0, 0.5), pts=[(cx0 + 6, top + 30), (cx0 + 6, top + 104)])
    d.edge(sp=(mid, top + 116), tgt=r2, en=(1, 0.5), pts=[(cx0 + 190, top + 116), (cx0 + 190, top + 190)])
    d.write(path, 790, 456)


def gain_plot_svg(w=150, h=110):
    """exact gain curve (blue) and the range of the learned gains (orange band), on a log axis"""
    def g(t):
        return 1.0 / (1.0 + 10 ** (-t))

    x0, x1, y0, y1 = 8, w - 4, h - 6, 10
    def px(t):
        return x0 + (t + 3) / 6 * (x1 - x0)

    def py(v):
        return y0 - v * (y0 - y1) / 1.25
    ts = [-3 + 6 * i / 60 for i in range(61)]
    upper = [(px(t), py(min(1.2, 2 * g(t)))) for t in ts]
    lower = [(px(t), py(0.5 * g(t))) for t in ts][::-1]
    band = " ".join(f"{a:.1f},{b:.1f}" for a, b in upper + lower)
    curve = " ".join(f"{px(t):.1f},{py(g(t)):.1f}" for t in ts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polygon points="{band}" fill="#ffe6cc" stroke="none"/>'
            f'<line x1="{x0}" y1="{py(1):.1f}" x2="{x1}" y2="{py(1):.1f}" stroke="#999" stroke-dasharray="3,2" stroke-width="0.8"/>'
            f'<polyline points="{curve}" fill="none" stroke="#2f5597" stroke-width="2"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#555" stroke-width="1"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1 - 6}" stroke="#555" stroke-width="1"/>'
            "</svg>")


def dstep(path):
    d = Diagram("dstep", 14)
    xk = d.cube(10, 104, 50, 50, CUBE_BLUE, size=10)
    d.text(-6, 66, 84, 34, f"{X('<i>k</i>−1')}<br><i>B</i>×<i>H</i>×<i>W</i>", fs=13)
    # spatial branch
    d.text(92, 0, 300, 16, "<i>spatial branch: compare with the LR-HSI</i>", align="left", fs=13)
    Db = d.box(86, 38, 130, 44, f"blur, then<br>downsample: {DTH}", GREEN)
    mh = d.op(236, 52, "−")
    yh = d.cube(233, 18, 22, 22, CUBE_BLUE, size=6)
    d.text(258, 20, 30, 16, YH, align="left", fs=13)
    rh = d.cube(262, 49, 22, 22, RED, size=6)
    d.text(229, 76, 86, 30, "mismatch <b><i>r</i></b><sub>H</sub><br><i>B</i>×<i>h</i>×<i>w</i>", fs=12)
    Dt = d.box(316, 38, 160, 44, f"back-project {DTHT}<br>"
                                 "spread over <i>H</i>×<i>W</i>", GREEN)
    d.edge(Db, mh)
    d.edge(yh, mh, ex=(0.5, 1), en=(0.5, 0))
    d.edge(mh, rh)
    d.edge(rh, Dt)
    # spectral branch
    d.text(92, 262, 300, 16, "<i>spectral branch: compare with the HR-MSI</i>", align="left", fs=13)
    Rb = d.box(86, 176, 130, 44, f"spectral<br>response: {RPH}", GREEN)
    mm = d.op(236, 190, "−")
    ym = d.cube(228, 226, 32, 32, TEAL, size=4)
    d.text(196, 234, 30, 16, YM, align="right", fs=13)
    rm = d.cube(260, 184, 28, 28, RED, size=4)
    d.text(230, 148, 86, 30, "mismatch <b><i>r</i></b><sub>M</sub><br><i>M</i>×<i>H</i>×<i>W</i>", fs=12)
    Rt = d.box(316, 176, 160, 44, f"back-project {RPHT}<br>"
                                  "spread over <i>B</i> bands", GREEN)
    d.edge(Rb, mm)
    d.edge(ym, mm, ex=(0.5, 0), en=(0.5, 1))
    d.edge(mm, rm)
    d.edge(rm, Rt)
    d.edge(xk, Db, ex=(1, 0.5), en=(0, 0.5), pts=[(76, 129), (76, 60)])
    d.edge(xk, Rb, ex=(1, 0.5), en=(0, 0.5), pts=[(76, 129), (76, 198)])
    sm = d.op(488, 121, "+")
    d.edge(Dt, sm, ex=(1, 0.5), en=(0.5, 0))
    d.edge(Rt, sm, ex=(1, 0.5), en=(0.5, 1))
    d.text(420, 120, 64, 34, "correction<br><b><i>G</i></b><sub><i>k</i></sub>", align="right", fs=12)
    # closed-form solve
    d.group(520, 14, 196, 222, "closed-form solve (Thm. 1)", "#6c8ebf", fill="#f4f8fd", fs=13)
    s1 = d.box(530, 44, 178, 46, "rotate into the eigenbasis<br><font style=\"font-size:12px\">one matrix product per axis</font>", BLUE)
    s2 = d.box(530, 108, 178, 46, "scale every component<br>by its gain <i>Γ<sub>bij</sub></i> (plot, right)", BLUE)
    s3 = d.box(530, 172, 178, 46, "rotate back to pixels", BLUE)
    for by, k in ((46, 1), (110, 2), (174, 3)):
        d.badge(534, by, k)
    d.edge(sm, s1, ex=(1, 0.5), en=(0, 0.5), pts=[(514, 129), (514, 66)])
    d.edge(s1, s2)
    d.edge(s2, s3)
    ad = d.op(734, 186, "+")
    d.edge(s3, ad)
    zk = d.cube(770, 169, 50, 50, CUBE_BLUE, size=10)
    d.text(756, 221, 80, 34, f"{X('<i>k</i>').replace('X', 'Z')}<br><i>B</i>×<i>H</i>×<i>W</i>", fs=13)
    d.edge(ad, zk)
    d.edge(xk, ad, ex=(0.5, 1), en=(0.5, 1), pts=[(35, 290), (742, 290)])
    d.text(150, 292, 520, 16, f"skip connection: where both observations agree, {X('<i>k</i>−1')} passes unchanged "
                              "(fixed point, Thm. 2)", fs=12, extra="fontColor=#555555;")
    # gain plot
    gx, gy = 846, 40
    d.text(gx - 10, gy - 30, 220, 16, "share of the mismatch removed", align="left", fs=12)
    d.image(gx, gy - 8, 150, 110, gain_plot_svg())
    d.text(gx - 12, gy + 2, 10, 14, "1", fs=10)
    d.text(gx - 12, gy + 88, 10, 14, "0", fs=10)
    d.text(gx + 154, gy + 14, 60, 16, "exact", align="left", fs=12, extra="fontColor=#2f5597;fontStyle=1;")
    d.text(gx + 154, gy + 44, 60, 32, "learned<br>range", align="left", fs=12, extra="fontColor=#b46504;fontStyle=1;")
    d.text(gx - 30, gy + 106, 220, 34, "how strongly the component is observed<br>"
                                       "(<i>α<sub>k</sub>λ</i><sup>r</sup><sub><i>i</i></sub><i>λ</i><sup>c</sup><sub><i>j</i></sub>"
                                       " + <i>β<sub>k</sub>ν<sub>b</sub></i>, log scale)", fs=11)
    d.write(path, 1060, 312)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    overview(os.path.join(here, "figures", "exactfuse_overview.drawio"))
    dstep(os.path.join(here, "figures", "exactfuse_dstep.drawio"))
