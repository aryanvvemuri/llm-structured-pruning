"""Draw the README figure: token F1 against model size, light and dark SVG.

No dependencies. The numbers are the 5-seed dev-holdout means recorded in
RESULTS.md (runs support-v2-longer, -allocator, -quant and -layer-drop, seeds
20261114-20261118). Sizes are megabytes of weights: 2 bytes per parameter for
16-bit models, measured MLX weight files for 8-bit and 4-bit.

    python scripts/make_results_figure.py docs/figures
"""
import sys
from pathlib import Path

W, H = 880, 536
LEFT, RIGHT, TOP, BOTTOM = 70, 850, 142, 452
X0, X1 = 150.0, 1070.0
Y0, Y1 = 48.0, 56.0

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "prune": "#2a78d6", "quant": "#eb6834", "layer": "#1baf7a", "ref": "#52514e"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "prune": "#3987e5", "quant": "#d95926", "layer": "#199e70", "ref": "#c3c2b7"},
}

# (label, group, size MB, F1, label dx, label dy, text-anchor)
POINTS = [
    ("Unpruned", "ref", 988.1, 54.89, -8, -13, "start"),
    ("Uniform 10%", "prune", 889.0, 54.32, 0, 21, "middle"),
    ("Uniform 20%", "prune", 789.9, 53.29, -12, 4, "end"),
    ("Curve 20%", "prune", 790.6, 54.58, -12, 4, "end"),
    ("Drop 3 layers", "layer", 898.6, 54.98, 0, -13, "middle"),
    ("Drop 3 layers + uniform 20%", "layer", 790.2, 53.92, -12, 4, "end"),
    ("Drop 7 layers", "layer", 779.3, 50.25, 12, 4, "start"),
    ("Unpruned, 8-bit", "quant", 525.0, 54.91, 0, -13, "middle"),
    ("Uniform 20%, 8-bit", "quant", 419.8, 53.30, 0, 21, "middle"),
    ("Unpruned, 4-bit", "quant", 278.1, 50.52, 12, 4, "start"),
    ("Uniform 20%, 4-bit", "quant", 222.3, 48.62, 12, 4, "start"),
]
LEGEND = [("ref", "Unpruned, 16-bit"), ("prune", "MLP pruning, 16-bit"),
          ("layer", "Layer dropping, 16-bit"), ("quant", "Quantized (MLX)")]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def sx(v):
    return LEFT + (v - X0) / (X1 - X0) * (RIGHT - LEFT)


def sy(v):
    return BOTTOM - (v - Y0) / (Y1 - Y0) * (BOTTOM - TOP)


def marker(group, x, y, t, r=6.0):
    """Shape repeats the colour, so identity never rests on colour alone."""
    ring = f'stroke="{t["surface"]}" stroke-width="2"'
    fill = f'fill="{t[group]}"'
    if group == "quant":
        return f'<rect x="{x - r + 0.5:.1f}" y="{y - r + 0.5:.1f}" width="{2 * r - 1:.1f}" height="{2 * r - 1:.1f}" rx="1.5" {fill} {ring}/>'
    if group == "layer":
        d = r + 1.5
        return f'<path d="M{x:.1f} {y - d:.1f} L{x + d:.1f} {y:.1f} L{x:.1f} {y + d:.1f} L{x - d:.1f} {y:.1f} Z" {fill} {ring}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {fill} {ring}/>'


def draw(theme):
    t = THEMES[theme]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}" role="img" aria-labelledby="t d">',
         '<title id="t">Token F1 against model size for pruned, layer-dropped and quantized Qwen2.5-0.5B</title>',
         '<desc id="d">8-bit quantization keeps the unpruned score at about half the size. 20 percent MLP '
         'pruning costs at most about 1.6 F1. 4-bit quantization and dropping seven layers cost 3 to 6 F1.</desc>',
         f'<rect width="{W}" height="{H}" rx="8" fill="{t["surface"]}"/>',
         f'<text x="{LEFT - 46}" y="34" font-size="17" font-weight="600" fill="{t["ink"]}">'
         '8-bit quantization halves the model for free; 20% pruning costs about 1 F1</text>',
         f'<text x="{LEFT - 46}" y="56" font-size="12.5" fill="{t["ink2"]}">'
         'Token F1 on the development holdout (mean of 5 seeds) against weight size. '
         'Seed-to-seed sd is 0.4–1.6 F1,</text>',
         f'<text x="{LEFT - 46}" y="73" font-size="12.5" fill="{t["ink2"]}">'
         'so points closer than about 1.5 F1 are not reliably different.</text>']
    x = LEFT - 46
    for group, name in LEGEND:
        o.append(marker(group, x + 7, 98, t, 5.0))
        o.append(f'<text x="{x + 20}" y="102" font-size="12.5" fill="{t["ink2"]}">{name}</text>')
        x += 34 + 7.1 * len(name)
    for v in range(48, 57, 2):
        y = sy(v)
        o.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{LEFT - 10}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{t["muted"]}">{v}</text>')
    o.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{BOTTOM}" y2="{BOTTOM}" stroke="{t["axis"]}" stroke-width="1"/>')
    for v in range(200, 1001, 200):
        xx = sx(v)
        o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{BOTTOM}" y2="{BOTTOM + 5}" stroke="{t["axis"]}" stroke-width="1"/>')
        o.append(f'<text x="{xx:.1f}" y="{BOTTOM + 20}" font-size="12" text-anchor="middle" fill="{t["muted"]}">{v}</text>')
    o.append(f'<text x="{(LEFT + RIGHT) / 2}" y="{BOTTOM + 42}" font-size="12.5" text-anchor="middle" fill="{t["ink2"]}">'
             'Weight size (MB) — smaller is better</text>')
    o.append(f'<text x="{LEFT - 46}" y="{TOP - 12}" font-size="12.5" fill="{t["ink2"]}">Token F1 — higher is better</text>')
    ref = sy(54.89)
    o.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{ref:.1f}" y2="{ref:.1f}" stroke="{t["ref"]}" stroke-width="1" '
             'stroke-dasharray="4 4" opacity="0.7"/>')
    o.append(f'<text x="{LEFT + 6}" y="{ref - 6:.1f}" font-size="11.5" fill="{t["muted"]}">unpruned quality, 54.89</text>')
    for label, group, size, f1, dx, dy, anchor in POINTS:
        x, y = sx(size), sy(f1)
        o.append(marker(group, x, y, t))
        o.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="12" text-anchor="{anchor}" fill="{t["ink"]}">'
                 f'{label} <tspan fill="{t["ink2"]}">{f1:.1f}</tspan></text>')
    o.append(f'<text x="{LEFT - 46}" y="{H - 14}" font-size="11" fill="{t["muted"]}">'
             'Qwen2.5-0.5B-Instruct on Doc2Dial replies. Token F1 is word overlap, not accuracy. '
             'Original model without training: 22.1 (off the chart).</text>')
    o.append('</svg>')
    return "\n".join(o) + "\n"


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = out / f"f1-vs-size-{theme}.svg"
        path.write_text(draw(theme), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
