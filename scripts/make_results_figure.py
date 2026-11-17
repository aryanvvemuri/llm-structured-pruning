"""Draw the README figures as dependency-free SVG, in light and dark.

    python scripts/make_results_figure.py docs/figures

Two figures:

* ``speed-vs-quality``: token F1 against batched generation speed relative to
  the unpruned model (PyTorch on an Apple GPU, float16, batch 16, 50 repeats;
  ``scripts/speed_benchmark.py``). Horizontal bars are 95% intervals.
* ``f1-vs-size``: token F1 against megabytes of weights (2 bytes per parameter
  for 16-bit models, measured MLX weight files for 8-bit and 4-bit).

F1 values are 5-seed dev-holdout means from RESULTS.md (runs support-v2-longer,
-allocator, -quant and -layer-drop, seeds 20261114-20261118). The numbers are
typed in below; edit them here when results change.
"""
import sys
from pathlib import Path

W = 880
LEFT, RIGHT = 70, 850
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "prune": "#2a78d6", "quant": "#eb6834", "layer": "#1baf7a", "ref": "#52514e"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "prune": "#3987e5", "quant": "#d95926", "layer": "#199e70", "ref": "#c3c2b7"},
}
NAMES = {"ref": "Unpruned", "prune": "MLP pruning", "layer": "Layer dropping", "quant": "Quantized (MLX)"}

UNPRUNED_F1 = 54.89

# Speed figure. (label, group, % faster than unpruned, interval low, high, F1, dx, dy, anchor)
SPEED = {
    "name": "speed-vs-quality",
    "title": "Pruning 20% of the model makes it 7–10% faster with no detectable quality loss",
    "subtitle": ["Token F1 (development holdout, mean of 5 seeds) against batched generation speed. "
                 "Bars are 95% intervals on speed.",
                 "Seed-to-seed sd is 0.4–1.3 F1, so points closer than about 1.5 F1 are not reliably different."],
    "desc": "Uniform 20 percent pruning is about 7 percent faster than the unpruned model and dropping three "
            "layers as well is about 10 percent faster, both within noise of the unpruned score. Dropping seven "
            "layers is about 23 percent faster but loses 4.6 F1.",
    "xlabel": "Generation speed against the unpruned model (batch 16) — further right is faster",
    "footer": "Qwen2.5-0.5B-Instruct on Doc2Dial replies. PyTorch, float16, one Apple GPU, 64 forced tokens, "
              "50 repeats. Token F1 is word overlap, not accuracy.",
    "x": (-3.0, 27.0), "xticks": [0, 5, 10, 15, 20, 25], "xfmt": lambda v: f"+{v}%" if v else "0%",
    "y": (49.0, 56.0), "yticks": [50, 52, 54, 56],
    "legend": ["ref", "prune", "layer"],
    "points": [
        ("Unpruned", "ref", 0.0, None, None, 54.89, 0, 22, "middle"),
        ("Uniform 20%", "prune", 7.2, 6.3, 8.8, 53.29, 0, 22, "middle"),
        ("Drop 3 layers", "layer", 5.9, 4.6, 7.5, 54.98, 0, -14, "middle"),
        ("Drop 3 layers + uniform 20%", "layer", 10.0, 7.6, 11.8, 53.92, 56, 4, "start"),
        ("Drop 7 layers", "layer", 22.6, 21.2, 24.4, 50.25, 0, -14, "middle"),
    ],
}

# Size figure. (label, group, MB, None, None, F1, dx, dy, anchor)
SIZE = {
    "name": "f1-vs-size",
    "ref_label": "left",
    "title": "8-bit quantization halves memory at no quality cost, and stacks with pruning",
    "subtitle": ["Token F1 (development holdout, mean of 5 seeds) against weight size. "
                 "Seed-to-seed sd is 0.4–1.6 F1,",
                 "so points closer than about 1.5 F1 are not reliably different."],
    "desc": "8-bit quantization keeps the unpruned score at about half the size. 20 percent MLP pruning costs "
            "at most about 1.6 F1. 4-bit quantization and dropping seven layers cost 3 to 6 F1.",
    "xlabel": "Weight size (MB) — further left is smaller",
    "footer": "Qwen2.5-0.5B-Instruct on Doc2Dial replies. Token F1 is word overlap, not accuracy. "
              "Original model without training: 22.1 (off the chart).",
    "x": (150.0, 1070.0), "xticks": [200, 400, 600, 800, 1000], "xfmt": str,
    "y": (48.0, 56.0), "yticks": [48, 50, 52, 54, 56],
    "legend": ["ref", "prune", "layer", "quant"],
    "points": [
        ("Unpruned", "ref", 988.1, None, None, 54.89, -8, -13, "start"),
        ("Uniform 10%", "prune", 889.0, None, None, 54.32, 0, 21, "middle"),
        ("Uniform 20%", "prune", 789.9, None, None, 53.29, -12, 4, "end"),
        ("Curve 20%", "prune", 790.6, None, None, 54.58, -12, 4, "end"),
        ("Drop 3 layers", "layer", 898.6, None, None, 54.98, 0, -13, "middle"),
        ("Drop 3 layers + uniform 20%", "layer", 790.2, None, None, 53.92, -12, 4, "end"),
        ("Drop 7 layers", "layer", 779.3, None, None, 50.25, 12, 4, "start"),
        ("Unpruned, 8-bit", "quant", 525.0, None, None, 54.91, 0, -13, "middle"),
        ("Uniform 20%, 8-bit", "quant", 419.8, None, None, 53.30, 0, 21, "middle"),
        ("Unpruned, 4-bit", "quant", 278.1, None, None, 50.52, 12, 4, "start"),
        ("Uniform 20%, 4-bit", "quant", 222.3, None, None, 48.62, 12, 4, "start"),
    ],
}
FIGURES = [SPEED, SIZE]


def marker(group, x, y, t, r=6.0):
    """Shape repeats the colour, so identity never rests on colour alone."""
    ring = f'stroke="{t["surface"]}" stroke-width="2"'
    fill = f'fill="{t[group]}"'
    if group == "quant":
        return (f'<rect x="{x - r + 0.5:.1f}" y="{y - r + 0.5:.1f}" width="{2 * r - 1:.1f}" '
                f'height="{2 * r - 1:.1f}" rx="1.5" {fill} {ring}/>')
    if group == "layer":
        d = r + 1.5
        return (f'<path d="M{x:.1f} {y - d:.1f} L{x + d:.1f} {y:.1f} L{x:.1f} {y + d:.1f} '
                f'L{x - d:.1f} {y:.1f} Z" {fill} {ring}/>')
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {fill} {ring}/>'


def draw(fig, theme):
    t = THEMES[theme]
    top, bottom, height = 142, 452, 536
    x0, x1 = fig["x"]
    y0, y1 = fig["y"]

    def sx(v):
        return LEFT + (v - x0) / (x1 - x0) * (RIGHT - LEFT)

    def sy(v):
        return bottom - (v - y0) / (y1 - y0) * (bottom - top)

    edge = LEFT - 46
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
         f'font-family="{FONT}" role="img" aria-labelledby="t d">',
         f'<title id="t">{fig["title"]}</title>',
         f'<desc id="d">{fig["desc"]}</desc>',
         f'<rect width="{W}" height="{height}" rx="8" fill="{t["surface"]}"/>',
         f'<text x="{edge}" y="34" font-size="17" font-weight="600" fill="{t["ink"]}">{fig["title"]}</text>']
    for i, line in enumerate(fig["subtitle"]):
        o.append(f'<text x="{edge}" y="{56 + 17 * i}" font-size="12.5" fill="{t["ink2"]}">{line}</text>')
    x = edge
    for group in fig["legend"]:
        o.append(marker(group, x + 7, 98, t, 5.0))
        o.append(f'<text x="{x + 20}" y="102" font-size="12.5" fill="{t["ink2"]}">{NAMES[group]}</text>')
        x += 40 + 7.1 * len(NAMES[group])
    for v in fig["yticks"]:
        y = sy(v)
        o.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{LEFT - 10}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{t["muted"]}">{v}</text>')
    o.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{bottom}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')
    for v in fig["xticks"]:
        xx = sx(v)
        o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{bottom}" y2="{bottom + 5}" stroke="{t["axis"]}" stroke-width="1"/>')
        o.append(f'<text x="{xx:.1f}" y="{bottom + 20}" font-size="12" text-anchor="middle" '
                 f'fill="{t["muted"]}">{fig["xfmt"](v)}</text>')
    o.append(f'<text x="{(LEFT + RIGHT) / 2}" y="{bottom + 42}" font-size="12.5" text-anchor="middle" '
             f'fill="{t["ink2"]}">{fig["xlabel"]}</text>')
    o.append(f'<text x="{edge}" y="{top - 12}" font-size="12.5" fill="{t["ink2"]}">Token F1 — higher is better</text>')
    ref = sy(UNPRUNED_F1)
    o.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{ref:.1f}" y2="{ref:.1f}" stroke="{t["ref"]}" stroke-width="1" '
             'stroke-dasharray="4 4" opacity="0.7"/>')
    if fig.get("ref_label") == "left":
        o.append(f'<text x="{LEFT + 6}" y="{ref - 6:.1f}" font-size="11.5" '
                 f'fill="{t["muted"]}">unpruned quality, {UNPRUNED_F1}</text>')
    else:
        o.append(f'<text x="{RIGHT - 6}" y="{ref + 15:.1f}" font-size="11.5" text-anchor="end" '
                 f'fill="{t["muted"]}">unpruned quality, {UNPRUNED_F1}</text>')
    for label, group, value, low, high, f1, dx, dy, anchor in fig["points"]:
        if low is not None:
            y = sy(f1)
            o.append(f'<line x1="{sx(low):.1f}" x2="{sx(high):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="{t[group]}" stroke-width="2" stroke-linecap="round" opacity="0.55"/>')
    for label, group, value, low, high, f1, dx, dy, anchor in fig["points"]:
        x, y = sx(value), sy(f1)
        o.append(marker(group, x, y, t))
        note = f"{f1:.1f}" if low is None else f"+{value:.0f}%, {f1:.1f} F1"
        o.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="12" text-anchor="{anchor}" fill="{t["ink"]}">'
                 f'{label} <tspan fill="{t["ink2"]}">{note}</tspan></text>')
    o.append(f'<text x="{edge}" y="{height - 14}" font-size="11" fill="{t["muted"]}">{fig["footer"]}</text>')
    o.append('</svg>')
    return "\n".join(o) + "\n"


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    for fig in FIGURES:
        for theme in THEMES:
            path = out / f'{fig["name"]}-{theme}.svg'
            path.write_text(draw(fig, theme), encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
