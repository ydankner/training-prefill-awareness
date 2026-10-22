"""Pack the paper's figure data (paper/figures/*.dat) into data.js for the slides.

Run from anywhere:  py -3 presentation/build_data.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "paper" / "figures"


def read_dat(name):
    rows = [l.split() for l in (FIG / name).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("%")]
    header, body = rows[0], rows[1:]
    return [dict(zip(header, map(float, r))) for r in body]


def grid(rows, x, y, v, nx, ny, y0):
    g = [[0.0] * nx for _ in range(ny)]
    for r in rows:
        g[int(r[y]) - y0][int(r[x])] = round(max(r[v], 0.0) if v == "excess" else r[v], 4)
    return g


data = {
    # Qwen3-32B: 64 layers (1-indexed) x 64 heads, attention to plant minus average token
    "attn32_base": grid(read_dat("attention-excess-base.dat"), "head", "layer", "excess", 64, 64, 1),
    "attn32_ft": grid(read_dat("attention-excess-ft.dat"), "head", "layer", "excess", 64, 64, 1),
    # Qwen3.8-27B: 16 softmax layers (every 4th) x 24 heads
    "attn38_ft": grid(read_dat("attention38-softmax-ft.dat"), "head", "row", "excess", 24, 16, 1),
    # Steering a clean token: 65 layers (0 = embedding) x 21 alpha steps, log-odds clipped to +-10
    "steer": grid(read_dat("steering-accessible-ftpair.dat"), "aidx", "layer", "logodds", 21, 65, 0),
    "steer_exact": [[int(r["aidx"]), int(r["layer"])] for r in read_dat("steering-accessible-ftpair-exact.dat")],
}

(HERE / "data.js").write_text("window.FIGDATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
                              encoding="utf-8")
print("wrote", HERE / "data.js")
