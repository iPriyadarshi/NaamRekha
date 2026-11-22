"""Phase 6 figures, drawn only from logged results (nothing typed in by hand).

  results/phase6_hindi_comparison.png   entity F1 per type, five systems, Hindi gold test
  results/phase6_error_kinds.png        error kinds per system, Hindi gold test

Sources: results/results.csv (scores) and results/error_analysis/error_counts.csv
(error counts, from notebooks/phase5-cross-stage-errors.ipynb). The Phase 3 scaling
curve and the Phase 4 transfer heatmap already exist and are not redrawn here.

Colors are the first slots of the validated categorical palette (surface #fcfcfb):
  3 series, all pairs:      #2a78d6 #eb6834 #1baf7a  (passes; aqua is below 3:1 contrast)
  4 series, adjacent pairs: + #eda100                (passes; aqua and yellow below 3:1)
Because aqua and yellow are below 3:1 on the light surface, the figures carry visible
labels and every number is also in a table in docs/phase6/report.md.

Run from the repo root:  uv run --with matplotlib python scripts/phase6_figures.py
Add --check to only verify that the data needed for the figures is present.
"""

from __future__ import annotations

import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results", "results.csv")
ERRORS = os.path.join(REPO_ROOT, "results", "error_analysis", "error_counts.csv")
OUT_COMPARISON = os.path.join(REPO_ROOT, "results", "phase6_hindi_comparison.png")
OUT_ERRORS = os.path.join(REPO_ROOT, "results", "phase6_error_kinds.png")

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TYPE_COLORS = {"PER": "#2a78d6", "ORG": "#eb6834", "LOC": "#1baf7a"}
KIND_COLORS = {"boundary": "#2a78d6", "missed": "#eb6834", "spurious": "#1baf7a", "type_confusion": "#eda100"}
KIND_LABELS = {"boundary": "Boundary", "missed": "Missed", "spurious": "Spurious", "type_confusion": "Type confusion"}


def notes_of(row: dict) -> dict[str, str]:
    return dict(p.split("=", 1) for p in row["notes"].split("; ") if "=" in p)


def load_scores() -> list[dict]:
    """The five Hindi systems, each from one row of results.csv."""
    with open(RESULTS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def pick(pred, what: str) -> dict:
        found = [r for r in rows if pred(r)]
        if len(found) != 1:
            raise SystemExit(f"expected exactly one results.csv row for {what}, found {len(found)}")
        return found[0]

    hi = lambda r: r["train_lang"] == "hi" and r["test_lang"] == "hi"
    picked = [
        ("MuRIL (seed 42)", pick(lambda r: r["model"] == "MuRIL" and hi(r) and r["subset_size"] == "300000"
                                 and notes_of(r).get("seed") == "42" and "phase" not in notes_of(r), "MuRIL 300k seed 42 (Phase 3)")),
        ("BiLSTM-CRF", pick(lambda r: r["model"] == "BiLSTM-CRF", "BiLSTM-CRF")),
        ("IndicBERT", pick(lambda r: r["model"] == "IndicBERT", "IndicBERT")),
        ("LLM few-shot", pick(lambda r: r["model"].startswith("LLM-few-shot"), "LLM few-shot")),
        ("LLM zero-shot", pick(lambda r: r["model"].startswith("LLM-zero-shot"), "LLM zero-shot")),
    ]
    return [{"name": n, **{k: float(r[k]) for k in ("F1_PER", "F1_ORG", "F1_LOC", "F1_overall")}} for n, r in picked]


def load_errors() -> list[dict]:
    systems = [("MuRIL (seed 42)", "MuRIL-hi-s42"), ("BiLSTM-CRF", "BiLSTM-CRF"), ("IndicBERT", "IndicBERT"),
               ("LLM few-shot", "LLM-few"), ("LLM zero-shot", "LLM-zero")]
    with open(ERRORS, encoding="utf-8", newline="") as f:
        rows = {r["system"]: r for r in csv.DictReader(f) if r["lang"] == "hi"}
    missing = [s for _, s in systems if s not in rows]
    if missing:
        raise SystemExit(f"error_counts.csv has no Hindi row for {missing}")
    out = []
    for name, key in systems:
        r = rows[key]
        d = {k: int(r[k]) for k in KIND_COLORS}
        assert sum(d.values()) == int(r["total"]), key
        out.append({"name": name, **d, "total": int(r["total"])})
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def draw_comparison(scores: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200, facecolor=SURFACE)
    style(ax)
    types = ("PER", "ORG", "LOC")
    width, gap = 0.25, 0.012
    for j, t in enumerate(types):
        xs = [i + (j - 1) * (width + gap) for i in range(len(scores))]
        ys = [s[f"F1_{t}"] for s in scores]
        ax.bar(xs, ys, width=width, color=TYPE_COLORS[t], edgecolor=SURFACE, linewidth=1.2, label=t, zorder=3)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([f"{s['name']}\noverall {s['F1_overall']:.4f}" for s in scores], color=INK, fontsize=9)
    ax.set_ylabel("Entity F1 (seqeval micro)", color=INK_2, fontsize=9)
    ax.set_title("Hindi gold test (864 sentences): entity F1 by type", loc="left", color=INK, fontsize=11, pad=24)
    # direct labels on the weakest type, the one the analysis keeps returning to
    for i, s in enumerate(scores):
        ax.text(i, s["F1_ORG"] + 0.015, f"{s['F1_ORG']:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    leg = ax.legend(ncols=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_2)
    fig.text(0.01, 0.005, "ORG values labelled. Source: results/results.csv. Every value is also in the report's tables.",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_COMPARISON, facecolor=SURFACE)
    plt.close(fig)


def draw_errors(errs: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=200, facecolor=SURFACE)
    style(ax)
    n = len(errs)
    ys = list(range(n))[::-1]                      # first system on top
    for y, e in zip(ys, errs):
        left = 0
        for kind in KIND_COLORS:
            ax.barh(y, e[kind], left=left, height=0.55, color=KIND_COLORS[kind], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if e[kind] >= 45:                       # label a segment only where the number fits
                ax.text(left + e[kind] / 2, y, str(e[kind]), ha="center", va="center", fontsize=8, color="#ffffff"
                        if kind in ("boundary",) else INK, zorder=4)
            left += e[kind]
        ax.text(left + 14, y, f"{e['total']:,}", ha="left", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([e["name"] for e in errs], color=INK, fontsize=9)
    ax.set_xlim(0, max(e["total"] for e in errs) * 1.12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Entity errors on the 864 Hindi gold sentences", color=INK_2, fontsize=9)
    ax.set_title("Errors by kind, fewest to most", loc="left", color=INK, fontsize=11, pad=24)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLORS[k]) for k in KIND_COLORS]
    leg = ax.legend(handles, [KIND_LABELS[k] for k in KIND_COLORS], ncols=4, frameon=False, loc="lower left",
                    bbox_to_anchor=(0, 1.0), fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_2)
    fig.text(0.01, 0.005, "Segments under 45 errors are unlabelled. Source: results/error_analysis/error_counts.csv.",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_ERRORS, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    scores, errs = load_scores(), load_errors()
    print("scores:", [(s["name"], round(s["F1_overall"], 4)) for s in scores])
    print("errors:", [(e["name"], e["total"]) for e in errs])
    if "--check" in sys.argv:
        print("data for both figures is present")
        return
    draw_comparison(scores)
    draw_errors(errs)
    print("wrote", os.path.relpath(OUT_COMPARISON, REPO_ROOT), "and", os.path.relpath(OUT_ERRORS, REPO_ROOT))


if __name__ == "__main__":
    main()
