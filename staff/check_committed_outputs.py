#!/usr/bin/env python3
"""Course-staff check: what do the committed outputs say about the score and its "uncertainty"?

Reads only files that are in the repository: model/results.json, model/model_selection.json
and the stored outputs of presentation.ipynb and model/notebooks/usage_example.ipynb. Nothing
is trained or downloaded.

For every committed recommendation it recomputes the three non-LDA factors of the authors'
score (model/src/recipe_recommender.py, lines 651-654),

    score = coverage^2 * (1 - exp(-matched / 4)) * flavor_alignment * rating,

from the printed coverage ("a/b ingredients") and the printed predicted_rating, and divides
the printed score by them. What is left is the flavor-alignment factor, the only place the
LDA topics enter the ranking. It also prints the posterior_uncertainty values, the "Match"
percentage the current app would draw for each card (app.py, line 502), and the
model-selection table.

Rounding: scores are printed to 4 decimals and ratings to 3, so the implied alignment is good
to about 0.001.

Standard library only:  python staff/check_committed_outputs.py [--repo PATH]
"""
import argparse
import json
import math
import pathlib
import re

TAU = 4.0  # SCORE_TAU, recipe_recommender.py line 67


def implied_alignment(score, coverage_text, rating):
    a, b = (int(x) for x in re.match(r"\s*(\d+)/(\d+)", coverage_text).groups())
    rest = (a / b) ** 2 * (1.0 - math.exp(-a / TAU)) * rating
    return score / rest, a, b


def match_pct(score):
    """The width of the 'Match' bar on each recipe card (app.py lines 501-502)."""
    return min(100, max(0, int(float(score) * 100)))


def parse_pandas_repr(text):
    """Rows of a pandas DataFrame repr, including the wrapped form with ' \\' headers.

    pandas right-justifies every column to its header, so a column's cells end where its
    header ends. Headers are split on runs of two or more spaces."""
    blocks, cur = [], []
    for line in text.splitlines():
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    rows = {}
    for block in blocks:
        header = block[0].rstrip().rstrip("\\").rstrip()
        cols = [(m.group(0), m.end()) for m in re.finditer(r"\S+(?: \S+)*", header)]
        for line in block[1:]:
            idx = line.split()[0]
            start = len(line) - len(line.lstrip()) + len(idx)
            row = rows.setdefault(idx, {})
            prev = start
            for name, end in cols:
                row[name] = line[prev:end].strip()
                prev = end
    return [rows[k] for k in sorted(rows, key=int)]


def cell_table(nb_path, cell_no):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for out in nb["cells"][cell_no].get("outputs", []):
        data = out.get("data", {})
        if "text/plain" in data and out.get("output_type") == "execute_result":
            return parse_pandas_repr("".join(data["text/plain"]))
    raise SystemExit(f"no table in {nb_path} cell {cell_no}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=pathlib.Path(__file__).resolve().parents[1], type=pathlib.Path)
    repo = ap.parse_args().repo

    print("=== 1. model/results.json: the four demo pantries, Top-5 each ===")
    res = json.loads((repo / "model" / "results.json").read_text(encoding="utf-8"))
    unc, aligns, matches = [], [], []
    for name, block in res["recommendations"].items():
        print(f"\n{name}  ({len(block['ingredients'])} ingredients)")
        print(f"  {'recipe':<44}{'score':>8}{'cover':>7}{'rating':>8}{'align':>8}{'uncert':>8}{'match%':>8}")
        for r in block["top5"]:
            al, a, b = implied_alignment(r["score"], r["coverage"], r["predicted_rating"])
            aligns.append(al)
            unc.append(r["posterior_uncertainty"])
            matches.append(match_pct(r["score"]))
            print(f"  {r['recipe_name'][:43]:<44}{r['score']:>8.4f}{a:>4}/{b:<2}{r['predicted_rating']:>8.3f}"
                  f"{al:>8.3f}{r['posterior_uncertainty']:>8.4f}{matches[-1]:>8}")
    n = len(unc)
    print(f"\n{n} recommendations: implied alignment between {min(aligns):.3f} and {max(aligns):.3f}")
    print(f"posterior_uncertainty: {sum(u == 0 for u in unc)} of {n} are exactly 0.0, maximum {max(unc)}")
    print(f"Match bar: {sum(m == 100 for m in matches)} of {n} cards would show 100%")

    print("\n=== 2. presentation.ipynb cell 23: the authors' own score anatomy (Italian-ish Top-5) ===")
    for r in cell_table(repo / "presentation.ipynb", 23):
        print(f"  {r['recipe'][:40]:<41} coverage^2={r['coverage^2']}  overlap={r['overlap_bonus']}"
              f"  alignment={r['alignment']}  rating={r['rating']}  score={r['= score']}")

    print("\n=== 3. Photo pantries: presentation.ipynb cells 33 (fridge.jpg) and 35 (vegetables.jpg) ===")
    for cell in (33, 35):
        print(f"cell {cell}")
        for r in cell_table(repo / "presentation.ipynb", cell):
            al, a, b = implied_alignment(float(r["score"]), r["coverage"], float(r["predicted_rating"]))
            print(f"  {r['recipe_name'][:44]:<45} score={float(r['score']):.4f}  {a}/{b}"
                  f"  implied alignment={al:.3f}  match%={match_pct(r['score'])}")

    print("\n=== 4. model/notebooks/usage_example.ipynb cell 4: an 8-ingredient pantry ===")
    for r in cell_table(repo / "model" / "notebooks" / "usage_example.ipynb", 4):
        al, a, b = implied_alignment(float(r["score"]), r["coverage"], float(r["predicted_rating"]))
        print(f"  {r['recipe_name'][:44]:<45} score={float(r['score']):.4f}  {a}/{b}"
              f"  implied alignment={al:.3f}  posterior_uncertainty={r['posterior_uncertainty']}")

    print("\n=== 5. model/model_selection.json: held-out perplexity by K ===")
    sel = json.loads((repo / "model" / "model_selection.json").read_text(encoding="utf-8"))
    for row in sel:
        print(f"  K={row['K']:>2}  perplexity={row['holdout_perplexity']:.1f}")
    best = min(sel, key=lambda r: r["holdout_perplexity"])["K"]
    grid = [r["K"] for r in sel]
    mono = all(x["holdout_perplexity"] < y["holdout_perplexity"] for x, y in zip(sel, sel[1:]))
    print(f"  argmin K={best}; smallest K in the grid={min(grid)}; perplexity rises at every step: {mono}")


if __name__ == "__main__":
    main()
