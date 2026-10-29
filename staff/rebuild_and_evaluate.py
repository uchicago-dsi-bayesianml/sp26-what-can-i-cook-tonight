#!/usr/bin/env python3
"""Course-staff check: rebuild the corpus and the model, look at the user posterior, and
evaluate the ranking on held-out recipes.

Needs the public Food.com archive from Kaggle (Majumder, Li et al., EMNLP 2019). Kaggle
serves it without a login at
    https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions
as a 280 MB `archive.zip`; download it anywhere outside the repository and pass its path.
The archive is read in place and nothing is written to the repository.

The script imports the authors' own functions from model/src/recipe_recommender.py, so it
needs the authors' model dependencies (numpy, pandas, scipy, scikit-learn, joblib, inflect),
but not torch, gradio or the detector. It takes about a minute on an 18-core laptop once the
archive is downloaded. The parallel fits are not bit-for-bit deterministic, so the third
decimal of the evaluation can move between runs.

  1. Rebuilds data/recipes_clean.csv in memory exactly as data/prepare_data.py does, and
     counts the zero ratings that the rating average keeps.
  2. Fits the authors' train_lda() with its defaults on the full corpus, prints the topics
     and the model-selection table, and re-runs the two demo pantries of run_demo.py, to
     compare with model/results.json. It then extends the authors' own K sweep
     (_select_k_perplexity, same settings) down to K = 2.
  3. Re-derives the user's topic posterior in NumPy (recipe_recommender.py lines 565-583)
     for four of the authors' pantries, prints the log-likelihood gap between the two best
     topics, and sets it beside the posterior that LDA itself gives for the same pantry (the
     variational Dirichlet over topic weights, from the same scikit-learn fit). Then prints
     the per-draw flavor alignment behind posterior_uncertainty for two Top-5 lists.
  4. Held-out evaluation. The model is refit on 90 percent of the recipes. For each of N
     held-out recipes with 4 to 12 ingredients, the pantry is the recipe minus one random
     ingredient plus three random distractors drawn by ingredient frequency. The authors'
     filter_candidates() and score_recipes() rank all candidates, and the same candidates
     are re-ranked by coverage only, Jaccard similarity, and the authors' score with the
     flavor-alignment factor removed. Reports where the held-out recipe lands.

Usage:  python staff/rebuild_and_evaluate.py --archive PATH/archive.zip [--queries 300] [--seed 0]
"""
import argparse
import json
import pathlib
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model" / "src"))
import recipe_recommender as rr  # noqa: E402  (the authors' module)

# The authors' demo pantries: model/src/run_demo.py, presentation.ipynb cell 31 (fridge.jpg,
# after mapping) and model/notebooks/usage_example.ipynb cell 4.
PANTRIES = {
    "Italian-ish": ["chicken", "garlic", "onion", "olive oil", "tomato", "basil",
                    "parmesan cheese", "pasta", "salt", "pepper"],
    "Baker's pantry": ["flour", "sugar", "butter", "egg", "vanilla", "baking soda",
                       "milk", "salt", "chocolate"],
    "fridge.jpg": ["garlic", "broccoli", "cherry tomato", "egg", "onion"],
    "usage_example": ["chicken", "garlic", "onion", "olive oil", "tomato", "rice",
                      "salt", "black pepper"],
}


# --------------------------------------------------------------------------- 1. corpus
def build_corpus(archive):
    """data/prepare_data.py lines 41-101, reading the two CSVs straight from the zip."""
    with zipfile.ZipFile(archive) as zf:
        chunks, dates = [], []
        with zf.open("RAW_interactions.csv") as f:
            for ch in pd.read_csv(f, usecols=["recipe_id", "date", "rating"], chunksize=200_000):
                dates += [ch["date"].min(), ch["date"].max()]
                chunks.append(ch.groupby("recipe_id")["rating"].agg(
                    n_ratings="count", rating_sum="sum",
                    n_zero=lambda s: int((s == 0).sum())))
        with zf.open("RAW_recipes.csv") as f:
            rec = pd.read_csv(f, usecols=["id", "name", "ingredients", "n_ingredients"],
                              low_memory=False)
    agg = pd.concat(chunks).groupby(level=0).sum()
    raw = (len(rec), int(agg["n_ratings"].sum()), min(dates), max(dates))
    agg["avg_rating"] = agg["rating_sum"] / agg["n_ratings"]
    df = rec.merge(agg, left_on="id", right_index=True, how="inner")
    df = df.dropna(subset=["name", "ingredients", "n_ingredients"])
    df = df[df["n_ratings"] >= 5]
    df = df[(df["n_ingredients"] >= 2) & (df["n_ingredients"] <= 35)]
    cache = {}

    def norm_list(cell):
        out, seen = [], set()
        for it in rr._coerce_ingredients(cell):
            if it not in cache:
                cache[it] = rr.normalize_token(it)
            t = cache[it]
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    df = df.assign(ingredients_norm=df["ingredients"].map(norm_list))
    df = df[df["ingredients_norm"].map(len) >= 2]
    out = pd.DataFrame({
        "recipe_id": df["id"].values,
        "recipe_name": df["name"].values,
        "ingredients": df["ingredients_norm"].map(json.dumps).values,
        "avg_rating": df["avg_rating"].round(4).values,
        "n_ratings": df["n_ratings"].astype(int).values,
        "n_zero": df["n_zero"].astype(int).values,
        "rating_sum": df["rating_sum"].values,
    })
    out.attrs["raw"] = raw
    return out


def report_corpus(df):
    lists = rr._get_recipe_ingredients(df)
    print("=== 1. Corpus, rebuilt as data/prepare_data.py does ===")
    n_rec, n_int, d0, d1 = df.attrs["raw"]
    print(f"raw files: {n_rec:,} recipes, {n_int:,} interactions dated {d0} to {d1}")
    print(f"recipes: {len(df):,}   distinct ingredient tokens: {len(set().union(*lists)):,}")
    nz = df["n_zero"]
    print(f"ratings on these recipes: {df['n_ratings'].sum():,}, of which {nz.sum():,} are 0 "
          f"({nz.sum() / df['n_ratings'].sum():.1%}); Food.com stores a review without a rating as 0")
    print(f"recipes with at least one 0 in their average: {(nz > 0).sum():,} ({(nz > 0).mean():.1%})")
    rated = df["n_ratings"] - nz
    no_zero = (df["rating_sum"] / rated.where(rated > 0)).dropna()
    print(f"mean avg_rating with zeros (as built): {df['avg_rating'].mean():.3f}; "
          f"without zeros: {no_zero.mean():.3f}; recipes whose average is exactly 0: "
          f"{(df['avg_rating'] == 0).sum()}")
    print(f"recipes left with fewer than 5 non-zero ratings: {(rated < 5).sum():,}")


# --------------------------------------------------------------------------- 2. model
def report_model(model, X, committed):
    print("\n=== 2. The authors' train_lda() with its defaults, full corpus ===")
    print(f"K={model.best_k}, vocab V={len(model.vocab)}, {model.n_samples} Bootstrap draws, "
          f"bootstrap_stability={model.bootstrap_stability:.6f} "
          f"(committed: {committed['bootstrap_stability']})")
    print("perplexity by K:", ", ".join(f"K={int(r.K)}: {r.holdout_perplexity:.1f}"
                                         for r in model.perplexity_table.itertuples()))
    for k, lab in enumerate(model.topic_labels):
        print(f"  topic {k}: {lab:<40} P(topic)={model.topic_prior[k]:.3f}")
    print("committed topics:", "; ".join(t["label"] for t in committed["topics"]))
    best, table = _quiet(rr._select_k_perplexity, X, (2, 3, 4, 5, 6), rr.ALPHA_PRIOR,
                         rr.BETA_PRIOR, 42, max_iter=8)
    print("extended sweep (same function, same settings), perplexity by K:",
          ", ".join(f"K={int(r.K)}: {r.holdout_perplexity:.1f}" for r in table.itertuples()),
          f"-> argmin K={best}")


def report_demo_repro(df, committed):
    """Re-run the two demo pantries of model/src/run_demo.py and compare with results.json."""
    for name in ("Italian-ish", "Baker's pantry"):
        recs = _quiet(rr.recommend, PANTRIES[name], df, top_n=5, model_path="/nonexistent")
        ours = [(r["recipe_name"], r["score"]) for r in recs]
        theirs = [(r["recipe_name"], r["score"]) for r in committed["recommendations"][name]["top5"]]
        same = [a[0] == b[0] for a, b in zip(ours, theirs)]
        gap = max(abs(a[1] - b[1]) for a, b in zip(ours, theirs))
        print(f"demo Top-5 '{name}': {sum(same)} of 5 recipes in the committed order; "
              f"largest score difference {gap:.4f}")


# --------------------------------------------------------------------------- 3. posterior
def user_posterior_numpy(pantry, model):
    """recipe_recommender.py lines 565-583, written out: P(k | ingredients) for each draw s,
    with every ingredient assumed to come from one and the same topic k."""
    idx = [model.ingr2idx[t] for t in rr.normalize_list(pantry) if t in model.ingr2idx]
    loglik = np.log(model.phi_samples[:, :, idx] + 1e-12).sum(axis=2)       # (S, K)
    logpost = loglik + np.log(model.topic_prior + 1e-12)[None, :]
    logpost -= logpost.max(axis=1, keepdims=True)
    post = np.exp(logpost)
    return post / post.sum(axis=1, keepdims=True), loglik, idx


def report_posterior(model, lda, rng):
    print("\n=== 3. The user's topic posterior: the authors' formula vs LDA's own posterior ===")
    for name, pantry in PANTRIES.items():
        post, loglik, idx = user_posterior_numpy(pantry, model)
        _, theirs = rr.infer_user_posterior(pantry, model)
        assert np.allclose(post, theirs), "NumPy re-derivation differs from the authors' function"
        m = post.mean(0)
        top = int(m.argmax())
        ll = np.sort(loglik.mean(0))[::-1]
        flips = float((post.argmax(1) != top).mean())
        print(f"\n{name}: {len(idx)} of {len(pantry)} ingredients in the vocabulary")
        print("  authors' single-topic posterior (mean over draws): "
              + "  ".join(f"T{k}={p:.4f}" for k, p in enumerate(m)))
        print(f"  log-likelihood gap, best minus second topic: {ll[0] - ll[1]:.1f} nats; "
              f"draws whose top topic differs: {flips:.0%}")
        x = np.zeros((1, len(model.vocab)))
        x[0, idx] = 1.0
        gamma = lda.transform(x, normalize=False)[0]
        theta = rng.dirichlet(gamma, size=20_000)
        lo, hi = np.percentile(theta, [5, 95], axis=0)
        print("  LDA's variational posterior over topic weights, mean [90% interval]: "
              + "  ".join(f"T{k}={gamma[k] / gamma.sum():.2f} [{lo[k]:.2f}, {hi[k]:.2f}]"
                          for k in range(len(gamma))))


def report_alignment(model, df, name):
    """Per-draw flavor alignment exp(-KL) for a pantry's Top-5 (recipe_recommender.py
    lines 589-601), the quantity behind posterior_uncertainty and fig4."""
    print(f"\n--- per-draw flavor alignment over the Bootstrap draws, '{name}' Top-5 ---")
    pantry = PANTRIES[name]
    recs = _quiet(rr.recommend, pantry, df, top_n=5, model_path="/nonexistent")
    _, upost = rr.infer_user_posterior(pantry, model)
    cand = _quiet(rr.filter_candidates, pantry, df)
    byname = {r["recipe_name"]: r for _, r in cand.iterrows()}
    for r in recs:
        _, rpost = rr.infer_user_posterior(byname[r["recipe_name"]]["_ingredients_norm"], model)
        kl = np.sum(rpost * (np.log(rpost + 1e-12) - np.log(upost + 1e-12)), axis=1)
        sim = np.exp(-kl)
        print(f"  {r['recipe_name'][:40]:<41} mean={sim.mean():.3f} std={sim.std():.3f} "
              f"1-min={1 - sim.min():.1e}  draws above 0.9: {(sim > 0.9).mean():.0%}, "
              f"below 0.1: {(sim < 0.1).mean():.0%}")


# --------------------------------------------------------------------------- 4. evaluation
def rank_of(scores, target):
    """1-based rank of the target, with ties broken at random in expectation."""
    s = scores[target]
    higher = int((scores > s).sum())
    ties = int((scores == s).sum()) - 1
    return 1 + higher + ties / 2


def evaluate(df_all, n_queries, seed):
    print("\n=== 4. Held-out evaluation ===")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df_all))
    n_test = len(df_all) // 10
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    t0 = time.time()
    model = _quiet(rr.train_lda, df_train)
    print(f"model refit on {len(df_train):,} training recipes in {time.time() - t0:.0f} s, K={model.best_k}")

    lists = rr._get_recipe_ingredients(df_all)
    vocab = model.vocab
    dfc = Counter()
    for lst in rr._get_recipe_ingredients(df_train):
        dfc.update(set(lst))
    freq = np.array([dfc[w] for w in vocab], dtype=float)
    freq /= freq.sum()

    mu = float(df_all["avg_rating"].mean())
    methods = ["authors' score", "score without alignment", "coverage only", "Jaccard"]
    ranks = {m: [] for m in methods}
    n_cand, maxpost = [], []
    done = 0
    for i in test_idx:
        recipe = lists.iloc[i]
        if not 4 <= len(recipe) <= 12:
            continue
        drop = recipe[rng.integers(len(recipe))]
        keep = [w for w in recipe if w != drop]
        pool = [j for j in rng.choice(len(vocab), size=20, replace=False, p=freq)
                if vocab[j] not in recipe][:3]
        pantry = keep + [vocab[j] for j in pool]
        cand = _quiet(rr.filter_candidates, pantry, df_all)
        if cand.empty or df_all["recipe_id"].iloc[i] not in set(cand["recipe_id"]):
            continue
        tpos = int(np.flatnonzero(cand["recipe_id"].values == df_all["recipe_id"].iloc[i])[0])
        _, upost = rr.infer_user_posterior(pantry, model)
        maxpost.append(float(upost.mean(0).max()))
        # score_recipes sorts its output; label each candidate by position to map back
        labelled = cand.assign(recipe_name=[str(j) for j in range(len(cand))])
        res = rr.score_recipes(pantry, labelled, upost, model, df_all, return_internal=True)
        full = np.empty(len(cand))
        for r in res:
            full[int(r["recipe_name"])] = r["_score"]
        cov = cand["coverage"].values.astype(float)
        inter = cand["_inter"].values.astype(float)
        size = cand["_size"].values.astype(float)
        rating = np.array([rr._bayesian_rating(a, n, mu) for a, n in
                           zip(cand["avg_rating"].values, cand["n_ratings"].values)])
        noalign = cov ** 2 * (1 - np.exp(-inter / rr.SCORE_TAU)) * rating
        jacc = inter / (len(set(rr.normalize_list(pantry))) + size - inter)
        covonly = cov + 1e-6 * inter          # coverage, ties broken by matched count
        for m, sc in zip(methods, [full, noalign, covonly, jacc]):
            ranks[m].append(rank_of(np.round(sc, 12), tpos))
        n_cand.append(len(cand))
        done += 1
        if done >= n_queries:
            break
    print(f"{done} queries; candidates per query: median {int(np.median(n_cand))}, "
          f"range {min(n_cand)} to {max(n_cand)}")
    print("(the pantry is built from the held-out recipe, so this proxy favours Jaccard by "
          "construction; the clean comparison is the first two rows, which differ only in the "
          "flavor-alignment factor)")
    mp = np.array(maxpost)
    print(f"user posterior: top topic above 0.99 in {(mp > 0.99).mean():.0%} of queries, "
          f"above 0.9 in {(mp > 0.9).mean():.0%}")
    print(f"  {'ranking':<26}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'MRR':>8}{'median rank':>13}")
    for m in methods:
        r = np.array(ranks[m])
        print(f"  {m:<26}{(r <= 1).mean():>8.3f}{(r <= 5).mean():>8.3f}{(r <= 10).mean():>8.3f}"
              f"{(1 / r).mean():>8.3f}{np.median(r):>13.1f}")
    a, b = np.array(ranks[methods[0]]), np.array(ranks[methods[1]])
    print(f"authors' score vs the same score without alignment: better in {(a < b).mean():.0%} "
          f"of queries, worse in {(a > b).mean():.0%}, equal in {(a == b).mean():.0%}")


def _quiet(fn, *args, **kw):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", required=True, help="path to the Food.com archive.zip")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    df = build_corpus(args.archive)
    report_corpus(df)

    committed = json.loads((REPO / "model" / "results.json").read_text(encoding="utf-8"))
    model = _quiet(rr.train_lda, df)
    X, _, _ = rr._build_doc_term_matrix(df, 500, 5)
    lda = rr._fit_sklearn_lda(X, model.best_k, rr.ALPHA_PRIOR, rr.BETA_PRIOR, max_iter=8,
                              random_state=42, n_jobs=-1)
    phi_hat = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    assert np.allclose(phi_hat, model.phi_mean), "refit differs from train_lda's main fit"
    report_model(model, X, committed)
    rr._STATE.update(model=model, df_id=id(df))
    report_demo_repro(df, committed)
    report_posterior(model, lda, rng)
    report_alignment(model, df, "Italian-ish")
    report_alignment(model, df, "usage_example")
    print(f"\n(sections 1-3 took {time.time() - t0:.0f} s)")
    evaluate(df, args.queries, args.seed)
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
