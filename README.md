<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# What Can I Cook Tonight?

> **Example project from Spring 2026.** This is a copy of [eason1218/what-can-i-cook-tonight](https://github.com/eason1218/what-can-i-cook-tonight), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project). The Food.com recipe data is "© Original Authors" and is not in this repository; [data/README.md](data/README.md) explains how to download it and rebuild the corpus.

Can a photo of what is in your fridge become a short list of recipes you can mostly cook tonight, ranked by a Bayesian model that also says how sure it is? The team took 53,573 Food.com recipes, treated each recipe as a document and each ingredient as a word, and fit a latent Dirichlet allocation (LDA) topic model with four "flavor topics" by scikit-learn's variational Bayes. Recipes the pantry mostly covers are ranked by coverage, a flavor-alignment term from the topics and a shrunk rating, and fifty refits on resampled recipes give each recommendation an uncertainty. In front sit a pre-trained YOLOv5 ingredient detector from another university's student project, a map from its 95 labels to the corpus's ingredients, a command-line pipeline and a Gradio app.

## Summary

**Data.** The Food.com Recipes and Interactions dataset on Kaggle (Majumder, Li et al., EMNLP 2019): 231,637 recipes and 1,132,367 reviews. `data/prepare_data.py` averages each recipe's ratings, keeps recipes with at least 5 ratings and 2 to 35 ingredients, and canonicalizes every ingredient phrase. That leaves 53,573 recipes over 7,056 ingredient tokens, with a median of 8 ingredients per recipe (`presentation.ipynb`, cell 6).

**Topic model.** Each recipe is a 0/1 vector over the 500 most common ingredients, and scikit-learn's `LatentDirichletAllocation` fits LDA by batch variational Bayes with Dirichlet priors α = 0.1 and β = 0.01. K is chosen from 4 to 12 by held-out perplexity, which rises at every step, so K = 4. Only the posterior mean of the topic-ingredient matrix, φ̂, is kept.

**Uncertainty.** Fifty bootstrap resamples of 10,000 recipes start from φ̂, take two online update passes, and are aligned to φ̂ with the Hungarian algorithm. The standard deviation of the flavor alignment across resamples is reported as `posterior_uncertainty`. The authors' model notes call the spread of φ "bootstrap stability, not a Bayesian posterior".

**Ranking.** The pantry's flavor profile is P(topic | ingredients) ∝ P(topic) Π φ[topic, ingredient], per resample, and each candidate gets the same profile of its own ingredients. The alignment is exp(−KL(recipe ‖ pantry)), and the score is coverage² × (1 − exp(−matched / 4)) × alignment × rating, with the rating shrunk toward the global mean by 5 pseudo-ratings. Diet and ingredient filters and a diversity re-ranking are optional.

**Detection and the app.** A YOLOv5m checkpoint from Hugging Face finds ingredients in a photo, and a precomputed map turns 93 of its 95 class names into corpus tokens. `run_pipeline.py` and the Gradio `app.py` chain photo, detection and recommendation. The first commit held a fully Bayesian LDA in PyMC, sampled with NUTS on 400 recipes, which the scikit-learn version replaced two days later.

The staff section below explains why the flavor-alignment term rarely changes the ranking and, on a held-out test, makes it worse when it does; why the pantry "posterior" is one-hot for the demo pantries and is not LDA's posterior; what the reported uncertainty measures; and where the README promises more than the app does.

## Team and contributions

Team "Softmax Squad", Spring 2026: Yizhuo Li, Elaine Wang, Qixin Cui, Kevin Fan and Zhetao Zhang.

- **Yizhuo Li** ([@eason1218](https://github.com/eason1218)): the repository and 14 of its 22 commits, made as "Yizhuo": both versions of the recommender with the topic alignment and its unit tests, `data/prepare_data.py`, the detection stage in `yolo/`, `run_pipeline.py`, the first committed `app.py`, the notebooks and figures, `model/README.md`, `yolo/README.md` and most of the write-up.
- **Elaine Wang** ([@welaine123](https://github.com/welaine123)): four commits on May 28, made as "Yiran": the app's redesign to a kitchen theme drawn in Figma, a full-width layout, and two rewrites of the write-up's introduction.
- **Qixin Cui** ([@bennycui2025](https://github.com/bennycui2025)): four edits to the write-up in GitHub's web editor on May 25, including the credit for the Hugging Face detector checkpoint.
- **Kevin Fan** and **Zhetao Zhang** have no commits in the repository.

The repository ran from May 22 to May 28, 2026. 18 of the 22 commits carry a `Co-Authored-By: Claude …` trailer. The commit that added `app.py` says it supersedes an uncommitted "teammate-handoff" demo, and `data/Data.ipynb` was run on Colab, so the committing account is not necessarily a file's only author, and members without commits may have contributed outside the repository. The attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/sp26-what-can-i-cook-tonight/commits/main).

## Getting started

The corpus, the model and the detector weights are not in the repository; [data/README.md](data/README.md) explains how to get the data, and the weights are a manual download. The authors worked on Windows with Python 3.13 and a CUDA GPU (`presentation.ipynb`, cells 2 and 29); the recommender needs no GPU.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "gradio>=6"
git clone https://github.com/ultralytics/yolov5 yolo/yolov5     # detector backend, AGPL-3.0
# download best.pt from huggingface.co/HYUNAHKO/Ingredients_object_detection to yolo/weights/best.pt
```

**The pipeline**, from the repository root:

1. `python data/prepare_data.py` downloads the dataset with `kagglehub` and writes `data/recipes_clean.csv`.
2. `python model/src/run_demo.py` trains the model into `model/models/lda_model.pkl` and rewrites `model/results.json` and `model/model_selection.json`. The authors' run took 43 seconds.
3. `python yolo/build_mapping.py` rebuilds the label map (optional; the committed one is current).
4. `python run_pipeline.py --image samples/fridge.jpg` runs photo to recipes; `python app.py` serves the Gradio page on port 7860.

`presentation.ipynb` runs from the repository root, the two model notebooks from their own folder. For typed ingredients, use `python model/src/recipe_recommender.py --ingredients chicken garlic onion tomato rice`.

**Staff checks.** `staff/check_committed_outputs.py` needs only the standard library and committed files: it recovers the flavor-alignment factor of every committed recommendation from the printed scores, lists `posterior_uncertainty` and the app's "Match" bar, and prints the model-selection table. `staff/rebuild_and_evaluate.py` needs the authors' model dependencies and the public Food.com archive; in about a minute it rebuilds the corpus and the model with the authors' functions, extends the K sweep to K = 2, re-derives the pantry posterior in NumPy beside LDA's own, and ranks 300 held-out recipes four ways.

```sh
python staff/check_committed_outputs.py
curl -L -o ~/Downloads/foodcom.zip \
  https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions
python staff/rebuild_and_evaluate.py --archive ~/Downloads/foodcom.zip
```

**What does not run as committed.** `app.py` raises an error at import if `data/recipes_clean.csv` is missing, and it has only a photo input: the write-up's "No detector? The recommender still works, just type your ingredients instead of uploading a photo" holds for the module's command line, not the app, which without the weights reports the detector unavailable and recommends nothing. `app.py` passes `css` and `theme` to `launch()`, which Gradio accepts from version 6, while `requirements.txt` asks for `gradio>=5`; Gradio 5's `launch()` (checked in the 5.49.1 source) has no such parameters, so the app stops with a `TypeError`. The app's diet menu offers only "vegan"; the vegetarian, must-use and exclude filters listed under the write-up's Features exist only in `recommend()` and on the command line. Each card's "Match" bar is `min(100, int(score × 100))` (`app.py`, line 502), but the score includes a 0-to-5 rating, so all 20 demo recommendations in `model/results.json` would show 100 percent. `assets/demo.png` shows the interface before the May 28 redesign, with a "Posterior uncertainty" line the current app no longer displays.

## Data

`.gitignore` keeps out the corpus, the model, the weights, the YOLOv5 clone and, since the staff edits, the raw Kaggle files. Details are in [data/README.md](data/README.md).

- **Source:** [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions), published with Majumder, Li, Ni and McAuley, "Generating Personalized Recipes from Historical User Preferences", EMNLP-IJCNLP 2019; recipes and reviews crawled from Food.com, 2000 to 2018.
- **Terms:** "Data files © Original Authors", not an open licence. Cite the paper and do not republish the files.
- **The detector:** `HYUNAHKO/Ingredients_object_detection`, trained by the Yonsei University YAICON team behind "Armchair Chef" on their own image dataset; no licence, no model card, no published accuracy. Its 95 classes are Korean fridge staples (wakame, napa cabbage, kimchi, enoki, udon) alongside Western ones.
- **The sample photos:** seven ingredient photos in `samples/`, from Wikimedia Commons according to the authors, with no author or licence recorded per file.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: features, architecture, ranking, methodology by stage, setup. |
| `app.py`, `run_pipeline.py` | The Gradio app and the command-line pipeline, photo to recipes. |
| `presentation.ipynb` | The executed end-to-end showcase: corpus, model selection, topics, the recommender's steps, filters, live detection on two sample photos, caveats. |
| `data/prepare_data.py`, `data/Data.ipynb` | Building `recipes_clean.csv`; the notebook is the first version, run on Colab. |
| `model/src/recipe_recommender.py` | Canonicalization, `train_lda`, `filter_candidates`, `infer_user_posterior`, `score_recipes`, filters, diversity, `recommend`, and a command line. |
| `model/src/topic_alignment.py`, `run_demo.py`, `visualize.py`, `make_flowchart.py`, `model/tests/` | Hungarian alignment of refit topics; the training run behind `results.json`; the figures and diagram; five unit tests. |
| `model/notebooks/` | `model_pipeline.ipynb`, a walkthrough on the cached fit, and `usage_example.ipynb`, one pantry and one call. |
| `model/results.json`, `model/model_selection.json` | The authors' training run: K, perplexities, topics, four demo Top-5 lists. |
| `model/figures/`, `model/README.md`, `model/docs/` | The figures, the model's documentation with equations, a flowchart and a talk script. |
| `yolo/` | Running the detector, building and applying the label map, the committed map, its notebook and README. |
| `samples/`, `assets/demo.png` | Sample photos and the app screenshot. |
| `data/README.md`, `staff/` | Course staff's data note and checks. |

## Method

**Preprocessing.** `normalize_token` lowercases each ingredient phrase, removes non-letters, drops 32 generic modifiers ("fresh", "ground", "boneless" and so on) and singularizes each word. The same function runs at query time, so pantry and recipe ingredients are compared in one form.

**Topic model.** LDA's generative model is θ_m ~ Dirichlet(0.1) for recipe *m*'s topic weights, φ_k ~ Dirichlet(0.01) for topic *k*'s ingredient distribution, and for each ingredient slot z ~ Categorical(θ_m), w ~ Categorical(φ_z), over the 500 most frequent ingredients. scikit-learn fits it by mean-field variational Bayes for 8 batch iterations. K is scored by the perplexity of a held-out tenth of a 12,000-recipe subsample. The final fit uses the full corpus; φ̂ is its normalized `components_`, and P(topic) the mean topic mixture of 15,000 recipes.

**Bootstrap.** Each of 50 replicates draws 10,000 recipes with replacement, initializes online LDA at `components_ = φ̂ × 10,000 + β`, runs two `partial_fit` passes, normalizes, and permutes its topics onto φ̂'s by cosine similarity. Their mean elementwise standard deviation, 0.000441, is `bootstrap_stability`.

**The pantry's flavor profile.** For each replicate *s*,

> P(topic = *k* | pantry) ∝ P(topic = *k*) × Π over the pantry's in-vocabulary ingredients of φ_s[*k*, ingredient],

which treats all of a pantry's ingredients as drawn from one topic. The same function profiles each candidate recipe.

**Score.** A candidate needs coverage of at least 0.7, relaxed to 0.5 if fewer than 20 recipes pass. Its alignment is exp(−KL(recipe profile ‖ pantry profile)), per replicate and averaged, and the standard deviation across replicates is `posterior_uncertainty`. The rating is (avg_rating × n + 5 × global mean) / (n + 5). The score is

> coverage² × (1 − exp(−matched / 4)) × alignment × rating,

where the second factor stops a 3-of-3 recipe from beating a 6-of-7 one. Filters drop recipes by ingredient token; diversity re-ranks by maximal marginal relevance on Jaccard similarity. `recommend()` returns five; the app shows four.

**Detection.** Boxes below confidence 0.25 are dropped and the most confident box per class kept. `build_mapping.py` maps class names to corpus tokens by exact match, then head noun, then RapidFuzz similarity of at least 90 (93 of 95 mapped, `mapping_vocab.ipynb`, cell 5); the yolo README lists its coarse mappings honestly, such as crab sticks to "stick".

**The first version.** Commit 8b73dcc sampled φ and θ with NUTS in PyMC, topic indicators summed out, on 400 recipes, comparing K by held-out log predictive density with WAIC and PSIS-LOO as diagnostics. Its `model_selection.json` has the held-out density best at K = 2; its `results.json` (K = 6, 2 chains of 300 draws) reports effective sample sizes for φ of 1.3 at the minimum and 6.5 at the median: the sampler had not mixed.

### What is Bayesian here

LDA is a Bayesian model, and scikit-learn's fit is variational Bayes: it returns Dirichlet approximations to the posterior over each topic's ingredient distribution and, through `transform`, over each recipe's topic weights. That is variational inference, a course topic, applied to a topic model. The project keeps only the posterior mean of φ and never uses the posterior over topic weights.

The rest is something else. The bootstrap is a frequentist device, and warm-started at φ̂ with two passes it measures how far a short update drifts as much as how φ would vary with new data; the authors say plainly that it is stability, not a posterior. The pantry profile is Bayes' rule for a different model from the one that was fit: a mixture of unigrams, in which all of a pantry's ingredients come from one topic, with φ̂ plugged in and the empirical topic frequency as the prior. Under LDA a pantry mixes topics, and LDA's posterior for it is a Dirichlet over topic weights. The rating is shrinkage with a prior worth 5 pseudo-ratings at the global mean, the posterior mean of a conjugate model of fixed strength; the code calls it Beta-Binomial, though star ratings are not binomial counts.

The score is a hand-built product, not an expected utility, and the reported uncertainty does not enter it. The fully Bayesian model with NUTS, held-out density, WAIC and PSIS-LOO is in the first commit; it did not mix, and the app does not use it.

## Results

**Choosing K.** Held-out perplexity from `model/model_selection.json`. Rows marked *staff* come from the staff rebuild, which calls the authors' `_select_k_perplexity` with the same settings and reproduces their five values.

| K | 2 (staff) | 3 (staff) | **4 (chosen)** | 5 (staff) | 6 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Perplexity | 236.6 | 258.6 | **289.0** | 302.5 | 322.2 | 364.1 | 402.9 | 447.0 |

**The topics.** From `model/results.json`; the rebuild gets the same labels and a `bootstrap_stability` of 0.000440 against 0.000441.

| Topic | Leading ingredients | P(topic) |
|---:|---|---:|
| 0 | cheddar cheese, onion, green onion | 0.1886 |
| 1 | olive oil, salt, parmesan cheese | 0.2455 |
| 2 | salt, onion, water | 0.2280 |
| 3 | egg, salt, flour | 0.3379 |

**A demo pantry.** The Italian-ish pantry (chicken, garlic, onion, olive oil, tomato, basil, parmesan, pasta, salt, pepper) leaves 53 candidates. The authors' score anatomy, `presentation.ipynb`, cell 23:

| Recipe | Matched | Coverage² | Overlap bonus | Alignment | Rating | Score |
|---|---:|---:|---:|---:|---:|---:|
| tomato sandwich from italy | 6 of 7 | 0.735 | 0.777 | 1.0 | 4.803 | 2.7412 |
| garlicky tomato bruschetta | 5 of 6 | 0.694 | 0.713 | 1.0 | 4.634 | 2.2959 |
| pasta with oil and garlic sauce | 5 of 6 | 0.694 | 0.713 | 1.0 | 4.625 | 2.2914 |
| baked parmesan tomatoes | 5 of 6 | 0.694 | 0.713 | 1.0 | 4.524 | 2.2415 |
| fresh mozzarella tomato basil salad | 5 of 6 | 0.694 | 0.713 | 1.0 | 4.515 | 2.2371 |

The rebuild reproduces this list and the Baker's-pantry list in order, with scores within 0.0004.

**Alignment and uncertainty in every committed pantry.** The alignment is the printed score divided by the other three factors (`staff/check_committed_outputs.py`), good to about 0.001.

| Pantry | Source | Ingredients | Alignment, Top-5 | `posterior_uncertainty` |
|---|---|---:|---|---|
| Italian-ish, plain and filtered | `model/results.json` | 10 | 1.000 for all 10 | 0.0 for all 10 |
| Baker's, plain and diversified | `model/results.json` | 9 | 0.991 to 1.000 | 0.0 for 6 of 10, at most 0.0112 |
| fridge.jpg | `presentation.ipynb`, cell 33 | 5 | 0.562 to 0.713 | not printed |
| vegetables.jpg | `presentation.ipynb`, cell 35 | 6 | 0.452 to 0.900 | not printed |
| Usage example | `usage_example.ipynb`, cell 4 | 8 | 0.251 to 0.977 | 0.0691 to 0.4473 |

**The pantry posterior.** From `staff/rebuild_and_evaluate.py`. The authors' column is `infer_user_posterior`, re-derived in NumPy and checked against it; the gap is the difference in log-likelihood between the two best topics. The last column is LDA's own variational posterior for the same pantry from the same fit: the mean weight of the leading topic and a 90 percent interval.

| Pantry | In vocabulary | Authors' posterior | Gap (nats) | LDA's posterior |
|---|---:|---|---:|---|
| Italian-ish | 10 | topic 1: 1.0000 | 46.4 | topic 1: 0.97 [0.87, 1.00] |
| Baker's | 8 of 9 | topic 3: 1.0000 | 45.3 | topic 3: 0.96 [0.84, 1.00] |
| fridge.jpg | 5 | topic 1: 0.72, topic 0: 0.28 | 3.7 | topic 1: 0.94 [0.75, 1.00] |
| Usage example | 8 | topic 2: 0.98, topic 1: 0.0065 | 9.5 | topic 2: 0.73 [0.46, 0.93], topic 1: 0.25 [0.06, 0.51] |

For fridge.jpg the leading topic differs between replicates in 26 percent of them. In the usage-example Top-5, the rebuild swaps "shmaltz" and "roasted cauliflower with garlic", the second with alignment 0.443 against the notebook's implied 0.569.

**Held-out ranking (course staff).** The write-up has no evaluation of the recommender. The staff script refits the model on 90 percent of the recipes. For 300 held-out recipes with 4 to 12 ingredients, the pantry is the recipe without one random ingredient plus three distractors drawn by ingredient frequency; the authors' `filter_candidates` and `score_recipes` rank the candidates (median 127 per query, 6 to 1,045), and the same candidates are re-ranked three other ways.

| Ranking | Hit@1 | Hit@5 | Hit@10 | Mean reciprocal rank |
|---|---:|---:|---:|---:|
| The authors' score | 0.730 | 0.833 | 0.883 | 0.781 |
| The same score without the alignment factor | 0.853 | 0.960 | 0.973 | 0.898 |
| Coverage only | 0.297 | 0.787 | 0.890 | 0.504 |
| Jaccard similarity of ingredient sets | 0.937 | 0.970 | 0.990 | 0.957 |

The alignment factor improves the held-out recipe's rank in 6 percent of queries, worsens it in 21 percent and leaves 73 percent unchanged; with `--seed 1`, 9, 17 and 74 percent, and hit@5 0.843 against 0.937. The pantry posterior puts more than 0.99 on one topic in 54 percent of queries. Because each pantry is built from the recipe it scores, the proxy favors Jaccard by construction; the comparison that isolates the topic model is between the first two rows. The third decimal can move between runs, as the parallel fits are not bit-for-bit deterministic.

**Detection.** On `samples/fridge.jpg`, a grid of five studio photos, the detector finds garlic (0.88), broccoli (0.87), cherry tomatoes (0.86), an egg (0.82) and an onion (0.77) (`presentation.ipynb`, cell 31). With those five the coverage bar relaxes to 0.5, leaving 25 candidates, and each of the top four recipes uses 2 of its 4 ingredients (cell 33). No accuracy of the detector is measured anywhere.

## What to learn from this project

Course staff added this section. The pipeline works end to end and is carefully engineered; the points below are about what its Bayesian part contributes.

- **Check whether the Bayesian part changes the answer.** For all 20 committed demo recommendations the alignment is between 0.991 and 1.000, so the ranking is coverage² × overlap bonus × rating. The authors' closing caveats come close to saying so: "recommendations are robust to [K] because coverage and rating dominate the score" (`presentation.ipynb`, cell 36). Where the alignment does vary, for pantries of five to eight ingredients, the held-out test finds it hurting more often than helping: removing it raises hit@5 from 0.833 to 0.960. A topic model can still earn its place, but only against the same score without it.
- **Use the posterior your model defines.** LDA says a pantry mixes topics; the pantry profile assumes it comes from one. With eight or more common ingredients the log-likelihood gaps between the two best topics are 9 to 46 nats, and the "posterior" is one-hot. LDA's own posterior for the usage-example pantry gives the second topic 0.25, with an interval up to 0.51, where the authors' formula gives 0.0065. A recipe and a pantry certain of different topics then have a KL divergence bounded only by the code's 10⁻¹² floor and an alignment of essentially zero, so a soft similarity becomes a match on the leading topic. Applying `transform` to the pantry would have used the model that was fit.
- **Make figures and captions agree.** The pantry-posterior figure (`model/figures/fig3_user_topic_posterior.png`, titled in `model/src/visualize.py`, line 136) and `presentation.ipynb`, cell 20, say that Italian collapses onto one topic and Baker's spreads across two, as do `model/README.md`, `model_pipeline.ipynb`, cell 15, and `presentation.ipynb`, cell 19. In both figures the Baker's bar stands at 1.0 on one topic, and the rebuild agrees (1.0000, a gap of 45.3 nats). The alignment-spread figure (`fig4_recommendation_uncertainty.png`, and `presentation.ipynb`, cell 25) plots a range of about 10⁻⁷: in the rebuild the smallest per-replicate alignment in the Italian Top-5 is 1 − 1.8 × 10⁻⁷. The axis carries an offset, so the violins read as real spread.
- **Know what your uncertainty measures.** `posterior_uncertainty` is the standard deviation, over 50 warm-started refits, of an alignment that sits near 0 or near 1. In the demo pantries it is exactly 0.0 for 16 of 20 recommendations. In the usage-example pantry it is 0.07 to 0.45 because the refits disagree about which single topic a recipe belongs to: in the rebuild, the alignment of "stewed tomatoes and garbanzo beans" is above 0.9 in 26 percent of refits and below 0.1 in 68 percent. That is the sensitivity of a hard decision to resampling, not a credible interval, and the redesigned app no longer shows it. The write-up's promise of a model that "tells you how confident it is" needs either to go or to be backed by a quantity with a clear meaning, such as the spread of LDA's posterior over the pantry's topic weights.
- **Beware a selection rule that picks the edge of its grid.** Perplexity is lowest at K = 4, the smallest value tried. The authors' own sweep extended downward gives 258.6 at K = 3 and 236.6 at K = 2, still falling, and the first version's held-out density also preferred K = 2. scikit-learn's perplexity comes from a variational bound and commonly rises with the number of topics, so here it cannot choose K, and "K = 4 is the data's choice" (cell 36) overstates it. Choose K by the downstream task, by topic coherence or by eye, and say which.
- **Evaluate the recommender, not just the topic model.** The only quantitative check in the repository is the perplexity table; the demo pantries show plausible lists, but nothing measures whether the ranking finds good recipes. A leave-ingredients-out test like the staff one, with coverage and Jaccard baselines, costs an afternoon.
- **Measure the detector you rely on.** The checkpoint has no published accuracy and the project measures none. The sample photos are studio shots on white, arranged in grids, and the classes are Korean fridge staples while the corpus comes from a US recipe site. A few dozen labelled photos of real fridges would say how often the detected pantry is right.
- **Read your data's conventions.** Food.com stores a review without stars as a 0, and the rating average keeps them: 44,208 ratings in 36.6 percent of the recipes, enough to move the corpus mean from 4.681 to 4.448 and to put two recipes at exactly 0. Dropping zeros before averaging is a one-line fix.
- **Describe borrowed components accurately from the start.** Until May 25 the write-up named no source for the detector and said the checkpoint "was trained on Linux (Colab)". Qixin Cui's edit that day added the Hugging Face credit, but on May 28 the methodology section called it "a custom YOLOv5 detector" and the yolo README said it "was trained externally (Ultralytics `train.py`, on Colab)", which reads as the team's own training run, until a correction later that day. The docstring of `yolo/detect.py` still says the weights "were trained with the ultralytics/yolov5 repo". Name a checkpoint, its authors and its licence status when it first appears.
- **Keep the README's promises in step with the code.** Typed ingredients work only on the module's command line, the vegetarian and must-use filters are not in the app, the app needs Gradio 6 while the requirements ask for 5, the "Match" bar reads 100 percent on every demo card, and the screenshot shows a field the app has lost. Each is small, and they are what a new user meets first.

What the project does well: an everyday question and a pipeline that answers it from a photo to a ranked list, with a polished app; one ingredient canonicalizer shared by data preparation and query, which is what makes coverage meaningful; a label map built against the corpus's own vocabulary, with an honest table of its failures; Hungarian alignment of the bootstrap topics, with unit tests; candid notes in `model/README.md` that the bootstrap is stability, not a posterior; a repository that the staff rebuild reproduces from the public download; and, in its history, a serious attempt at the fully Bayesian model with held-out density, WAIC and PSIS-LOO, whose failure to mix was a good reason to change course.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Spring 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code is unchanged.

- Added `.gitignore` entries for virtual environments and for the raw Food.com files, so that a download saved inside `data/` cannot be committed. No files were removed: the repository has no editor clutter, and `model/models/.gitkeep` holds a one-line note on what is trained there.
- Added `data/README.md`, on the corpus, its terms and how to rebuild it, the detector checkpoint and YOLOv5, and the sample photos.
- Added `staff/check_committed_outputs.py` and `staff/rebuild_and_evaluate.py`, the checks described above.
- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

The authors released their code under the MIT License ([LICENSE](LICENSE)). It does not cover the Food.com data ("© Original Authors"), the detector checkpoint (no licence stated), YOLOv5 (AGPL-3.0) or the sample photos (Wikimedia Commons, not individually attributed); see [data/README.md](data/README.md).
