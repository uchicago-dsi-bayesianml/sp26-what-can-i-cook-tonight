# Data

The recipe corpus that the recommender reads, `data/recipes_clean.csv`, is not in this repository, and neither are the detector weights or the YOLOv5 code. `.gitignore` keeps all three out. What is here are the authors' two scripts for this stage, `prepare_data.py` and `Data.ipynb`, and a few small files that the other stages produced and the authors committed:

- `model/results.json` and `model/model_selection.json`: the output of `model/src/run_demo.py`, the authors' training run on the full corpus (the selected number of topics, the perplexity table, the topics and four demo Top-5 lists). The app reads `results.json`, and `staff/check_committed_outputs.py` reads both.
- `yolo/yolo_vocab_mapping.json`: the map from the detector's 95 class names to ingredient tokens of the corpus, built by `yolo/build_mapping.py`.
- `samples/*.jpg`: seven ingredient photos for trying the detector, described below.

## The recipe corpus

**Source.** [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) on Kaggle, published by Shuyang Li with the paper: Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni and Julian McAuley, "Generating Personalized Recipes from Historical User Preferences", EMNLP-IJCNLP 2019. It holds 231,637 recipes and 1,132,367 user interactions (reviews, most with a star rating) crawled from Food.com, dated 2000 to 2018. The project uses two of its files, `RAW_recipes.csv` and `RAW_interactions.csv`.

**Terms.** Kaggle lists the licence as "Data files © Original Authors": the recipes and reviews belong to the Food.com users who wrote them, and the dataset is not under an open licence. Use it for study and research, cite the paper, and do not republish the files. This repository contains none of them; a few recipe names and ingredient lists appear in the notebooks' printed outputs.

**Getting it.** `data/prepare_data.py` downloads the dataset with `kagglehub`, which may ask for a Kaggle API token. At the time of writing (October 2026) Kaggle also serves the whole dataset as a 280 MB `archive.zip`, without a login, at `https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions`. `staff/rebuild_and_evaluate.py` reads that archive in place. `data/Data.ipynb`, the authors' first version of this stage, run on Colab, reads the same archive from a file named `archive (2).zip` and uses spaCy as well.

**Rebuilding `recipes_clean.csv`.** From the repository root:

```sh
python data/prepare_data.py      # -> data/recipes_clean.csv
```

The script

1. aggregates `RAW_interactions.csv` into a count `n_ratings` and a mean `avg_rating` per recipe;
2. keeps recipes with at least 5 ratings and 2 to 35 ingredients;
3. canonicalizes every ingredient phrase with `normalize_token` from `model/src/recipe_recommender.py` (lowercase, remove punctuation, drop modifiers such as "fresh" and "ground", singularize each word) and drops recipes left with fewer than two ingredients;
4. writes `recipe_id, recipe_name, ingredients (a JSON list), avg_rating, n_ratings`.

The authors' run gives 53,573 recipes and 7,056 distinct ingredient tokens (`presentation.ipynb`, cell 6). The staff rebuild gets the same 53,573 recipes and 7,053 tokens; the fitted topics, the perplexity table and the two demo Top-5 lists still match the committed ones.

**Zero ratings.** Food.com stores a review that came without a star rating as a rating of 0, and step 1 averages those zeros in. Per the staff script, 44,208 of the 811,336 ratings on the kept recipes are 0 (5.4 percent), 19,614 recipes (36.6 percent) have at least one 0 in their average, the mean `avg_rating` is 4.448 with the zeros and 4.681 without, and 2,614 recipes would fall below the 5-rating threshold if zeros were not counted. The minimum `avg_rating` of 0.00 in `model/notebooks/model_pipeline.ipynb`, cell 4, is one of the two recipes whose every review is a zero.

**People.** The interactions identify users by numeric IDs only. Nothing in this repository identifies a person.

## The ingredient detector

**The checkpoint.** `yolo/weights/best.pt` is the file `best.pt` from the Hugging Face repository [`HYUNAHKO/Ingredients_object_detection`](https://huggingface.co/HYUNAHKO/Ingredients_object_detection), uploaded in December 2024. It is a YOLOv5m model fine-tuned on 95 ingredient classes for "Armchair Chef" (방구석 흑백요리사), a student project at Yonsei University's YAICON 5th ([HYUNAHKO/Recipe-Generation](https://github.com/HYUNAHKO/Recipe-Generation)), on the image dataset [`HYUNAHKO/ingredients_dataset`](https://huggingface.co/datasets/HYUNAHKO/ingredients_dataset). The model repository, the dataset and the project repository have no licence and no model card, and no accuracy has been published for the checkpoint. This team did not train it. Download it by hand to `yolo/weights/best.pt`; never commit it.

**YOLOv5.** The checkpoint is loaded through [ultralytics/yolov5](https://github.com/ultralytics/yolov5), which the setup instructions clone into `yolo/yolov5/`. Ultralytics licenses YOLOv5 under AGPL-3.0 and sells a separate enterprise licence. The YOLOv5 code is not part of this repository, and the repository's MIT licence does not cover it or the checkpoint.

## The sample photos

`samples/` holds single photos of a tomato, broccoli, garlic, an egg and an onion, plus `fridge.jpg`, a grid of five of them, and `vegetables.jpg`, a grid of six ingredients. The authors' `samples/README.md` says they were downloaded from Wikimedia Commons ("CC / public-domain") and resized. No file has its author, licence or source page recorded, which CC BY and CC BY-SA images require. Course staff kept the photos as the authors left them; treat them as unattributed until their Commons pages are found.
