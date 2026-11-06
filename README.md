# Predicting Airline Passenger Satisfaction (Kaggle Playground S6E10)

Binary classification of airline passenger satisfaction, scored by ROC AUC.
Kaggle Playground Series, Season 6 Episode 10 (October 2026, about 3,700 teams).

| Result | Value |
|---|---|
| Public leaderboard | **0.96225**, rank **134 / 3,735** (top 4%) |
| Best private score among my submissions | **0.96237** (a stack chosen by cross-validation) |
| Best single model, 5-fold OOF AUC | 0.96109 (LightGBM / XGBoost with in-fold target encoding) |

## What worked

**1. `Flight Distance` acts as a route ID.** The column takes only about 3,500 distinct values, so each value
almost identifies one route. I added label-free per-route statistics (mean service ratings, age, class shares),
computed over train, test and the original public dataset, plus the satisfaction rate per route in the original
dataset. This raised a single LightGBM from 0.95875 to 0.96044 OOF AUC, the largest gain in the project.
The idea came from the public repo
[ChrisLegge/kaggleAirlineSatisfaction](https://github.com/ChrisLegge/kaggleAirlineSatisfaction).

**2. Conditional target encoding inside each fold.** I used scikit-learn `TargetEncoder` (inner `cv=5`) on
interactions such as rating x travel type x class and route x class, plus frequency encoding. This gave the best
single models (0.96109 OOF).

**3. Diverse model families instead of more boosted trees.** The OOF predictions of LightGBM, XGBoost and CatBoost
correlate at 0.9993, so stacking them adds almost nothing. RealMLP and TabM (via `pytabkit`), models trained
without route features, and a "teacher" feature (a model trained only on the original dataset) all received real
weight in the stack.

**4. Logistic stacking on logits, checked with nested CV.** The final stack combined 278 members (my models plus
public OOF libraries shared on Kaggle, after removing near-duplicates). It reached a nested-CV AUC of 0.96221.
On the public leaderboard, scores ran about 0.0003 below CV, and CV gains carried over about one to one.

## Lesson: trust cross-validation over the public leaderboard

The public leaderboard used only 20% of the test set. Late in the competition, many public files climbed the
leaderboard by blending other submissions with weights tuned on the leaderboard itself. My leaderboard-chasing
submission scored 0.96225 public but **0.96211 private**. My stacks chosen by cross-validation scored only 0.96185
public but **0.96235 to 0.96237 private**.

| Submission | Chosen by | Public | Private |
|---|---|---|---|
| v11: CV stack, 50/50 rank blend with a public stack | nested CV | 0.96185 | **0.96237** |
| v12: 70% v11 + 30% public pillar | nested CV | 0.96185 | **0.96237** |
| v18: public blend file | public leaderboard | 0.96217 | 0.96216 |
| v21: "endgame" public blend | public leaderboard | **0.96225** | 0.96211 |

Things that did not help: an LGBM parameter grid, a LightGBM level-2 stacker, extra route-level means from the
original data, an auxiliary model predicting the `Inflight service` column, route-level residual correction, and
fine-tuning tabular foundation models (too expensive for the gain).

## Project layout

```
src/
  common.py        paths, column lists, 5-fold StratifiedKFold (seed 42)
  feats.py         feature builders: route stats, original-data target encoding, in-fold conditional TE, teacher
  baseline.py      plain LightGBM baseline
  train.py         train one model config with 5-fold CV -> artifacts/zoo/<name>_{oof,test}.npy
  nn.py            RealMLP / TabM via pytabkit
  blend.py         rank mean vs hill climbing vs logistic stacking, compared with nested CV
  stack_lgb.py     LightGBM level-2 stacker (tested, not used)
  exp_orig.py      experiments with the original dataset
  exp_route.py     experiments with route features
  fetch_public.py  download the inputs of a public Kaggle notebook through the Kaggle API
  mega_build.py    cache public OOF members as logits
  mega_extra*.py   cache OOF libraries published later in the competition
  mega_exp.py      large logistic stack: public members + own zoo, nested CV
  export_colab.py, ingest_colab.py   move feature matrices to Colab and results back
colab/             GPU notebooks (TabM, RealMLP, GBDT, conditional-TE models) and colab_gpu.py
```

## Reproduce

```bash
pip install -r requirements.txt
# put train.csv, test.csv, sample_submission.csv in data/
# put the original Airline Passenger Satisfaction dataset at data/original/original.csv
cd src
python baseline.py
python train.py lgb_ctx --threads 16
python train.py xgb_ctx --threads 16
python blend.py
```

Competition data is not included, following the Kaggle rules. Download it from the
[competition page](https://www.kaggle.com/competitions/playground-series-s6e10).
Everything ran on an 8-core laptop CPU, with a Colab T4 GPU for the neural models.
