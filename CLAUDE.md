# CLAUDE.md — recommender_101

Guidance for Claude Code (and humans) working in this repo. Read this before writing code,
notebooks, or prose.

---

## 1. What this project is

An educational **"101"** on how recommendation actually works, built **from scratch on CPU**.
Six notebooks + one Streamlit app + a small `src/` library. The build-up is linear and each
chapter earns the next:

```
popularity baseline → collaborative filtering → matrix factorization → arena → cold start → framework
```

**The thesis (say it in code, not just prose):**
- Collaborative filtering is *"find users similar to you and copy their taste"* — similarity + copying, nothing more.
- Matrix factorization is *compression*: factor the giant sparse ratings table into a few hidden
  taste dimensions per user and per item; the dot product predicts a rating. Same idea as embeddings.
- The **"recommend the most popular items to everyone"** baseline is shockingly strong — competitive
  with or ahead of the fancy methods on a large fraction of real ranking metrics.

**The central, non-negotiable lesson:** **report the popularity baseline FIRST, always.** A recommender
that has not been compared to popularity has not been evaluated. Every notebook and every arena result
is measured against that line.

This repo is honest by design, and the demonstrated result on MovieLens-small is **not** the rosy one
the PRD predicted. We kept the truth — a deliberate, user-approved choice. **Do not "fix" the notebooks
to make the fancy method win.** What the data actually shows (see `outputs/data/04_scoreboard.json`):
- **Neighbourhood CF (item-based) is the top-N accuracy winner** — *not* matrix factorization.
- **Matrix factorization trained on explicit ratings LOSES to popularity at top-N.** It minimises RMSE
  (and beats an item-mean RMSE baseline at that), but RMSE ≠ ranking — the Cremonesi 2010 / Rendle 2020
  gap. Making MF win top-N needs a *ranking* objective (implicit / BPR), which is future work.
- **Popularity is competitive on accuracy yet worst at discovery** (coverage, novelty, diversity) — the
  two-sided story the arena exists to show.
- Personalized methods **collapse on cold-start**; popularity is the fallback until history accrues.

Do not oversell MF. The offline/online gap is a caveat we state plainly, not a dismissal of offline eval.

Part of the Dima806 "101" family — it combines ideas from `embeddings_101` (MF produces embeddings),
`gradient_descent_101` (the SGD that trains MF), `distance_arena` (similarity is a distance-metric
choice), `metrics_101` (rank, don't score accuracy), `train_test_leakage_demo` (the no-leak split), and
`bandits_101` (cold-start is explore–exploit). README links them all.

---

## 2. Hard constraints (design everything around these)

| Constraint | Rule |
|---|---|
| Hardware | **2 CPU cores, 8 GB RAM, no GPU.** Everything must run here comfortably. |
| Python | **3.11+** |
| Packages | **`uv` only. No `pip`, no `conda`.** Add deps via `pyproject.toml` + `uv sync`. |
| Dataset | **MovieLens-small (~100k ratings)**, cached to `data/` and a **frozen copy committed** to survive URL rot. |
| Notebook runtime | **< 6 minutes each** on 2 CPUs. The heaviest cell (MF SGD + full arena) is ~4 min and **must be cached**. |
| Dependencies | Stay within the approved list (§6). No deep-learning frameworks, no `surprise`/`implicit`/`lightfm` — **we build the recommenders ourselves.** |
| Reproducibility | Seed everything (splits, MF init/SGD). Determinism is asserted in tests. |

If a change risks breaking any of these (e.g. adds a GPU dep, blows the runtime budget, pulls in a
recommender library), **stop and flag it** rather than proceeding.

---

## 3. Repository structure

```
src/
  config.py                      # pydantic-settings; loads config/settings.yaml (dims, neighbors, top-k, seeds)
  recommenders/
    popularity.py                # most-popular baseline — ignores the user entirely (the yardstick)
    user_knn.py                  # user-based CF from scratch — similarity + copy neighbors' likes
    item_knn.py                  # item-based CF from scratch — the dual; more stable at scale
    matrix_factorization.py      # MF via SGD from scratch — user + item embedding matrices
    content.py                   # content-based (genre features) — works for cold-start items
  data/
    ratings.py                   # MovieLens loader → sparse user-item matrix, cached; `python -m src.data.ratings`
  evaluation/
    splits.py                    # leave-last-out / temporal holdout PER USER (no leakage)
    ranking.py                   # precision@k, recall@k, NDCG@k, MAP, catalog coverage
    beyond_accuracy.py           # novelty, diversity, popularity-bias of recommendations
    comparison.py                # the arena harness — runs every method, builds the scoreboard
  visualisation.py               # shared plotting helpers (matplotlib)
  artifacts.py                   # save_figure() -> outputs/figures/*.png, save_json() -> outputs/data/*.json
notebooks/                       # 01..06, built + executed with outputs embedded, see §7
app/streamlit_app.py             # 4 views, see §8
tests/                           # see §9
config/settings.yaml             # all tunable knobs + seeds live here, not hardcoded
data/                            # MovieLens cached here (movielens_*.parquet) — frozen copy, committed
outputs/figures/                 # every notebook figure as .png (16 files)
outputs/data/                    # every notebook result as .json (17 files)
```

Import root is the repo root; the package is imported as `src.<...>` (e.g. `from src.recommenders.popularity import ...`).
The project is intentionally **not** installed as a wheel (`[tool.uv] package = false`) — we run from the repo root.

---

## 4. The five recommenders (all from scratch)

1. **Popularity** — rank items by how many users liked them; recommend the top-k to everyone.
   The mandatory baseline. Keep it a few lines; its simplicity is the point.
2. **User-based kNN CF** — find the most similar users (by rating overlap), recommend what they liked
   that the target hasn't seen. **Expose the similarity computation** so it reads as literal code.
3. **Item-based kNN CF** — the dual: recommend items similar to ones the user liked. More stable /
   scalable because item similarities drift slowly.
4. **Matrix factorization (SGD)** — learn user and item embedding matrices; predicted rating = dot product.
   Small latent dim, L2 regularization, seeded init. Reuse the gradient-descent/embeddings framing.
   Predicts ratings well (beats an item-mean RMSE baseline) and recovers synthetic planted structure,
   **but ranks poorly at top-N — on real data it loses to popularity** (the RMSE-vs-ranking gap). The
   clearest link to `embeddings_101`, not the arena's accuracy winner.
5. **Content-based** — recommend items whose features (genres) resemble what the user liked. Needs no
   other users, so it survives cold-start. The complement that fixes CF's blind spot.

Keep the implementations readable over clever. These are teaching artifacts — the code *is* the explanation.

---

## 5. Evaluation (this is where recommenders fool you)

- **Split:** leave-last-out / temporal holdout **per user** — hold out each user's most recent interactions
  as the test set, train on the rest. **No holdout may leak into training** (asserted in tests).
- **Rank, don't score accuracy:** judge top-k lists with **precision@k, recall@k, NDCG@k, MAP, coverage**.
  Accuracy/RMSE is the wrong tool here (the `metrics_101` lesson).
- **Beyond accuracy:** also report **novelty, diversity, and popularity bias** — popularity has great
  accuracy metrics but terrible discovery, a cost the accuracy numbers hide. Always show both sides.
- **The arena** (`comparison.py`): one harness runs all methods on the same split and produces the
  scoreboard with the popularity line drawn through it.

---

## 6. Stack & dependencies

**Runtime:** `scikit-learn`, `scipy` (sparse matrices), `numpy`, `pandas`, `matplotlib`, `plotly`,
`streamlit`, `pydantic`, `pydantic-settings`, `pyyaml`.

**Dev:** `pytest`, `pytest-cov`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.

Do not add a recommender library or a DL framework — building the methods ourselves is the whole project.

---

## 7. Notebooks (01–06)

All six are **built and executed** (outputs embedded; total ~2.5 min, each well < 6 min). Each starts
with a bootstrap cell (find repo root, fix `sys.path`, `chdir`) so it runs from anywhere, imports real
logic from `src/`, and uses `src/artifacts.py` to save **every figure to `outputs/figures/*.png`** and
**every textual/numerical result to `outputs/data/*.json`**. To rebuild: edit the scratch builder, then
`jupyter nbconvert --to notebook --execute --inplace notebooks/NN_*.ipynb`.

1. **`01_the_popularity_baseline.ipynb`** — load ratings, build sparse matrix, set up honest per-user
   holdout + ranking metrics, implement popularity, record its scores. Frame the rule: *the bar to beat.*
2. **`02_collaborative_filtering.ipynb`** — user- and item-based CF from scratch, similarity drawn out as
   code (item neighbours of *Forrest Gump* are sensible). Honest result: CF **does** beat popularity on
   accuracy, but modestly and `k`-dependently; its edge **collapses for sparse users** (shown by activity
   quartile) — the seed of cold-start.
3. **`03_matrix_factorization.ipynb`** — MF via SGD; recovers synthetic planted structure (r≈0.99); real
   item embeddings lean toward clustering by genre with no labels. Then the money result: **MF wins RMSE
   but loses ranking to popularity** (`03_rmse_vs_ranking.png`) — the project's title, demonstrated.
4. **`04_arena.ipynb`** — all five vs popularity across ranking + beyond-accuracy. Honest scoreboard:
   **item-CF leads accuracy, MF trails popularity, popularity is worst at discovery** (coverage/novelty/
   diversity). Saves `04_scoreboard.json` — the canonical numbers the other notebooks/README reuse.
5. **`05_cold_start.ipynb`** — at zero history only popularity works; CF overtakes after **one** rating
   (`05_cold_start_curve.png`). Content stays weak (coarse genres help cold *items*, not cold *users*).
   Framed as **explore–exploit** (→ `bandits_101`).
6. **`06_decision_framework.ipynb`** — reuses the saved JSON into a decision table + accuracy-vs-discovery
   map + the honest offline/online closer. Ships the recommenders + harness as reusable code.

**Workflow tip:** run `/compact` at notebook boundaries to keep context lean.

---

## 8. Streamlit app (`app/streamlit_app.py`)

Four views: (1) **Recommend-to-me** — a user's top-k from popularity/CF/MF side by side with overlap
highlighted; (2) **Taste-vector explorer** — inspect learned latent dims, items clustering by genre;
(3) **Baseline showdown** — pick a metric, see every method against the popularity line;
(4) **Cold-start simulator** — new user → CF fails, popularity works → add ratings → personalized takes over.
Reuse cached artifacts; don't retrain on every interaction.

---

## 9. Testing & success criteria

`make test` must pass with **zero errors**. The invariants that guard the project's claims:

- Every recommender returns a valid top-k list (right length, no unseen-item leakage, no already-rated items).
- **MF recovers the planted latent structure** on the synthetic ground-truth matrix (asserted).
- **No user's holdout leaks into training** in the split (asserted).
- **NDCG@k from scratch matches a reference implementation to 1e-9.**
- **MF beats popularity on the *synthetic* grouped data** (`test_mf_beats_popularity_on_ranking`) — where
  latent structure is strong and planted. This is the one place MF wins ranking; on real MovieLens it does
  not (see §1). Keep the test synthetic; don't assert an MF top-N win on real data.

Demonstrated outcomes the notebooks reproduce (19 tests pass; `make ci` green): item-CF is the top-N
accuracy winner; **explicit-rating MF loses to popularity at top-N** while winning on RMSE; neighbourhood
CF beats popularity on accuracy but modestly; popularity has the worst novelty/diversity/coverage; a cold
user breaks every personalized method until history accrues.

---

## 10. Decision framework (the takeaway to protect)

- **Always report the popularity baseline first.**
- Sparse data / few ratings per user → item-based CF or content-based.
- Plenty of interactions + want best accuracy → **item-based CF** (MF only if trained for ranking).
- New users → popularity, then CF/MF as history builds; new items → content (explore–exploit).
- Optimize beyond accuracy (novelty, diversity) — a recommender that only surfaces popular items offers
  no discovery, which defeats the purpose.
- Offline wins are **necessary but not sufficient** — offline metrics reward recommending what the user
  would have found anyway.

---

## 11. Conventions

- **Config, not constants:** latent dims, neighbor counts, top-k, and all seeds live in
  `config/settings.yaml`, loaded via `src/config.py` (pydantic-settings). Don't hardcode knobs in `src/`.
- **Style:** `ruff format` + `ruff check` (line length 99; rule set in `pyproject.toml`). Type-check `src/`
  with `ty`. Run `make lint` before committing.
- **Sparse first:** use `scipy.sparse` for the user-item matrix; never densify ~100k ratings unnecessarily.
- **Readable over clever:** these are teaching implementations. Comment the *why* (the idea), not the obvious.
- **Determinism:** thread the configured seed through every split, init, and SGD run.
- **Artifacts:** notebooks save figures via `save_figure()` → `outputs/figures/*.png` and results via
  `save_json()` → `outputs/data/*.json` (both in `src/artifacts.py`, numpy/pandas-aware). Prefix each
  file with the notebook number (`04_scoreboard.json`). Shared plotting helpers in `src/visualisation.py`.

---

## 12. Commands

```bash
make setup      # first-time: install uv, uv sync --all-extras, register the Jupyter kernel
make data       # download + cache MovieLens-small (python -m src.data.ratings)
make lint       # ruff format + ruff check --fix + ty check
make test       # pytest   (make test-cov for coverage)
make notebooks  # execute all notebooks end-to-end (fails on any error)
make run        # launch the Streamlit app
make lab        # JupyterLab
make ci         # sync + lint + test   (what CI runs)
make dev        # fast loop: lint + test
```

---

## 13. Token / workflow rules

- **Caveman mode: ON.** Prefer terse, code-first responses when generating implementation and notebooks;
  skip boilerplate prose. (This CLAUDE.md is the detailed reference; day-to-day output stays lean.)
- `/compact` at notebook boundaries.
- When in doubt about a method's behavior, **write the test or the baseline comparison** rather than asserting it.
