# CLAUDE.md

Guidance for helping someone **use** this DerivaML CIFAR-10 example —
running it, understanding how it works, reading its results, and
adapting it to their own model and data.

**Usage and commands live in the docs**, not here. Point the user to:
- [README.md](README.md) — setup, authentication, loading data,
  configuring catalogs, running the model (the Quick Start §1–8 has
  every command).
- [CIFAR10.md](CIFAR10.md) — the end-to-end CIFAR-10 walkthrough
  (load → train → sweep → ROC analysis) and the loader's design.
- [Experiments.md](Experiments.md) — experiment and multirun presets.

This file is the orientation a helper needs: what the project is, where
things live, and the gotchas a user actually hits.

## What this is

A runnable reference: a **CIFAR-10 CNN trained and tracked on a Deriva
catalog** via DerivaML, with 7 model variants and several experiment
presets. A user typically clones it to (a) learn how DerivaML
structures a reproducible ML workflow, then (b) swap in their own model
and data. The platform underneath:

- **deriva-ml** — the core Python library for reproducible ML on Deriva
  catalogs (datasets, workflows, executions, features, assets, lineage).
- **Hydra-zen** — Python-first configuration (no YAML).
- **uv** — dependency management and script execution.

## The user's path (what they're usually doing)

1. **Set up** — `uv sync` (+ `--group=torch` for the model,
   `--group=jupyter` for notebooks), authenticate to a Deriva host.
   (README §3–5.)
2. **Load data** — `load-cifar10` creates/populates a catalog with
   CIFAR-10 images and dataset families. (README §6, CIFAR10.md.)
3. **Wire configs to their catalog** — the dataset RIDs in
   `src/configs/datasets.py` ship from a *previous* demo catalog and
   must be replaced with the RIDs `load-cifar10` printed. Until then,
   runs that reference those datasets fail. (README §7 has the
   procedure.)
4. **Run** — `deriva-ml-run` trains a model / runs an experiment preset
   / a multirun sweep, writing outputs and provenance to the catalog.
   Always offer `dry_run=true` first. (README §8.)
5. **Analyze** — ROC notebook + evaluation. (CIFAR10.md.)

When helping, find which step the user is on before answering.

## Where things live

- `src/configs/` — Hydra-zen config (Python). `datasets.py` (dataset
  RIDs), `cifar10_cnn.py` (model + hyperparameters), `experiments.py`
  (model+dataset presets), `multiruns.py` (sweeps), `deriva.py`
  (connections), `assets.py` (output asset RIDs), `dev/` (per-env
  overrides).
- `src/models/cifar10_cnn.py` — the CNN, training loop, prediction
  recording. `model_protocol.py` — the interface a model function
  implements (the seam a user replaces).
- `src/scripts/` — the data-loading package (`load-cifar10` lives here).
- `notebooks/` — analysis (ROC). `tests/` — config smoke tests.
- `CIFAR10.md` §"Reusing for your own data" — the guide for swapping in
  the user's own model/data.

## Gotchas a user will hit

- **Configs ship with stale RIDs.** `src/configs/datasets.py` points at
  a previous demo catalog. After `load-cifar10`, the user must update
  those RIDs (and `version=`) to the ones the loader reported, or runs
  fail with "dataset not found." README §7 is the procedure. This is the
  single most common first-run snag.
- **Use labeled datasets for evaluation.** `cifar10_small_labeled_split`
  / `cifar10_labeled_split` carry ground truth on both train and test
  partitions — the right choice for ROC, accuracy, any evaluation. The
  plain `*_split` configs are training-only.
- **`load-cifar10 --num-images` has a floor for the small-split family.**
  Below ~1002 images the "small" Toronto split would be byte-identical
  to the full split and the datasets phase errors
  (`SmallVariantDegenerateError`). Use `--num-images >= 1002`, or the
  labeled-split family (distinct at any size).
- **macOS DataLoaders: `num_workers=0`.** `fork()` + MPS/GPU threads
  deadlock. The example keeps DataLoaders single-worker on macOS; keep
  it that way.
- **Provenance needs a clean tree.** DerivaML records the running
  script's git commit hash; a dirty working tree triggers warnings (it
  refuses to record a hash that wouldn't reproduce the code). For quick
  local iteration: `DERIVA_ML_ALLOW_DIRTY=true uv run <command>`. Never
  in a real run — the warning is what protects reproducibility.
- **Run tests with `uv run python -m pytest`, not `uv run pytest`.** The
  venv's `pytest` shim can have a stale shebang; `uv sync --reinstall`
  fixes it.
- **`dry_run=true` before any catalog-writing run.** It validates
  config + connection without writing.

## Notebook runner specifics

- **`--config` on `deriva-ml-run-notebook` does NOT override the
  `run_notebook()` config name** — use positional Hydra overrides
  (e.g. `assets=my_assets_prod`).
- **`--host` / `--catalog` are papermill parameters, not Hydra
  overrides** — they set the notebook's connection target but don't
  change which `deriva_ml=` config resolves. To target a non-default
  catalog, pass `deriva_ml=<config_name>` AND register the connection in
  `src/configs/dev/deriva_<env>.py`.

## When helping with code

If the user asks you to change or extend the example (a new model
variant, a new experiment, a config tweak):

- **Use `uv` for everything** — `uv run <cmd>`; never bare `pytest` /
  `ruff` / `python`.
- **Google-style docstrings** (`Args:` / `Returns:` / `Raises:` /
  runnable `Example:`) on new functions, methods, classes.
- **Match the existing patterns** — model functions follow
  `model_protocol.py`; configs are Hydra-zen dataclasses; new modules
  get a config smoke test alongside the existing `tests/`.
- **No dead code, no over-engineering** — add only what the task needs.

## Related docs

- [README.md](README.md) — setup and the full command list.
- [CIFAR10.md](CIFAR10.md) — walkthrough + reuse-for-your-own-data guide.
- [Experiments.md](Experiments.md) — experiment/multirun presets.
- [docs/design/](docs/design/) — the design docs for the shipped
  example (experiments, datasets, features, model), useful for
  understanding *why* the example is structured as it is.
