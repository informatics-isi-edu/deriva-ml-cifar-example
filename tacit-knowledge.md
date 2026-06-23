# Tacit Knowledge

This file records **tacit knowledge** — the *why*, the *intent*, and the
*background* behind decisions made about this project's models and data.

The **catalog** is the source of record for everything else: data contents,
RIDs, dataset versions, workflow URLs and checksums, executions, lineage.
Don't replicate catalog-stored facts here. Don't ask this file what's in
the catalog — query the catalog directly (resources first, tools next).
When this file *needs* to reference a catalog entity, link to it
(`deriva://catalog/{host}/{cat}/ml/...`) instead of inlining its contents.

Each entry captures a decision: what was chosen, what alternatives were
considered, what was rejected and why, and any background context a future
reader would need to evaluate whether the decision still holds.

---

<a id="tk-001"></a>
### tk-001 — Convention — the offline smoke test stops *before* `dry_run=true`; that command needs a provisioned catalog
**When:** 2026-06-23T00:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

Lesson from a mislabeled smoke test: `deriva-ml-run dry_run=true` does **not**
belong in an *offline* (no-catalog) smoke test. It was included as if it were
catalog-free, failed against the fresh checkout, and the failure was wrongly
reported as a template defect. It is not — it is expected behavior when no
DerivaML catalog exists yet. The offline smoke test that genuinely never
touches a catalog is: `uv sync`, `uv run python -m pytest tests/`,
`uv run ruff check src tests`, `uv run ruff format --check src tests`,
`uv run deriva-ml-run --list-configs`, and `uv run deriva-ml-run --cfg job`
(the last resolves and prints the composed Hydra config without constructing
an `Execution`). `dry_run=true` is a *catalog* smoke step — run it only once a
catalog is provisioned.

The underlying mechanism, for reference:

During that smoke test, the dry run aborted with
`DerivaMLTableNotFound: Workflow_Type` against the shipped placeholder
connection `default_deriva` (`hostname=localhost`, `catalog_id=0`).

What was actually missing: **the whole DerivaML schema, because catalog 0 is
an empty placeholder that was never provisioned with DerivaML.** The error is
*not* about a missing workflow-type vocabulary *term*. `Execution.__init__`
validates the workflow type via `lookup_term(MLVocab.workflow_type, wt)`,
which calls `name_to_table("Workflow_Type")` *first* — and that searches the
catalog's schemas for a table named `Workflow_Type`. Catalog 0 has no
`deriva-ml` schema at all, so the table lookup fails before any term value is
ever compared. The validation fired correctly and loudly; it was reporting
"this catalog hasn't been set up for DerivaML," which is true. Against a real,
provisioned DerivaML catalog this same dry run succeeds.

Why this matters for the offline smoke test: `dry_run=true` suppresses the
catalog *writes* but still *reads* the catalog to validate the workflow type,
so it requires a reachable, DerivaML-provisioned catalog — it is **not** a
no-catalog command. The `lookup_term` read happens before the `dry_run` guard
in `deriva_ml/execution/execution.py`; that ordering is upstream `deriva-ml`
behavior, not a template concern. (`catalog_id=0` being a placeholder is
documented in the `src/configs/deriva.py` docstring.)

Implications for collaborators: README §8 presents `deriva-ml-run
dry_run=true` as the canonical "no catalog writes" smoke step, which misleads
on a fresh checkout — there's no provisioned catalog to read yet, so it
aborts. To exercise it, first `load-cifar10` into a real catalog (or point at
an existing one) and run `deriva-ml-run dry_run=true --host <h> --catalog
<id>`. For a genuinely offline structural smoke test, stop at `uv sync` +
`pytest` + `ruff` + `deriva-ml-run --list-configs` + `deriva-ml-run --cfg job`
(the last resolves and prints the composed Hydra config without constructing
an `Execution`, so it never touches a catalog).

**Weighed alternatives:** *(none captured — this is an observed behavior of the
shipped code, not a choice made in this session.)*

<a id="tk-002"></a>
### tk-002 — Convention — `load-cifar10 --num-images` must clear the small-variant floor (`>= 1002`, practically `2000`)
**When:** 2026-06-23T14:40:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (the offline smoke test couldn't reach a catalog; this is the first real load)

A `load-cifar10 --num-images 1000` run failed in the *datasets* phase (images
uploaded fine first) with `SmallVariantDegenerateError`:

> At this catalog size (train_pool=500, test_pool=500) the 'small' Toronto
> split family would be byte-identical to the full Toronto split.
> SMALL_TRAIN_SIZE=500 and SMALL_TEST_SIZE=500 require strictly larger source
> pools to yield a distinct sample. Re-run with --num-images >= 1002 ... or
> skip the small Toronto split and use the labeled-split family instead.

Why this is correct behavior, not a bug: `--num-images N` splits ~50/50 into
train/test pools (N=1000 → 500/500). The `Small_Training` / `Small_Testing`
variants are a stratified `subsample()` of 500 each. Sampling 500 from a pool
of exactly 500 yields a byte-identical copy — a degenerate "subset" that would
silently mislead anyone who later compared `Small_Training` against `Training`
expecting them to differ. `_require_small_variant_distinct` (the same guard
exercised by `test_require_small_variant_distinct_rejects_*` in the suite)
refuses rather than create the degenerate dataset. The choice of `N=1000` in
this session landed *exactly* on the degenerate boundary — a poor parameter
pick, not a template fault.

Implications for collaborators: when creating a CIFAR-10 catalog that needs
the small Toronto split family (`cifar10_small_training` / `_small_testing`,
which the example model uses by default), pass `--num-images >= 1002` so each
partition strictly exceeds the 500-sample size. In practice pick a round
number with headroom (`--num-images 2000` → 1000/1000 pools). The error is
loud and prescriptive, so the failure mode is self-correcting — but the
README's "10000 for first-time setup" and the absence of any stated *minimum*
make 1000 an easy, wrong guess.

**Weighed alternatives:** the error itself names a second path — skip the small
Toronto family and use the labeled-split family (`split_dataset()` partitions
training images directly and stays distinct at any catalog size). Chose to
re-load at a larger `--num-images` instead, because the example model's default
dataset (`cifar10_small_labeled_split`) wants the small family present.

<a id="tk-003"></a>
### tk-003 — Convention — `load-cifar10` builds a fixed set of named datasets, and `--dry-run` provisions the schema
**When:** 2026-06-23T14:42:30-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-002](#tk-002) (the small-variant floor that constrains how many images the load needs)

Two durable facts learned while standing up a CIFAR-10 catalog to exercise the
template end-to-end after the offline smoke test (the catalog itself was a
throwaway, so its RIDs/host/catalog-id are deliberately *not* recorded here —
fetch live catalog state from `ml.find_datasets()` instead).

**`--dry-run` is not a no-op — it provisions the schema.** The loader's
`--dry-run` creates the catalog and installs the full schema (`cifar10_*`
domain model + `deriva-ml`), and only skips the *data* writes (image upload,
features, dataset hierarchy). So a "dry run" leaves a real, schema-provisioned
but empty catalog behind. Useful to know both for verifying auth/connectivity
cheaply and for not accidentally accumulating orphaned catalogs.

**The config-name → dataset-role mapping is a project decision** (the catalog
stores each dataset's RID, role, and origin tags, but not which
`src/configs/datasets.py` name is *intended* to pin it — that association lives
in the template). `load-cifar10` always builds the same named family; the
durable mapping is:

| `datasets.py` config name | Dataset role / origin |
|---|---|
| `cifar10_complete` | Complete (Labeled) — the root superset |
| `cifar10_split` / `_training` / `_testing` | canonical Toronto Split + its Training/Testing children (`Split_Partition`) |
| `cifar10_small_training` / `_small_testing` | stratified `Subsample` of Training/Testing (sibling pair, no parent Split) |
| `cifar10_labeled_split` / `_training` / `_testing` | training-derived labeled holdout Split + children |
| `cifar10_small_labeled_split` / `_training` / `_testing` | small labeled holdout Split + children (the example model's default) |

(The full structural detail lives in the forward design doc
`docs/design/dataset/cifar10-input-datasets.md`.)

Implications for collaborators: to run the example model against a freshly
loaded catalog, either pass `--host <h> --catalog <id>` on the CLI, or fill the
RIDs into `src/configs/datasets.py` (per README §7) — the shipped defaults
carry stale RIDs from a prior demo catalog and won't resolve. Resolve the
actual RIDs + versions for your catalog via `ml.find_datasets()` before pinning
them. If you load with `DERIVA_ML_ALLOW_DIRTY=true` (uncommitted tree), the
recorded git provenance hash won't reflect the working tree — fine for a
throwaway catalog, but don't cite that provenance as reproducible.
