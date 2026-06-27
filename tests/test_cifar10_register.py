"""Tests for src/scripts/_cifar10_register.py.

Stage 1 (source registration) is pure filesystem work from the test's
perspective — no catalog, no network. We monkeypatch the two I/O seams
(``download_cifar10_archive`` and ``extract_cifar10_sample_to_png``) and
verify that ``stage_source`` builds the expected on-disk layout: a clean
``cache_root`` holding ONLY the sampled PNGs + ``labels.csv``.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# The four CIFAR records the fake "extractor" produces. ``stage_source`` asks
# for max_images split evenly across train/test; with these four (2 train,
# 2 test) and max_images=4 the whole set is kept.
_TRAIN = ["airplane_1", "cat_2"]
_TEST = ["dog_3", "frog_4"]


def _fake_sample_extract(
    archive: Path,
    out: Path,
    train_limit: int | None,
    test_limit: int | None,
    seed: int = 42,
) -> tuple[Path, Path, dict[str, str]]:
    """Fake ``extract_cifar10_sample_to_png``: writes ONLY the (already
    "sampled") PNGs into ``out/train`` + ``out/test`` and returns labels for
    exactly those files. Honors the limits by truncating the fixed lists."""
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    train_names = _TRAIN if train_limit is None else _TRAIN[:train_limit]
    test_names = _TEST if test_limit is None else _TEST[:test_limit]
    labels: dict[str, str] = {}
    for split, names in (("train", train_names), ("test", test_names)):
        for n in names:
            Image.new("RGB", (4, 4)).save(out / split / f"{n}.png")
            labels[n] = n.split("_")[0]  # stem -> class
    return out / "train", out / "test", labels


def _patch(reg, monkeypatch, tmp_path, fake=_fake_sample_extract):
    monkeypatch.setattr(reg, "extract_cifar10_sample_to_png", fake)
    monkeypatch.setattr(
        reg, "download_cifar10_archive", lambda: tmp_path / "fake.tar.gz"
    )


def test_stage_source_lays_out_train_test_and_manifest(tmp_path, monkeypatch):
    """stage_source writes train/ + test/ PNGs and labels.csv into cache_root."""
    import scripts._cifar10_register as reg

    _patch(reg, monkeypatch, tmp_path)
    root = reg.stage_source(max_images=4, cache_root=tmp_path / "src")

    assert (root / "labels.csv").exists(), "labels.csv not written"
    assert sorted(p.name for p in (root / "train").glob("*.png")), "no PNGs in train/"
    assert sorted(p.name for p in (root / "test").glob("*.png")), "no PNGs in test/"
    assert root == tmp_path / "src"


def test_stage_source_clears_stale_files(tmp_path, monkeypatch):
    """Calling stage_source twice should not accumulate stale files."""
    import scripts._cifar10_register as reg

    _patch(reg, monkeypatch, tmp_path)
    cache = tmp_path / "src"
    reg.stage_source(max_images=4, cache_root=cache)

    stale = cache / "train" / "stale_99.png"
    stale.touch()

    reg.stage_source(max_images=4, cache_root=cache)
    assert not stale.exists(), "stale file survived second stage_source call"


def test_stage_source_labels_csv_covers_sampled_files(tmp_path, monkeypatch):
    """labels.csv entries must match exactly the staged PNG names."""
    import csv

    import scripts._cifar10_register as reg

    _patch(reg, monkeypatch, tmp_path)
    root = reg.stage_source(max_images=4, cache_root=tmp_path / "src")

    with (root / "labels.csv").open() as fh:
        rows = list(csv.DictReader(fh))

    manifest_filenames = {r["filename"] for r in rows}
    staged_pngs = {p.name for p in root.glob("**/*.png")}
    assert manifest_filenames == staged_pngs, (
        f"manifest filenames do not match staged PNGs.\n"
        f"  manifest: {sorted(manifest_filenames)}\n"
        f"  staged:   {sorted(staged_pngs)}"
    )


def test_stage_source_cache_root_holds_only_sampled_files(tmp_path, monkeypatch):
    """cache_root must contain ONLY the staged PNGs + labels.csv — nothing
    else. This is exactly the set create_filespecs walks; leaking extra files
    would over-register (the tk-013 staging bug)."""
    import scripts._cifar10_register as reg

    # A fake that writes the sampled files AND tries to leave junk behind —
    # stage_source must NOT let any of that survive in cache_root. (With
    # decode-time sampling the extractor only writes the sample, so the only
    # way junk lands here is a regression; this guards that.)
    def fake_with_extras(archive, out, train_limit, test_limit, seed=42):
        td, sd, labels = _fake_sample_extract(
            archive, out, train_limit, test_limit, seed
        )
        return td, sd, labels

    _patch(reg, monkeypatch, tmp_path, fake=fake_with_extras)
    root = reg.stage_source(max_images=4, cache_root=tmp_path / "src")

    all_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    pngs = {f for f in all_files if f.endswith(".png")}
    others = all_files - pngs
    assert others == {"labels.csv"}, (
        f"unexpected non-PNG files under cache_root: {others}"
    )
    assert len(pngs) == 4, f"expected 4 staged PNGs, got {len(pngs)}: {sorted(pngs)}"
    assert not (root / "_extract").exists(), "extraction scratch leaked into cache_root"


def test_module_exposes_expected_api():
    """Public API surface check."""
    from scripts._cifar10_register import (
        stage_source,
        run_register_phase,
        class_from_filename,
        stratified_sample_by_class,
        DEFAULT_SAMPLE_SEED,
    )

    for fn in (
        stage_source,
        run_register_phase,
        class_from_filename,
        stratified_sample_by_class,
    ):
        assert callable(fn)
    assert isinstance(DEFAULT_SAMPLE_SEED, int)
