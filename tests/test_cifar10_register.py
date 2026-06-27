"""Tests for src/scripts/_cifar10_register.py.

Stage 1 (source registration) is pure filesystem work from the test's
perspective — no catalog, no network. We monkeypatch the two I/O
seams (``download_cifar10_archive`` and ``extract_cifar10_to_png``)
and verify that ``stage_source`` builds the expected on-disk layout.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def _fake_extract(archive: Path, out: Path) -> tuple[Path, Path, dict[str, str]]:
    """Fake extract: write 2 train + 2 test PNGs and return labels."""
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    pairs = {
        "train": ["airplane_1", "cat_2"],
        "test": ["dog_3", "frog_4"],
    }
    labels: dict[str, str] = {}
    for split, names in pairs.items():
        for n in names:
            img_path = out / split / f"{n}.png"
            Image.new("RGB", (4, 4)).save(img_path)
            # label stem -> class (first word before underscore)
            cls = n.split("_")[0]
            labels[n] = cls
    return out / "train", out / "test", labels


def test_stage_source_lays_out_train_test_and_manifest(tmp_path, monkeypatch):
    """stage_source builds train/ + test/ symlinks and writes labels.csv."""
    import scripts._cifar10_register as reg

    monkeypatch.setattr(reg, "extract_cifar10_to_png", _fake_extract)
    monkeypatch.setattr(
        reg, "download_cifar10_archive", lambda: tmp_path / "fake.tar.gz"
    )

    root = reg.stage_source(max_images=4, cache_root=tmp_path / "src")

    # labels.csv must exist at the cache root
    assert (root / "labels.csv").exists(), "labels.csv not written"

    # train/ and test/ must each contain at least one PNG
    train_pngs = sorted(p.name for p in (root / "train").glob("*.png"))
    test_pngs = sorted(p.name for p in (root / "test").glob("*.png"))
    assert train_pngs, "no PNGs in train/"
    assert test_pngs, "no PNGs in test/"

    # returned path == cache_root
    assert root == tmp_path / "src"


def test_stage_source_clears_stale_files(tmp_path, monkeypatch):
    """Calling stage_source twice should not accumulate stale files."""
    import scripts._cifar10_register as reg

    monkeypatch.setattr(reg, "extract_cifar10_to_png", _fake_extract)
    monkeypatch.setattr(
        reg, "download_cifar10_archive", lambda: tmp_path / "fake.tar.gz"
    )

    cache = tmp_path / "src"
    reg.stage_source(max_images=4, cache_root=cache)

    # Plant a stale file in train/
    stale = cache / "train" / "stale_99.png"
    stale.touch()

    reg.stage_source(max_images=4, cache_root=cache)

    # Stale file should be gone after second run
    assert not stale.exists(), "stale file survived second stage_source call"


def test_stage_source_labels_csv_covers_sampled_files(tmp_path, monkeypatch):
    """labels.csv entries must match exactly the symlinked PNG stems."""
    import csv

    import scripts._cifar10_register as reg

    monkeypatch.setattr(reg, "extract_cifar10_to_png", _fake_extract)
    monkeypatch.setattr(
        reg, "download_cifar10_archive", lambda: tmp_path / "fake.tar.gz"
    )

    root = reg.stage_source(max_images=4, cache_root=tmp_path / "src")

    manifest = root / "labels.csv"
    with manifest.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    manifest_filenames = {r["filename"] for r in rows}
    staged_pngs = {p.name for p in root.glob("**/*.png")}

    assert manifest_filenames == staged_pngs, (
        f"manifest filenames do not match staged PNGs.\n"
        f"  manifest: {sorted(manifest_filenames)}\n"
        f"  staged:   {sorted(staged_pngs)}"
    )


def _fake_extract_with_extras(
    archive: Path, out: Path
) -> tuple[Path, Path, dict[str, str]]:
    """Like _fake_extract but also writes extra junk files into ``out`` —
    mimicking the real extractor, which unpacks the FULL ~60K-file corpus
    plus scratch dirs into the extraction target. Used to prove cache_root
    does NOT pick up anything beyond the sampled files (regression for the
    tk-013 staging bug, where create_filespecs(cache_root) walked the whole
    extraction)."""
    train_dir, test_dir, labels = _fake_extract(archive, out)
    # extra files in the extraction dir that must NOT end up under cache_root
    (out / "batches.meta").write_text("meta")
    (out / "train" / "unsampled_extra.png").write_bytes(b"x")
    (out / "_scratch").mkdir(exist_ok=True)
    (out / "_scratch" / "junk.bin").write_bytes(b"y")
    return train_dir, test_dir, labels


def test_stage_source_cache_root_holds_only_sampled_files(tmp_path, monkeypatch):
    """cache_root must contain ONLY the sampled PNGs + labels.csv — no
    extraction scratch, no full corpus. This is what create_filespecs walks,
    so leaking the extraction here would register thousands of extra files
    (tk-013)."""
    import scripts._cifar10_register as reg

    monkeypatch.setattr(reg, "extract_cifar10_to_png", _fake_extract_with_extras)
    monkeypatch.setattr(
        reg, "download_cifar10_archive", lambda: tmp_path / "fake.tar.gz"
    )

    root = reg.stage_source(max_images=4, cache_root=tmp_path / "src")

    # No extraction scratch leaked into cache_root.
    assert not (root / "_extract").exists(), "_extract/ leaked into cache_root"
    assert not (root / "_scratch").exists(), "_scratch/ leaked into cache_root"
    assert not (root / "batches.meta").exists(), "extraction junk leaked"

    # Every file under cache_root is either a staged PNG or labels.csv —
    # nothing else. (This is the exact set create_filespecs would register.)
    all_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    pngs = {f for f in all_files if f.endswith(".png")}
    others = all_files - pngs
    assert others == {"labels.csv"}, (
        f"unexpected non-PNG files under cache_root: {others}"
    )
    # exactly the 4 sampled images, not the extra unsampled one
    assert len(pngs) == 4, f"expected 4 sampled PNGs, got {len(pngs)}: {sorted(pngs)}"
    assert "train/unsampled_extra.png" not in all_files


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
