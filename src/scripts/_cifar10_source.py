"""CIFAR-10 data source — download from the Toronto open mirror.

This module isolates the data-source layer (network fetch, extract,
batch decode) so it can be unit-tested without touching DerivaML.

The upstream archive is the canonical Python pickle distribution at
``https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz``. It
contains six pickle files (``data_batch_1`` .. ``data_batch_5``
and ``test_batch``) plus a ``batches.meta`` file. Each batch has
labels for every image — the Toronto distribution is fully labeled
on both train and test, unlike the Kaggle competition format.
"""

from __future__ import annotations

import csv
import logging
import pickle
import random
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deriva-ml-model-template"
CIFAR10_SOURCE_CACHE = DEFAULT_CACHE_DIR / "cifar10_source"


def download_cifar10_archive(cache_path: Path | None = None) -> Path:
    """Download the CIFAR-10 archive, or return the cached copy.

    Args:
        cache_path: Where to store the archive. Defaults to
            ``~/.cache/deriva-ml-model-template/cifar-10-python.tar.gz``.

    Returns:
        Path to the (now-present) archive file.

    Example:
        >>> archive = download_cifar10_archive()
        >>> archive.name
        'cifar-10-python.tar.gz'
    """
    if cache_path is None:
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DEFAULT_CACHE_DIR / "cifar-10-python.tar.gz"

    if cache_path.exists():
        logger.info(f"Using cached CIFAR-10 archive at {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
    urllib.request.urlretrieve(CIFAR10_URL, cache_path)
    logger.info(f"Downloaded to {cache_path}")
    return cache_path


def load_batch(batch_path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    """Load one CIFAR-10 pickle batch into image array + labels.

    Args:
        batch_path: Path to a CIFAR-10 batch pickle (``data_batch_N``
            or ``test_batch``).

    Returns:
        Tuple of ``(images, labels, filenames)``:
          - images: ``np.ndarray`` of shape ``(N, 32, 32, 3)``, ``uint8``,
            HWC, RGB.
          - labels: list of int class indices (0-9).
          - filenames: list of original filenames (str, decoded from bytes).

    Example:
        >>> imgs, labels, names = load_batch(Path("data_batch_1"))
        >>> imgs.shape
        (10000, 32, 32, 3)
    """
    with batch_path.open("rb") as fh:
        batch = pickle.load(fh, encoding="bytes")

    raw = batch[b"data"]
    images = raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = list(batch[b"labels"])
    filenames = [fn.decode("utf-8") for fn in batch[b"filenames"]]
    return images, labels, filenames


def extract_cifar10_to_png(
    archive_path: Path, output_dir: Path
) -> tuple[Path, Path, dict[str, str]]:
    """Extract the CIFAR-10 archive into a train/test PNG layout.

    Writes images as PNG files under ``output_dir/train/`` and
    ``output_dir/test/``, named to match the original CIFAR-10
    filenames (without re-numbering). Returns a labels mapping
    keyed by filename stem (no extension).

    Args:
        archive_path: Path to ``cifar-10-python.tar.gz``.
        output_dir: Directory to write ``train/`` and ``test/`` into.
            Created if it doesn't exist. A ``_extract/`` scratch
            subdirectory is created inside ``output_dir`` during
            processing and removed at the end; any pre-existing
            ``_extract/`` inside ``output_dir`` is deleted on entry.

    Returns:
        Tuple of ``(train_dir, test_dir, labels)`` where ``labels`` is
        a mapping of ``filename_stem -> class_name`` for *all* images
        (both train and test — the Toronto distribution labels both).

    Example:
        >>> train, test, labels = extract_cifar10_to_png(
        ...     Path("cifar-10-python.tar.gz"), Path("./out")
        ... )
        >>> labels["frog_42"]
        'frog'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Extract archive to a working subdir.
    extract_root = output_dir / "_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir()
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_root, filter="data")
    batches_dir = extract_root / "cifar-10-batches-py"

    # Load class names from batches.meta.
    with (batches_dir / "batches.meta").open("rb") as fh:
        meta = pickle.load(fh, encoding="bytes")
    class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

    labels: dict[str, str] = {}
    train_batches = sorted(batches_dir.glob("data_batch_*"))
    for batch_path in train_batches:
        images, lbl_ints, filenames = load_batch(batch_path)
        for img, lbl, fname in zip(images, lbl_ints, filenames):
            out_path = train_dir / fname
            Image.fromarray(img).save(out_path)
            labels[Path(fname).stem] = class_names[lbl]

    images, lbl_ints, filenames = load_batch(batches_dir / "test_batch")
    for img, lbl, fname in zip(images, lbl_ints, filenames):
        out_path = test_dir / fname
        Image.fromarray(img).save(out_path)
        labels[Path(fname).stem] = class_names[lbl]

    # Clean up the temporary extraction directory.
    shutil.rmtree(extract_root)

    return train_dir, test_dir, labels


def _stratified_pick(
    items: list[tuple[Any, str, str]], limit: int | None, seed: int
) -> list[tuple[Any, str, str]]:
    """Class-balanced pick of ``limit`` items from decoded CIFAR records.

    Args:
        items: Decoded records as ``(image_array, class_name, filename)``.
        limit: Total number to keep, spread as evenly as possible across
            classes. ``None`` keeps everything (no sampling).
        seed: RNG seed for a reproducible pick.

    Returns:
        The selected subset of ``items`` (a list, order not significant).

    Example:
        >>> picked = _stratified_pick(items, limit=100, seed=42)  # doctest: +SKIP
        >>> len(picked)
        100
    """
    if limit is None or limit >= len(items):
        return items

    rng = random.Random(seed)
    by_class: dict[str, list[tuple[Any, str, str]]] = {}
    for rec in items:
        by_class.setdefault(rec[1], []).append(rec)
    for recs in by_class.values():
        rng.shuffle(recs)

    classes = sorted(by_class)
    base, extra = divmod(limit, len(classes))
    picked: list[tuple[Any, str, str]] = []
    # Give ``extra`` leftover slots to the first classes (deterministic order).
    for i, cls in enumerate(classes):
        want = base + (1 if i < extra else 0)
        picked.extend(by_class[cls][:want])
    return picked


def extract_cifar10_sample_to_png(
    archive_path: Path,
    output_dir: Path,
    train_limit: int | None,
    test_limit: int | None,
    seed: int = 42,
) -> tuple[Path, Path, dict[str, str]]:
    """Extract a *sampled* CIFAR-10 PNG tree directly into ``output_dir``.

    Decodes the pickle batches in memory, class-balanced-samples the
    requested counts, and writes **only the sampled PNGs** into
    ``output_dir/train/`` and ``output_dir/test/``. Nothing un-sampled is
    ever written to disk — so ``output_dir`` is safe to hand straight to
    ``FileSpec.create_filespecs`` for by-reference registration (it contains
    exactly the sample). The archive's tar is unpacked to a temp dir that is
    removed before return.

    Args:
        archive_path: Path to ``cifar-10-python.tar.gz``.
        output_dir: Directory to write ``train/`` and ``test/`` into; created
            if missing. Only the sampled PNGs land here.
        train_limit: Number of training images to keep (class-balanced).
            ``None`` keeps all ~50K.
        test_limit: Number of test images to keep (class-balanced). ``None``
            keeps all ~10K.
        seed: RNG seed; the test split uses ``seed + 1`` so the two splits
            sample independently but reproducibly.

    Returns:
        ``(train_dir, test_dir, labels)`` where ``labels`` maps
        ``filename_stem -> class_name`` for the *sampled* files only.

    Example:
        >>> train, test, labels = extract_cifar10_sample_to_png(
        ...     Path("cifar-10-python.tar.gz"), Path("./out"), 1000, 1000
        ... )  # doctest: +SKIP
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cifar10_tar_") as tmp:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        batches_dir = Path(tmp) / "cifar-10-batches-py"

        with (batches_dir / "batches.meta").open("rb") as fh:
            meta = pickle.load(fh, encoding="bytes")
        class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

        def decode(batch_paths: list[Path]) -> list[tuple[Any, str, str]]:
            items: list[tuple[Any, str, str]] = []
            for bp in batch_paths:
                images, lbl_ints, filenames = load_batch(bp)
                for img, lbl, fname in zip(images, lbl_ints, filenames):
                    items.append((img, class_names[lbl], fname))
            return items

        train_items = decode(sorted(batches_dir.glob("data_batch_*")))
        test_items = decode([batches_dir / "test_batch"])

    train_pick = _stratified_pick(train_items, train_limit, seed)
    test_pick = _stratified_pick(test_items, test_limit, seed + 1)

    labels: dict[str, str] = {}
    for sub, picks in ((train_dir, train_pick), (test_dir, test_pick)):
        for img, cls, fname in picks:
            Image.fromarray(img).save(sub / fname)
            labels[Path(fname).stem] = cls

    return train_dir, test_dir, labels


def write_labels_manifest(root: Path, labels: dict[str, str]) -> Path:
    """Write a filename->class manifest to ``root/labels.csv``.

    Keys of ``labels`` are filename stems (no extension); each row is
    ``<stem>.png,<class>``. This is the durable, registered label source the
    decoupled upload execution reads (no in-memory labels dict crosses the
    execution boundary).

    Args:
        root: Directory to write the manifest into.
        labels: Mapping of filename stem (no extension) to class name.

    Returns:
        Path to the written manifest file (``root/labels.csv``).

    Example:
        >>> from pathlib import Path
        >>> labels = {"frog_42": "frog", "cat_7": "cat"}
        >>> manifest = write_labels_manifest(Path("/tmp"), labels)
        >>> manifest.name
        'labels.csv'
    """
    manifest = root / "labels.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "class"])
        for stem, cls in sorted(labels.items()):
            writer.writerow([f"{stem}.png", cls])
    return manifest
