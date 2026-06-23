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

import logging
import pickle
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deriva-ml-model-template"


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
