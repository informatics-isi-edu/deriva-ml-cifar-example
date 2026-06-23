"""Unit tests for src/scripts/_cifar10_source.py."""

from __future__ import annotations

import pickle
import tarfile
from unittest.mock import patch

import numpy as np

from scripts._cifar10_source import (
    download_cifar10_archive,
    extract_cifar10_to_png,
    load_batch,
)


def _fake_batch(num_images: int, label_offset: int = 0) -> dict:
    """Build a fake CIFAR-10 batch matching the upstream pickle format."""
    return {
        b"data": np.zeros((num_images, 3072), dtype=np.uint8),
        b"labels": [(i + label_offset) % 10 for i in range(num_images)],
        b"filenames": [
            f"img_{i + label_offset}.png".encode() for i in range(num_images)
        ],
        b"batch_label": b"testing batch",
    }


def test_download_uses_cache_when_present(tmp_path):
    cache = tmp_path / "cifar-10-python.tar.gz"
    cache.write_bytes(b"fake-archive-bytes")

    # urlretrieve should NOT be called when the cache exists.
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        result = download_cifar10_archive(cache_path=cache)

    assert result == cache
    mock_retrieve.assert_not_called()


def test_load_batch_returns_images_labels_filenames(tmp_path):
    batch_path = tmp_path / "data_batch_1"
    with batch_path.open("wb") as fh:
        pickle.dump(_fake_batch(num_images=4), fh)

    images, labels, filenames = load_batch(batch_path)

    assert images.shape == (4, 32, 32, 3)
    assert images.dtype == np.uint8
    assert labels == [0, 1, 2, 3]
    assert filenames == ["img_0.png", "img_1.png", "img_2.png", "img_3.png"]


def test_load_batch_preserves_channel_order(tmp_path):
    """A wrong reshape/transpose axis order would silently pass the shape
    assertion in the other test. This pins channel ordering: CIFAR-10's
    packed layout is [R(1024), G(1024), B(1024)] per row, and the decoded
    image at HWC index (h, w, c) must come from the c-th 1024-block."""
    # Build one image where R=10, G=20, B=30 at every pixel.
    flat = np.empty((1, 3072), dtype=np.uint8)
    flat[0, 0:1024] = 10  # R block
    flat[0, 1024:2048] = 20  # G block
    flat[0, 2048:3072] = 30  # B block

    batch = {
        b"data": flat,
        b"labels": [7],
        b"filenames": [b"channel_test.png"],
        b"batch_label": b"testing batch",
    }
    batch_path = tmp_path / "data_batch_channel"
    with batch_path.open("wb") as fh:
        pickle.dump(batch, fh)

    images, _, _ = load_batch(batch_path)

    assert images.shape == (1, 32, 32, 3)
    # Every pixel should be (R=10, G=20, B=30).
    assert images[0, 0, 0, 0] == 10
    assert images[0, 0, 0, 1] == 20
    assert images[0, 0, 0, 2] == 30
    assert images[0, 15, 27, 0] == 10  # random pixel: still R=10
    assert images[0, 31, 31, 2] == 30  # last pixel: still B=30


def test_extract_writes_pngs_and_returns_labels(tmp_path):
    # Build a minimal tarball that mimics cifar-10-python.tar.gz.
    archive = tmp_path / "cifar-10-python.tar.gz"
    work = tmp_path / "build"
    cifar_dir = work / "cifar-10-batches-py"
    cifar_dir.mkdir(parents=True)

    for idx, name in enumerate(["data_batch_1", "data_batch_2", "test_batch"]):
        with (cifar_dir / name).open("wb") as fh:
            pickle.dump(_fake_batch(num_images=2, label_offset=idx * 2), fh)

    # meta file with class names (decoded against b"label_names").
    meta = {
        b"label_names": [
            b"airplane",
            b"automobile",
            b"bird",
            b"cat",
            b"deer",
            b"dog",
            b"frog",
            b"horse",
            b"ship",
            b"truck",
        ],
    }
    with (cifar_dir / "batches.meta").open("wb") as fh:
        pickle.dump(meta, fh)

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(cifar_dir, arcname="cifar-10-batches-py")

    out = tmp_path / "out"
    train_dir, test_dir, labels = extract_cifar10_to_png(archive, out)

    assert train_dir == out / "train"
    assert test_dir == out / "test"
    train_pngs = sorted(train_dir.glob("*.png"))
    test_pngs = sorted(test_dir.glob("*.png"))
    assert len(train_pngs) == 4  # 2 batches × 2 images
    assert len(test_pngs) == 2

    # Every PNG has a labels entry, and labels are class names (not ints).
    valid_classes = {
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    }
    for png in train_pngs + test_pngs:
        assert png.stem in labels
        assert labels[png.stem] in valid_classes

    # Pin specific label assignments (catches class-index off-by-one bugs).
    # _fake_batch assigns labels [i + label_offset] % 10 for i in range(num_images),
    # and our build loop passes label_offset = idx * 2:
    #   batch 1 (idx=0, num=2) → labels [0, 1] → classes [airplane, automobile]
    #   batch 2 (idx=1, num=2) → labels [2, 3] → classes [bird, cat]
    #   test_batch (idx=2, num=2) → labels [4, 5] → classes [deer, dog]
    assert labels["img_0"] == "airplane"
    assert labels["img_2"] == "bird"
    assert labels["img_4"] == "deer"
