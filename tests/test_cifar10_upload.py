"""Tests for src/scripts/_cifar10_upload.py — pure-function tests only.

These tests exercise ``tag_url_to_path`` and ``read_labels_manifest``
without touching the catalog or network.  The catalog-bound
``run_upload_phase`` is exercised live in Task 8's end-to-end run.
"""

from __future__ import annotations

from pathlib import Path


def test_tag_url_to_path():
    from scripts._cifar10_upload import tag_url_to_path

    url = "tag://HostA,2026-06-25:file:///var/cache/cifar10_source/train/cat_2.png"
    assert (
        tag_url_to_path(url).as_posix() == "/var/cache/cifar10_source/train/cat_2.png"
    )


def test_tag_url_to_path_deep_path():
    from scripts._cifar10_upload import tag_url_to_path

    url = "tag://myhost.example.com,2025-01-01:file:///home/user/.cache/deep/nested/file.png"
    result = tag_url_to_path(url)
    assert result == Path("/home/user/.cache/deep/nested/file.png")


def test_read_labels_manifest_round_trips(tmp_path):
    from scripts._cifar10_source import write_labels_manifest
    from scripts._cifar10_upload import read_labels_manifest

    written = write_labels_manifest(tmp_path, {"cat_2": "cat", "dog_3": "dog"})
    m = read_labels_manifest(written)
    assert m["cat_2.png"] == "cat"
    assert m["dog_3.png"] == "dog"


def test_read_labels_manifest_all_classes(tmp_path):
    """Round-trip a larger manifest covering all 10 CIFAR-10 classes."""
    from scripts._cifar10_source import write_labels_manifest
    from scripts._cifar10_upload import read_labels_manifest

    labels = {
        f"{cls}_{i}": cls
        for i, cls in enumerate(
            [
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
            ]
        )
    }
    manifest = write_labels_manifest(tmp_path, labels)
    result = read_labels_manifest(manifest)

    for stem, cls in labels.items():
        assert result[f"{stem}.png"] == cls


def test_module_exposes_expected_api():
    """Public API surface check for _cifar10_upload."""
    from scripts._cifar10_upload import (
        tag_url_to_path,
        read_labels_manifest,
        run_upload_phase,
    )

    for fn in (tag_url_to_path, read_labels_manifest, run_upload_phase):
        assert callable(fn)
