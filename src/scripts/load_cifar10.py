#!/usr/bin/env python3
"""CIFAR-10 Dataset Loader for DerivaML — orchestrator + CLI.

This is the thin entry point. The actual work lives in three
focused modules — see ``CIFAR10.md`` §"Loader Walkthrough" for
a guided tour of how they compose:

    - :mod:`scripts._cifar10_schema`   (Stage 1: catalog + schema)
    - :mod:`scripts._cifar10_assets`   (Stage 2: upload + features)
    - :mod:`scripts._cifar10_datasets` (Stage 3: dataset hierarchy)

This script wires those three stages together for the common
end-to-end case and exposes ``--phase`` for running a single
stage when resuming a partial load.

Prerequisites:
    Deriva Authentication: ``deriva-globus-auth-utils login --host <hostname>``

Usage:
    Full end-to-end run::

        load-cifar10 --hostname localhost --create-catalog cifar10_demo --num-images 500

    Load into an existing catalog::

        load-cifar10 --hostname ml.derivacloud.org --catalog-id 99

    Run a single stage (resume after a partial failure)::

        load-cifar10 --hostname localhost --catalog-id 99 --phase schema
        load-cifar10 --hostname localhost --catalog-id 99 --phase images
        load-cifar10 --hostname localhost --catalog-id 99 --phase datasets

    Dry run (schema only, no image download)::

        load-cifar10 --hostname localhost --create-catalog test --dry-run

    Show Chaise URLs in the summary::

        load-cifar10 --hostname localhost --create-catalog demo --show-urls
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from scripts._cifar10_assets import run_assets_phase
from scripts._cifar10_datasets import run_datasets_phase
from scripts._cifar10_schema import create_or_connect_catalog, run_schema_phase

# Logging configuration ------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)
logger.propagate = False

_deriva_ml_logger = logging.getLogger("deriva_ml")
_deriva_ml_logger.setLevel(logging.INFO)
_deriva_ml_logger.addHandler(_handler)
_deriva_ml_logger.propagate = False

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 dataset into a DerivaML catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hostname", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--catalog-id")
    group.add_argument(
        "--create-catalog",
        metavar="PROJECT_NAME",
        help=(
            "Create a fresh catalog and point the alias PROJECT_NAME at it. "
            "Re-running creates a new catalog and retargets the alias; the "
            "previously-aliased catalog is left in place (delete with "
            "delete_ermrest_catalog if no longer needed)."
        ),
    )
    parser.add_argument("--domain-schema")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-images", type=int, default=None, metavar="N")
    parser.add_argument("--show-urls", action="store_true")
    parser.add_argument(
        "--phase",
        choices=["all", "schema", "images", "datasets"],
        default="all",
        help=(
            "Run a single phase. 'schema' is idempotent; 'images' uploads + "
            "features; 'datasets' creates the hierarchy. Default: 'all'."
        ),
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    """Route to one or more stages based on ``--phase``.

    Args:
        args: Parsed command-line arguments. If ``None``, arguments are
            parsed from ``sys.argv``.

    Returns:
        Exit code: ``0`` for success.
    """
    if args is None:
        args = parse_args()

    phase = getattr(args, "phase", "all")
    ml, catalog_id, domain_schema = create_or_connect_catalog(args)
    project_name = args.create_catalog if args.create_catalog else domain_schema

    asset_stats: dict[str, Any] = {}
    datasets: dict[str, str] = {}

    if phase in ("all", "schema"):
        run_schema_phase(ml, project_name)
        if phase == "schema":
            # Echo the catalog id in the completion banner so a setup
            # persona running --phase schema (the most common first-run
            # invocation) never has to re-run the command just to recover
            # the id. See 2026-05-26 e2e finding setup/01.
            _print_done(
                "SCHEMA PHASE COMPLETE",
                f"Catalog ID: {catalog_id} — re-run with "
                f"--catalog-id {catalog_id} --phase images or --phase datasets.",
            )
            return 0

    if phase in ("all", "images") and not args.dry_run:
        asset_stats = run_assets_phase(
            ml, max_images=args.num_images, batch_size=args.batch_size
        )

    if phase in ("all", "datasets") and not args.dry_run:
        datasets = run_datasets_phase(ml, batch_size=args.batch_size)

    _print_summary(args, catalog_id, domain_schema, datasets, asset_stats, ml)
    return 0


def _print_done(title: str, hint: str) -> None:
    """Print a two-line completion banner.

    Args:
        title: Banner heading (e.g. "SCHEMA PHASE COMPLETE").
        hint: One-line follow-up instruction shown beneath the title.
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print(f"  {hint}")
    print("=" * 60 + "\n")


def _print_summary(
    args: argparse.Namespace,
    catalog_id: str | int,
    domain_schema: str,
    datasets: dict[str, str],
    asset_stats: dict[str, Any],
    ml,
) -> None:
    """Print the final summary banner.

    Args:
        args: Parsed CLI args (reads hostname, show_urls).
        catalog_id: Catalog ID that was loaded into.
        domain_schema: Domain schema name.
        datasets: Mapping of dataset name to RID.
        asset_stats: Stats returned by run_assets_phase (may be empty).
        ml: Connected DerivaML instance (used for URL resolution).
    """
    dataset_urls: dict[str, str] = {}
    if args.show_urls and datasets:
        logger.info("Fetching Chaise URLs for datasets...")
        for name, rid in datasets.items():
            try:
                dataset_urls[name] = ml.cite(rid, current=True)
                logger.info(f"  {name}: {dataset_urls[name]}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  Failed to get URL for {name}: {e}")
                dataset_urls[name] = ""

    print("\n" + "=" * 60)
    print("  CIFAR-10 LOADING COMPLETE")
    print("=" * 60)
    print(f"  Hostname:      {args.hostname}")
    print(f"  Catalog ID:    {catalog_id}")
    print(f"  Schema:        {domain_schema}")
    print("")
    if datasets:
        print("  Datasets created:")
        dataset_display = [
            ("Complete (Labeled)", "complete"),
            ("Split", "split"),
            ("Training (Labeled)", "training"),
            ("Testing (Labeled)", "testing"),
            # Small_Split parent dataset dropped in v1.42 migration —
            # Small_Training/Small_Testing are sibling subsample()
            # outputs, not children of a Small_Split.
            ("Small_Training (Labeled, Subsample)", "small_training"),
            ("Small_Testing (Labeled, Subsample)", "small_testing"),
            ("Labeled_Split", "labeled_split"),
            ("Labeled_Training", "labeled_training"),
            ("Labeled_Testing", "labeled_testing"),
            ("Small_Labeled_Split", "small_labeled_split"),
            ("Small_Labeled_Training", "small_labeled_training"),
            ("Small_Labeled_Testing", "small_labeled_testing"),
        ]
        for display_name, key in dataset_display:
            if key in datasets:
                rid = datasets[key]
                if args.show_urls and dataset_urls:
                    print(f"    - {display_name}: {rid}")
                    print(f"      URL: {dataset_urls.get(key, 'N/A')}")
                else:
                    print(f"    - {display_name}: {rid}")
    if asset_stats:
        print("")
        print(f"  Images loaded: {asset_stats.get('total_images', 'n/a')}")
        print(f"    - Training: {asset_stats.get('training_images', 'n/a')}")
        print(f"    - Testing:  {asset_stats.get('testing_images', 'n/a')}")
        print(f"  Features added: {asset_stats.get('features_added', 'n/a')}")
    if not args.show_urls:
        print("")
        print("  Tip: Use --show-urls to display Chaise URLs for each dataset")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    sys.exit(main(parse_args()))
