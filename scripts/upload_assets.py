"""Upload asset files to a catalog asset table.

This script uploads files (model weights, checkpoints, etc.) to a catalog asset
table with full provenance tracking. It creates a DerivaML execution, stages
the files, commits them with ``Execution.commit_output_assets`` (ADR-0009),
and sets descriptions on the uploaded records.

Files and their descriptions are defined in a TOML manifest file.
See ``assets.toml`` for the default manifest format.

Usage:
    # Upload using default manifest (assets.toml)
    uv run python scripts/upload_assets.py

    # Upload a specific file
    uv run python scripts/upload_assets.py path/to/model.pth

    # Upload with a custom manifest
    uv run python scripts/upload_assets.py --manifest my_assets.toml

    # Upload to a specific host
    uv run python scripts/upload_assets.py --host www.example.org

    # Upload to a different asset table
    uv run python scripts/upload_assets.py --table Execution_Asset

    # Dry run (show what would be uploaded without uploading)
    uv run python scripts/upload_assets.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11 fallback

from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration

# ---------------------------------------------------------------------------
# Defaults — update these for your project
# ---------------------------------------------------------------------------

DEFAULT_HOSTNAME = "localhost"
DEFAULT_CATALOG_ID = "6"
DEFAULT_TABLE = "Model_Artifact"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "assets.toml"

# Note: deriva-ml v2.0 (ADR-0009) removed the per-call upload tuning
# parameters (chunk_size, timeout, max_retries, retry_delay). Tune
# HTTP session timeout and Hatrac chunk size on the DerivaML instance
# itself when the upstream surface gains support; until then, defaults
# apply for all uploads here.


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load a TOML manifest mapping filenames to descriptions.

    The manifest format is:

        [files]
        "model_weights.pth" = "Description of these weights."
        "another_model.pth" = "Description of these other weights."

    Args:
        manifest_path: Path to the TOML manifest file.

    Returns:
        Dict mapping filename (str) to description (str).
    """
    with manifest_path.open("rb") as f:
        data = tomllib.load(f)
    return data.get("files", {})


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------


def upload_assets(
    files: dict[str, str],
    file_paths: list[Path],
    hostname: str,
    catalog_id: str,
    table: str,
    dry_run: bool = False,
) -> None:
    """Upload files to a catalog asset table.

    Args:
        files: Mapping of filename -> description (from manifest or CLI).
        file_paths: Resolved paths to the files to upload.
        hostname: Catalog hostname (e.g., localhost).
        catalog_id: Catalog identifier (e.g., 6).
        table: Target asset table name (e.g., Model_Artifact).
        dry_run: If True, show what would be uploaded without uploading.
    """
    # Validate all files exist before starting
    missing = [p for p in file_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: File not found: {p}", file=sys.stderr)
        sys.exit(1)

    print(f"Host: {hostname}, Catalog: {catalog_id}, Table: {table}")
    print(f"Files to upload: {len(file_paths)}")
    for p in file_paths:
        size_mb = p.stat().st_size / 1e6
        desc = files.get(p.name, "(no description)")
        print(f"  {p.name} ({size_mb:.1f} MB) — {desc[:80]}")

    if dry_run:
        print("\n[dry run] No uploads performed.")
        return

    # Connect and create execution for provenance tracking
    ml = DerivaML(hostname, catalog_id)

    workflow = ml.create_workflow(
        name="Upload Assets",
        workflow_type="Ingest",
        description="Upload asset files to the catalog.",
    )

    config = ExecutionConfiguration(
        workflow=workflow,
        description=(
            f"Upload {len(file_paths)} file(s) to {table}: "
            + ", ".join(p.name for p in file_paths)
        ),
    )

    # Use the execution as a context manager — exiting transitions it through
    # Running → Stopped, which is the prerequisite for commit_output_assets.
    with ml.create_execution(config) as execution:
        print(f"\nExecution: {ml.cite(execution.execution_rid)}")

        # Stage each file for upload. asset_file_path registers it in the
        # execution's manifest; commit_output_assets will then transfer
        # everything in the staging directory.
        for filepath in file_paths:
            print(f"Staging {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)")
            execution.asset_file_path(table, str(filepath))

    # Commit output assets with progress reporting. Run after the context
    # exit (the public API contract — committing inside the context would
    # race with the Stopped transition).
    def progress(p):
        print(f"  {p.file_name}: {p.percent_complete:.1f}%", flush=True)

    print("\nCommitting output assets...", flush=True)
    execution.commit_output_assets(progress_callback=progress)
    print("Upload done!", flush=True)

    # Set descriptions on the uploaded records.
    # We fetch all records from the table and match by filename.
    if any(files.get(p.name) for p in file_paths):
        print("Setting descriptions...", flush=True)
        schema = ml.domain_schema
        pb = ml.pathBuilder()
        records = pb.schemas[schema].tables[table].entities().fetch()
        for r in records:
            desc = files.get(r["Filename"])
            if desc:
                pb.schemas[schema].tables[table].update(
                    [{"RID": r["RID"], "Description": desc}]
                )
                print(f"  {r['RID']}: {r['Filename']} — description set")

    print("Done!", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Upload asset files to a DerivaML catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all files from the default manifest (assets.toml)
  uv run python scripts/upload_assets.py

  # Upload specific files (descriptions come from manifest if available)
  uv run python scripts/upload_assets.py weights/my_model.pth

  # Upload to a remote catalog
  uv run python scripts/upload_assets.py \\
      --host www.example.org --table Model_Artifact weights/model.pth

  # Preview what would be uploaded
  uv run python scripts/upload_assets.py --dry-run
""",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=("Files to upload. If omitted, uploads all files listed in the manifest."),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"TOML manifest mapping filenames to descriptions (default: {DEFAULT_MANIFEST.name})",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOSTNAME,
        help=f"Catalog hostname (default: {DEFAULT_HOSTNAME})",
    )
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG_ID,
        help=f"Catalog ID (default: {DEFAULT_CATALOG_ID})",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Target asset table (default: {DEFAULT_TABLE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading.",
    )

    args = parser.parse_args()

    # Load manifest for descriptions (and optionally for file list)
    manifest_files: dict[str, str] = {}
    if args.manifest.exists():
        manifest_files = load_manifest(args.manifest)
    elif not args.files:
        print(
            f"ERROR: No files specified and manifest not found: {args.manifest}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine which files to upload
    if args.files:
        # Explicit file paths from CLI
        file_paths = [p.resolve() for p in args.files]
    else:
        # All files from manifest — look in the repo root's weights/ directory
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
        file_paths = [weights_dir / name for name in manifest_files]

    if not file_paths:
        print("No files to upload.", file=sys.stderr)
        sys.exit(1)

    upload_assets(
        files=manifest_files,
        file_paths=file_paths,
        hostname=args.host,
        catalog_id=args.catalog,
        table=args.table,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
