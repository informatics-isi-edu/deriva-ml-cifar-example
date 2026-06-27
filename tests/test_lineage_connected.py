"""Regression test: source-file → image lineage is connected via a shared execution.

This test locks in the fix for the tk-011 gap, where source-file provenance was
disconnected from the Image assets uploaded by the loader. The fix (two-execution
ingest) establishes a shared execution that both produced the Image assets and
consumed the source File dataset as an Input.

The assertion chain is::

    Image_Execution row (Asset_Role="Output") → Execution RID (upload_exec_rid)
    CIFAR_Source root File dataset (source_directory=".") → source_root_rid
    → Dataset_Execution row: Dataset == source_root_rid AND Execution == upload_exec_rid

That triangle proves source-file provenance and image assets are joined through a
concrete execution record, and that the lineage graph is connected end-to-end.

Note on the Image table schema: the standard deriva-ml asset table does NOT carry
a direct ``Execution`` FK column on the asset row itself. The producing-execution
link is stored in the ``<domain_schema>.Image_Execution`` association table with an
``Asset_Role`` discriminator (``"Output"`` for producing executions, ``"Input"`` for
consuming ones). This test reads that association table to find the upload execution.

The test is gated on ``DERIVA_ML_LIVE_LOCALHOST=1`` so CI suites without a live
``dev-localhost`` container skip it automatically. To run it locally::

    export DERIVA_ML_LIVE_LOCALHOST=1
    DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/test_lineage_connected.py -v -s
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest

LIVE = os.environ.get("DERIVA_ML_LIVE_LOCALHOST") == "1"
HOSTNAME = "localhost"

# Use 2000 images — the verified count from the two-execution ingest (tk-011
# live verification on catalog 278). Enough to populate both train and test
# partitions robustly; well above the small-variant bootstrap floor.
NUM_IMAGES = 2000


pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="DERIVA_ML_LIVE_LOCALHOST=1 required; needs the dev-localhost container.",
)


def _run_cli(*args: str) -> None:
    """Invoke the ``load-cifar10`` CLI under ``uv run`` and fail loudly.

    Args:
        *args: Arguments to forward to ``load-cifar10`` (everything
            after the command name).

    Raises:
        AssertionError: If the CLI exits with a non-zero status. The
            assertion message includes the full ``stdout``/``stderr``
            so failures in CI logs are diagnosable without re-running.
    """
    cmd = ["uv", "run", "load-cifar10", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"load-cifar10 failed (exit {result.returncode}).\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def test_source_image_lineage_connected_via_shared_execution():
    """End-to-end: source File dataset and Image assets share an upload execution.

    Creates a throwaway catalog, runs the ``load-cifar10`` ingest
    (schema → register → upload), then asserts:

    1. A root ``CIFAR_Source``-typed File dataset exists with
       ``source_directory == "."``.
    2. At least one ``Image_Execution`` row with ``Asset_Role == "Output"``
       exists, giving us the upload execution RID.
    3. The ``Dataset_Execution`` association table has a row linking
       ``source_root_rid`` to ``upload_exec_rid``.

    Point 3 is the core regression assertion: it proves that the upload
    execution that produced the Image assets is the same execution that
    consumed the source File dataset as Input, closing the
    source→image lineage gap recorded as tk-011.

    Cleans up the catalog in a try/finally so a failure mid-run still
    deletes the test catalog.

    Note on column names discovered by live inspection (catalog 278):

    - ``<domain_schema>.Image_Execution``: columns ``Image``, ``Execution``,
      ``Asset_Role``. The producing execution has ``Asset_Role == "Output"``.
    - ``deriva-ml.Dataset_Execution``: columns ``Dataset``, ``Execution``,
      ``Dataset_Version``. The upload execution consuming the source File
      dataset as Input is recorded here.

    Args:
        None
    """
    from deriva.core import DerivaServer, get_credential
    from deriva_ml import DerivaML

    project_name = f"lineage-check-{uuid.uuid4().hex[:8]}"

    # --- Phase 1: schema (creates the catalog, installs domain model). ---
    _run_cli(
        "--hostname",
        HOSTNAME,
        "--create-catalog",
        project_name,
        "--phase",
        "schema",
    )

    # Resolve the alias to the freshly-created catalog id.
    server = DerivaServer("https", HOSTNAME, credentials=get_credential(HOSTNAME))
    alias = server.connect_ermrest_alias(project_name)
    catalog_id = str(alias.retrieve()["alias_target"])
    catalog = server.connect_ermrest(catalog_id)

    try:
        # --- Run register + upload phases (schema already done above). ---
        # The test only needs source registration and image upload to verify
        # the lineage connection; the datasets phase is not required for the
        # Dataset_Execution or Image_Execution assertions.
        _run_cli(
            "--hostname",
            HOSTNAME,
            "--catalog-id",
            catalog_id,
            "--phase",
            "register",
            "--num-images",
            str(NUM_IMAGES),
            "--keep-source-cache",  # skip cleanup so the run finishes faster
        )
        _run_cli(
            "--hostname",
            HOSTNAME,
            "--catalog-id",
            catalog_id,
            "--phase",
            "upload",
            "--num-images",
            str(NUM_IMAGES),
            "--keep-source-cache",  # skip cleanup so the run finishes faster
        )

        # --- Connect to the catalog. ---
        ml = DerivaML(hostname=HOSTNAME, catalog_id=catalog_id)
        pb = ml.catalog.getPathBuilder()

        # The Image asset table lives in the domain schema (= project_name,
        # which equals ml.default_schema after create_or_connect_catalog set it
        # up). We discover it at runtime so the test is not hard-coded to any
        # particular project/alias name.
        domain_schema = ml.default_schema
        print(f"\nDomain schema: {domain_schema}", file=sys.stderr)

        # --- Step 1: find the upload execution RID from Image_Execution. ---
        # The standard deriva-ml asset table does NOT carry a direct Execution
        # FK on the asset row itself. The producing-execution link is in the
        # ``<domain_schema>.Image_Execution`` association table.  Rows with
        # ``Asset_Role == "Output"`` are the producing-execution records.
        # (Verified live on catalog 278: Image_Execution columns are
        # Image / Execution / Asset_Role.)
        ie_tbl = pb.schemas[domain_schema].Image_Execution
        output_rows = list(
            ie_tbl.filter(ie_tbl.Asset_Role == "Output").entities().fetch()
        )

        assert output_rows, (
            f"No Image_Execution rows with Asset_Role='Output' found in "
            f"schema {domain_schema!r}. "
            "Did the upload phase run successfully?"
        )

        # All output rows should share the same upload execution; take the first.
        upload_exec_rid: str = str(output_rows[0]["Execution"])
        sample_image_rid: str = str(output_rows[0]["Image"])

        print(
            f"Sample Image RID: {sample_image_rid}  upload_exec_rid: {upload_exec_rid}",
            file=sys.stderr,
        )

        # --- Step 2: find the source root File dataset RID. ---
        # Use the Dataset_Dataset_Type association table directly to enumerate
        # datasets tagged CIFAR_Source, then pick the one with
        # source_directory == ".". (ml.find_datasets() does not accept
        # dataset_types= in this version of the library — verified live.)
        ds_dt = pb.schemas["deriva-ml"].Dataset_Dataset_Type
        cifar_source_rows = list(
            ds_dt.filter(ds_dt.Dataset_Type == "CIFAR_Source").entities().fetch()
        )

        assert cifar_source_rows, (
            "No CIFAR_Source-typed datasets found in Dataset_Dataset_Type. "
            "Did the register phase run successfully?"
        )

        source_root_rid: str | None = None
        for row in cifar_source_rows:
            ds = ml.lookup_dataset(row["Dataset"])
            if ds.source_directory == ".":
                source_root_rid = row["Dataset"]
                break

        assert source_root_rid is not None, (
            f"Could not find a CIFAR_Source dataset with source_directory == '.'. "
            f"Candidates: {[r['Dataset'] for r in cifar_source_rows]}"
        )

        print(f"source_root_rid: {source_root_rid}", file=sys.stderr)

        # --- Step 3: assert the connection via Dataset_Execution. ---
        # The association table ``deriva-ml:Dataset_Execution`` links a Dataset
        # to an Execution. Its FK columns are ``Dataset`` and ``Execution``
        # (confirmed by live inspection: columns are RID, RCT, RMT, RCB, RMB,
        # Dataset, Execution, Dataset_Version). A row here means the execution
        # consumed the dataset as an Input.
        de_tbl = pb.schemas["deriva-ml"].Dataset_Execution
        de_rows = list(
            de_tbl.filter(de_tbl.Dataset == source_root_rid).entities().fetch()
        )
        linked_exec_rids = {str(r["Execution"]) for r in de_rows}

        print(
            f"Dataset_Execution rows for source_root_rid {source_root_rid}: "
            f"{linked_exec_rids}",
            file=sys.stderr,
        )

        # Core regression assertion: the upload execution consumed the source
        # root File dataset as Input. If this fails, source→image lineage is
        # broken again (tk-011 regression).
        assert upload_exec_rid in linked_exec_rids, (
            f"Lineage gap detected (tk-011 regression).\n"
            f"  source_root_rid:  {source_root_rid}\n"
            f"  upload_exec_rid:  {upload_exec_rid}\n"
            f"  executions linked to source dataset: {linked_exec_rids}\n"
            "The upload execution is not recorded as a consumer of the "
            "source File dataset in Dataset_Execution. "
            "The two-execution ingest fix is broken."
        )

    finally:
        # Tear down the throwaway catalog so repeated test runs don't
        # accrete catalogs on dev-localhost.
        try:
            catalog.delete_ermrest_catalog(really=True)
        except Exception as e:  # noqa: BLE001
            print(
                f"Warning: failed to delete throwaway catalog {catalog_id}: {e}",
                file=sys.stderr,
            )
