"""CIFAR-10 Stage 1: catalog + schema setup.

This module demonstrates the **schema-installation pattern** for
DerivaML: how to open or create a catalog, register a domain
model (asset table, vocabulary, feature), and declare the
workflow + dataset types your loader will use.  Copy this module
as the starting point when building a loader for your own data —
everything here is idempotent and re-runnable.

It handles the schema-installation concern for the
CIFAR-10 example: creating or connecting to a catalog and
installing the domain model (Image asset table, Image_Class
vocabulary with 10 terms, Image_Classification feature),
workflow types, dataset types, and Chaise annotations.

All operations are idempotent — safe to re-run against a
catalog that's already been set up.

Public API:
    - ``create_or_connect_catalog(args)`` — open a catalog
      (existing or fresh).
    - ``setup_domain_model(ml)`` — install Image table,
      Image_Class vocab, Image_Classification feature.
    - ``setup_workflow_types(ml)`` — register the three
      workflow types we use (CIFAR_Data_Load, Image
      Classification, ROC Analysis Notebook).
    - ``setup_dataset_types(ml)`` — register the six dataset
      types (Complete, Training, Testing, Split, Labeled,
      Unlabeled).
    - ``apply_annotations(ml, project_name)`` — Chaise navbar
      branding.
    - ``run_schema_phase(ml, project_name)`` — orchestrator
      that runs all five steps in order.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from deriva.core import DerivaServer, get_credential
from deriva.core.ermrest_model import Schema
from deriva_ml import DerivaML
from deriva_ml.catalog import set_catalog_provenance
from deriva_ml.core.ermrest import ColumnDefinition
from deriva_ml.core.enums import BuiltinTypes
from deriva_ml.schema import create_or_retarget_ml_catalog

from models.cifar10_classes import CIFAR10_CLASSES

logger = logging.getLogger(__name__)

#: Dataset types used for organizing CIFAR-10 data
DATASET_TYPES: list[tuple[str, str, list[str]]] = [
    ("Complete", "A complete dataset containing all data", ["complete", "entire"]),
    (
        "Training",
        "A dataset subset used for model training",
        ["training", "train", "Train"],
    ),
    ("Testing", "A dataset subset used for model testing/evaluation", ["test", "Test"]),
    ("Split", "A dataset that contains nested dataset splits", ["split"]),
    (
        "Labeled",
        "A dataset containing records with ground truth labels",
        ["labeled", "annotated"],
    ),
    (
        "Unlabeled",
        "A dataset containing records without ground truth labels",
        ["unlabeled", "unannotated"],
    ),
]


def _lookup_alias_target(hostname: str, alias: str) -> str | None:
    """Resolve an ermrest alias to its current catalog id, or ``None``.

    Used to detect whether a re-run of ``--create-catalog <alias>`` is
    going to retarget an existing alias (orphaning the previously-aliased
    catalog) versus creating the alias for the first time. The actual
    create/retarget is still performed by
    :func:`deriva_ml.schema.create_or_retarget_ml_catalog`; this helper
    only inspects state so the loader can print an accurate banner.

    Any failure to resolve the alias is treated as "alias does not exist"
    so the banner falls through to the safe "CREATED NEW CATALOG" message
    — we never want a transient network blip to mis-report a fresh create
    as a retarget.

    Args:
        hostname: Server hostname (e.g., ``"localhost"``).
        alias: Alias name to resolve (typically the project name).

    Returns:
        The numeric catalog id (as a string) the alias currently points
        at, or ``None`` if the alias does not exist or cannot be
        resolved.

    Example:
        >>> _lookup_alias_target("localhost", "cifar10_demo")  # doctest: +SKIP
        '18'
    """
    try:
        server = DerivaServer("https", hostname, credentials=get_credential(hostname))
        existing_alias = server.connect_ermrest_alias(alias)
        target = existing_alias.alias_target
    except Exception:
        return None
    return str(target) if target is not None else None


def create_or_connect_catalog(
    args: argparse.Namespace,
) -> tuple[DerivaML, str | int, str]:
    """Connect to an existing catalog or create a new one.

    If ``args.create_catalog`` is set, creates a new catalog (with a fresh
    domain schema and catalog provenance), otherwise connects to the catalog
    identified by ``args.catalog_id``.

    Args:
        args: Parsed CLI args. Reads ``hostname``, ``catalog_id``,
            ``create_catalog``, ``domain_schema``.

    Returns:
        Tuple of ``(ml, catalog_id, domain_schema)`` — a connected
        ``DerivaML`` instance, the catalog ID, and the resolved domain
        schema name.
    """
    if args.create_catalog:
        logger.info(
            f"Creating new catalog on {args.hostname} with project name: {args.create_catalog}"
        )

        # Detect whether the alias already exists *before* calling the
        # helper, so we can print an accurate banner. The helper always
        # creates a fresh catalog and (re)points the alias at it — if the
        # alias resolved before this call, the previous catalog is being
        # orphaned by the retarget; the banner must say so. See the
        # deriva-ml docstring on create_or_retarget_ml_catalog and the
        # 2026-05-26 e2e finding setup/01 for the misleading-message bug
        # this guards against.
        prior_catalog_id = _lookup_alias_target(args.hostname, args.create_catalog)

        # create_or_retarget_ml_catalog always creates a fresh catalog and
        # points the alias at it. Re-running with the same project name
        # retargets the alias to the new catalog id; the previously-aliased
        # catalog is left in place (not auto-deleted). See deriva-ml's
        # create_schema.create_or_retarget_ml_catalog docstring.
        catalog = create_or_retarget_ml_catalog(
            args.hostname, args.create_catalog, alias=args.create_catalog
        )
        model = catalog.getCatalogModel()
        model.create_schema(Schema.define(args.create_catalog))

        catalog_id = catalog.catalog_id
        domain_schema = args.create_catalog

        print(f"\n{'=' * 60}")
        if prior_catalog_id is None:
            print("  CREATED NEW CATALOG")
        else:
            print("  RETARGETED ALIAS TO NEW CATALOG")
            print(f"  Previous catalog id (orphaned): {prior_catalog_id}")
        print(f"  Hostname:    {args.hostname}")
        print(f"  Catalog ID:  {catalog_id}")
        print(f"  Alias:       {args.create_catalog}")
        print(f"  Schema:      {domain_schema}")
        print(f"{'=' * 60}\n")

        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(catalog_id),
            domain_schemas={domain_schema},
        )

        set_catalog_provenance(
            ml.catalog,
            name=f"CIFAR-10 ({args.create_catalog})",
            description="CIFAR-10 image classification catalog created by load_cifar10.py",
            workflow_url="https://github.com/informatics-isi-edu/deriva-ml-model-template/blob/main/src/scripts/load_cifar10.py",
        )
        logger.info("Set catalog provenance")
        return ml, catalog_id, domain_schema

    logger.info(f"Connecting to {args.hostname}, catalog {args.catalog_id}")
    ml = DerivaML(
        hostname=args.hostname,
        catalog_id=str(args.catalog_id),
        domain_schemas={args.domain_schema} if args.domain_schema else None,
    )
    domain_schema = ml.default_schema
    logger.info(f"Connected to catalog, domain schema: {domain_schema}")
    return ml, args.catalog_id, domain_schema


def setup_domain_model(ml: DerivaML) -> dict[str, Any]:
    """Create the CIFAR-10 domain model in the catalog.

    Sets up the necessary tables, vocabularies, and features for storing
    CIFAR-10 image data with classification labels.

    Args:
        ml: Connected DerivaML instance.

    Returns:
        Status dictionary with keys:
        - vocabulary: {status, name} for Image_Class vocabulary
        - asset_table: {status, table_name} for Image table
        - feature: {status, feature_name} for Image_Classification

    Note:
        This function is idempotent - it checks for existing objects before
        creating them, so it's safe to run multiple times.

        Creates:
        - `Image_Class` vocabulary with 10 CIFAR-10 class terms
        - `Image` asset table for storing image files
        - `Image_Classification` feature linking images to classes
    """
    results = {}

    # Check existing vocabularies across both schemas
    vocabs = [
        v.name
        for schema in [ml.ml_schema, ml.default_schema]
        for v in ml.model.schemas[schema].tables.values()
        if ml.model.is_vocabulary(v)
    ]

    # Create Image_Class vocabulary if needed
    if "Image_Class" not in vocabs:
        logger.info("Creating Image_Class vocabulary...")
        ml.create_vocabulary(
            vocab_name="Image_Class",
            comment="CIFAR-10 image classification categories",
        )
        results["vocabulary"] = {"status": "created", "name": "Image_Class"}
    else:
        logger.info("Image_Class vocabulary already exists")
        results["vocabulary"] = {"status": "exists", "name": "Image_Class"}

    # Add class terms
    existing_terms = {t.name for t in ml.list_vocabulary_terms("Image_Class")}

    logger.info("Adding CIFAR-10 class terms...")
    for class_name, description, synonyms in CIFAR10_CLASSES:
        if class_name not in existing_terms:
            ml.add_term(
                table="Image_Class",
                term_name=class_name,
                description=description,
                synonyms=synonyms,
            )
            logger.info(f"  Added term: {class_name}")
        else:
            logger.info(f"  Term exists: {class_name}")

    # Check existing tables in domain schema
    tables = [t.name for t in ml.model.schemas[ml.default_schema].tables.values()]

    # Create Image asset table if needed
    if "Image" not in tables:
        logger.info("Creating Image asset table...")
        ml.create_asset(
            asset_name="Image",
            column_defs=[],
            comment="CIFAR-10 32x32 RGB images",
        )
        results["asset_table"] = {"status": "created", "table_name": "Image"}
    else:
        logger.info("Image asset table already exists")
        results["asset_table"] = {"status": "exists", "table_name": "Image"}

    # Enable Image as dataset element type
    logger.info("Enabling Image as dataset element type...")
    element_types = {t.name for t in ml.list_dataset_element_types()}
    if "Image" not in element_types:
        ml.add_dataset_element_type("Image")

    # Create Image_Classification feature with Confidence score
    # - Image_Class: vocabulary term for the predicted/actual class
    # - Confidence: float value (0-1) representing prediction confidence/probability
    logger.info("Creating Image_Classification feature...")
    confidence_column = ColumnDefinition(
        name="Confidence",
        type=BuiltinTypes.float4,
        nullok=True,
        comment="Prediction confidence/probability (0-1)",
    )
    try:
        ml.create_feature(
            target_table="Image",
            feature_name="Image_Classification",
            comment="CIFAR-10 class label and optional confidence score for each image",
            terms=["Image_Class"],
            metadata=[confidence_column],
            optional=["Confidence"],
        )
        results["feature"] = {
            "status": "created",
            "feature_name": "Image_Classification",
        }
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("Image_Classification feature already exists")
            results["feature"] = {
                "status": "exists",
                "feature_name": "Image_Classification",
            }
        else:
            raise

    return results


def setup_workflow_types(ml: DerivaML) -> None:
    """Ensure required workflow types exist.

    Creates workflow types in the Workflow_Type vocabulary for tracking
    CIFAR-10 data loading and model training executions.

    Args:
        ml: Connected DerivaML instance.
    """
    existing_types = {t.name for t in ml.list_vocabulary_terms("Workflow_Type")}

    if "CIFAR_Data_Load" not in existing_types:
        logger.info("Creating CIFAR_Data_Load workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="CIFAR_Data_Load",
            description="Workflow for loading CIFAR-10 dataset into catalog",
        )

    if "Image Classification" not in existing_types:
        logger.info("Creating Image Classification workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="Image Classification",
            description="Workflows for training and evaluating image classification models",
        )

    if "ROC Analysis Notebook" not in existing_types:
        logger.info("Creating ROC Analysis Notebook workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="ROC Analysis Notebook",
            description="Jupyter notebook that computes ROC curves and AUC scores for classification experiments",
        )

    if "Dataset_Split" not in existing_types:
        logger.info("Creating Dataset_Split workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="Dataset_Split",
            description=(
                "Workflow that partitions an existing dataset into train/test "
                "(and optionally validation) child datasets via "
                "deriva_ml.dataset.split.split_dataset"
            ),
        )


def setup_dataset_types(ml: DerivaML) -> None:
    """Ensure required dataset types exist in Dataset_Type vocabulary.

    Creates the dataset type terms needed for organizing CIFAR-10 data
    into Complete, Training, Testing, and Split datasets.

    Args:
        ml: Connected DerivaML instance.
    """
    logger.info("Setting up dataset types...")

    existing_terms = {t.name for t in ml.list_vocabulary_terms("Dataset_Type")}

    for type_name, description, synonyms in DATASET_TYPES:
        if type_name not in existing_terms:
            try:
                ml.add_term(
                    table="Dataset_Type",
                    term_name=type_name,
                    description=description,
                    synonyms=synonyms,
                )
                logger.info(f"  Added dataset type: {type_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not add dataset type {type_name}: {e}")


def apply_annotations(ml: DerivaML, project_name: str) -> None:
    """Apply catalog Chaise annotations (navbar branding, page title).

    Args:
        ml: Connected DerivaML instance.
        project_name: Used in the navbar brand and head title.

    Example:
        >>> apply_annotations(ml, "cifar10_demo")
    """
    ml.apply_catalog_annotations(
        navbar_brand_text=f"CIFAR-10 ({project_name})",
        head_title="CIFAR-10 ML Catalog",
    )


def run_schema_phase(ml: DerivaML, project_name: str) -> None:
    """Run Stage 1 end-to-end against a connected catalog.

    Sets up the domain model, workflow types, dataset types, and
    applies Chaise annotations. Idempotent — safe to re-run on a
    catalog that already has the schema installed.

    Args:
        ml: Connected DerivaML instance.
        project_name: Used for catalog annotations.

    Example:
        >>> ml, _, _ = create_or_connect_catalog(args)
        >>> run_schema_phase(ml, project_name="cifar10_demo")
    """
    logger.info("Setting up domain model...")
    setup_domain_model(ml)
    logger.info("Domain model setup complete")

    logger.info("Applying catalog annotations...")
    apply_annotations(ml, project_name)

    setup_workflow_types(ml)
    setup_dataset_types(ml)
