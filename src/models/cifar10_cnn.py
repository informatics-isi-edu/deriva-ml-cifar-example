"""CIFAR-10 2-Layer CNN Model.

A small convolutional network for CIFAR-10 classification, used as the
canonical end-to-end example of integrating PyTorch with DerivaML.

Architecture:
- Conv2d(3, 32) -> ReLU -> MaxPool2d
- Conv2d(32, 64) -> ReLU -> MaxPool2d
- Linear(64*8*8, hidden_size) -> ReLU
- Linear(hidden_size, 10)

Expected accuracy: ~60-70% with default parameters.

File layout (read top-to-bottom to follow the runner):

1. **Model + training/eval primitives** (pure PyTorch, no DerivaML).
   ``SimpleCNN``, ``train_one_epoch``, ``evaluate``, ``predict_batch``.
2. **DerivaML harness** (the only place catalog concerns enter).
   ``build_loaders`` walks ``execution.datasets`` and produces
   DataLoaders by ``Dataset_Type``. ``record_predictions`` writes
   feature rows + a CSV asset. ``save_training_artifacts`` bundles
   weights + log into ``Execution_Asset`` files.
3. **Entry point** ``cifar10_cnn`` dispatches train-vs-eval and
   sequences the harness calls in order.

The ML pieces never see ``Execution`` or ``DerivaML``; the harness
pieces never see logits. The boundary is the DataLoader (in) and the
list-of-prediction-records (out).
"""

from __future__ import annotations

import csv
import random
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from deriva_ml import DerivaML, MLAsset, ExecAssetType
from deriva_ml.dataset import DatasetBag
from deriva_ml.execution import Execution

from models.cifar10_classes import CIFAR10_CLASS_NAMES, CIFAR10_CLASS_TO_IDX


# CIFAR-10's 32x32 RGB images normalized to [-1, 1].
_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# ---------------------------------------------------------------------------
# Model + training/eval primitives (pure PyTorch)
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """A simple 2-layer CNN for CIFAR-10 classification.

    Architecture:
        Conv(3 → conv1_channels, 3x3, pad=1) → ReLU → MaxPool(2x2)
        Conv(conv1_channels → conv2_channels, 3x3, pad=1) → ReLU → MaxPool(2x2)
        Flatten → Linear(conv2_channels*8*8 → hidden_size) → ReLU → Dropout
        Linear(hidden_size → num_classes)

    Args:
        conv1_channels: Output channels for the first conv layer.
        conv2_channels: Output channels for the second conv layer.
        hidden_size: Hidden layer size in the fully-connected stack.
        dropout_rate: Dropout probability after the hidden layer.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_size: int = 128,
        dropout_rate: float = 0.0,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        # After two 2x2 pools: 32x32 -> 16x16 -> 8x8.
        self.fc1 = nn.Linear(conv2_channels * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy_pct)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels, _rids in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    """Evaluate the model on a loader. Returns (avg_loss, accuracy_pct, n_labeled).

    Unlabeled rows (label == -1) are skipped for loss/accuracy — only
    labeled rows contribute. ``n_labeled`` is the count of labeled rows
    actually used; if zero, ``avg_loss`` and ``accuracy_pct`` are NaN.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss_sum = 0.0
    loss_batches = 0
    correct = 0
    n_labeled = 0
    with torch.no_grad():
        for inputs, labels, _rids in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if torch.isfinite(loss):  # skip all-unlabeled batches
                loss_sum += loss.item()
                loss_batches += 1
            _, predicted = outputs.max(1)
            labeled_mask = labels != -1
            n_labeled += int(labeled_mask.sum().item())
            correct += int((predicted.eq(labels) & labeled_mask).sum().item())
    if n_labeled == 0:
        return float("nan"), float("nan"), 0
    return loss_sum / max(loss_batches, 1), 100.0 * correct / n_labeled, n_labeled


def predict_batch(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run inference and return one record per element.

    Each record is a dict ``{rid, predicted_class, confidence, probs}``
    where ``probs`` is a numpy array of per-class probabilities in
    ``class_names`` order. Pure PyTorch — no catalog calls.
    """
    model.eval()
    out: list[dict[str, Any]] = []
    with torch.no_grad():
        for inputs, _labels, rids in loader:
            inputs = inputs.to(device)
            probabilities = F.softmax(model(inputs), dim=1)
            confidences, predicted = probabilities.max(1)
            for i, rid in enumerate(rids):
                out.append(
                    {
                        "rid": rid,
                        "predicted_class": class_names[predicted[i].item()],
                        "confidence": confidences[i].item(),
                        "probs": probabilities[i].cpu().numpy(),
                    }
                )
    return out


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training.

    The training DataLoader is given its own ``torch.Generator`` in
    :func:`build_loaders` so that shuffling is reproducible too. This
    function handles only the global RNG state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only: don't fail on CUDA ops without a deterministic kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# DerivaML harness (the only code that knows about Execution / DatasetBag)
# ---------------------------------------------------------------------------


# Recognized Dataset_Type role terms. Qualifiers like ``Labeled`` are
# orthogonal and ignored here; only role terms pick a lane. To add a new
# role, add the vocabulary term to the catalog AND add an entry to
# ``build_loaders``' dispatch loop.
_ROLE_TRAINING = "training"
_ROLE_TESTING = "testing"
_ROLE_VALIDATION = "validation"
_ROLE_SPLIT = "split"
_LEAF_ROLES = (_ROLE_TRAINING, _ROLE_TESTING, _ROLE_VALIDATION)


def _bag_role(bag: DatasetBag) -> str | None:
    """Return the leaf role of a bag, or None if it has no recognized role.

    A bag's ``Dataset_Type`` is a set of catalog vocabulary terms (one or
    more of ``Training``/``Testing``/``Validation``/``Split``/qualifiers
    like ``Labeled``). We pick the first leaf role found; ``Split`` is
    handled upstream by ``_flatten_to_leaves`` (it expands to children,
    then this is called on each child).
    """
    roles = {t.lower() for t in bag.dataset_types}
    for role in _LEAF_ROLES:
        if role in roles:
            return role
    return None


def _load_image(path: Any, _row: dict[str, Any]) -> PIL.Image.Image:
    return PIL.Image.open(path).convert("RGB")


def _target_to_class_idx(rec: Any) -> int:
    return CIFAR10_CLASS_TO_IDX[rec.Image_Class]


def _rid_collate(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate ``(tensor, int, str)`` triples — strings can't be tensorised."""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    rids = [b[2] for b in batch]
    return images, labels, rids


@dataclass(frozen=True)
class _LaneConfig:
    """Per-role DataLoader policy.

    Adding a new role (a new vocabulary term in ``_LEAF_ROLES``) is one
    entry here plus one line in ``build_loaders``' return tuple.
    """

    missing: str
    """``bag.as_torch_dataset`` policy for elements with no feature
    value. ``'skip'`` drops them (training/validation: loss/accuracy
    are undefined without labels); ``'unknown'`` keeps them with a
    sentinel target (testing: we still want to record predictions on
    unlabeled rows)."""

    shuffle: bool
    """Whether the DataLoader shuffles between epochs. Only training."""

    use_seeded_generator: bool
    """Whether to give the DataLoader a seeded ``torch.Generator`` for
    reproducible shuffle order. Only matters when ``shuffle=True``."""


_LANE_CONFIGS: dict[str, _LaneConfig] = {
    _ROLE_TRAINING: _LaneConfig(
        missing="skip",
        shuffle=True,
        use_seeded_generator=True,
    ),
    _ROLE_TESTING: _LaneConfig(
        missing="unknown",
        shuffle=False,
        use_seeded_generator=False,
    ),
    _ROLE_VALIDATION: _LaneConfig(
        missing="skip",
        shuffle=False,
        use_seeded_generator=False,
    ),
}


def _flatten_to_leaves(execution: Execution) -> list[DatasetBag]:
    """Walk ``execution.datasets``, expand any ``Split`` parent to its
    leaf children. Non-Split bags pass through.
    """
    leaves: list[DatasetBag] = []
    for bag in execution.datasets:
        roles = {t.lower() for t in bag.dataset_types}
        if _ROLE_SPLIT in roles:
            leaves.extend(bag.list_dataset_children())
        else:
            leaves.append(bag)
    return leaves


def _make_loader(
    bag: DatasetBag,
    role: str,
    batch_size: int,
    seed: int | None,
) -> DataLoader:
    """Build one DataLoader for one bag's role-specific lane.

    Looks up the role's policy in ``_LANE_CONFIGS``. Adding a new role
    means adding an entry there; this function doesn't need to change.
    """
    cfg = _LANE_CONFIGS[role]
    dataset = bag.as_torch_dataset(
        element_type="Image",
        sample_loader=_load_image,
        transform=_TRANSFORM,
        targets=["Image_Classification"],
        target_transform=_target_to_class_idx,
        missing=cfg.missing,
    )
    generator = None
    if cfg.use_seeded_generator and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    # macOS DataLoader: num_workers=0 to avoid fork() + MPS deadlock.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=cfg.shuffle,
        num_workers=0,
        collate_fn=_rid_collate,
        generator=generator,
    )


def build_loaders(
    execution: Execution,
    batch_size: int,
    require_training: bool = False,
    seed: int | None = None,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None, list[str]]:
    """Walk ``execution.datasets`` and build DataLoaders by ``Dataset_Type``.

    The harness:

      1. Fail fast if ``execution.datasets`` is empty (misconfig: a
         Hydra ``datasets=<group>`` override resolved to an empty list).
      2. Flatten any ``Split`` parent bag to its children.
      3. For each leaf bag, look up its role and build a DataLoader
         from the role's lane policy (``_LANE_CONFIGS``).
      4. If ``require_training`` is set and no Training bag was found,
         raise — fail loudly rather than silently produce a non-training
         "training run".

    Args:
        execution: DerivaML execution; ``execution.datasets`` is the
            list of downloaded bags.
        batch_size: Batch size for all loaders.
        require_training: If True, raise ``RuntimeError`` when no bag
            dispatches to the training lane.
        seed: Drives the training DataLoader's shuffle order. ``None``
            uses PyTorch's default global generator.

    Returns:
        ``(train_loader, test_loader, val_loader, class_names)``. Any
        loader may be ``None`` if no bag dispatched to that lane. The
        Validation lane (D01) is wired through to the training loop as
        a per-epoch metric but doesn't drive save-best — that's
        intentional for a demo. Plug early stopping in here if you
        want it.

    Raises:
        RuntimeError: If ``execution.datasets`` is empty (step 1), or
            if ``require_training=True`` and no Training-typed bag is
            present after flattening (step 4).
    """
    # Fail fast on the misconfig case: a Hydra `datasets=<group>` group
    # that resolved to an empty list. This typically means the
    # placeholder registries in src/configs/datasets.py weren't filled
    # in for the current catalog. The execution row in the catalog
    # will already be open by the time we get here (deriva-ml creates
    # it before invoking the model function), so this is the earliest
    # the runner itself can catch it. Full upstream prevention would
    # need a deriva-ml pre_check hook.
    if not execution.datasets:
        raise RuntimeError(
            "Execution has no input datasets. This typically means a "
            "Hydra `datasets=<group>` override resolved to an empty list "
            "(e.g. a placeholder `datasets_store([], name=...)` in "
            "src/configs/datasets.py that wasn't filled in for this "
            "catalog). Fill in the dataset RIDs for the catalog you're "
            "running against, or pass a different `datasets=<group>` "
            "override."
        )

    leaves = _flatten_to_leaves(execution)
    loaders: dict[str, DataLoader] = {}
    for bag in leaves:
        role = _bag_role(bag)
        if role is None:
            warnings.warn(
                f"Bag {bag.dataset_rid} has "
                f"Dataset_Type={list(bag.dataset_types)!r}; no recognised "
                f"role among {list(_LEAF_ROLES)}. Skipping.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        loaders[role] = _make_loader(bag, role, batch_size, seed)
        print(f"  {role.capitalize()} samples: {len(loaders[role].dataset)}")

    if require_training and _ROLE_TRAINING not in loaders:
        # Safety rail: fail loudly when the primary input is missing.
        # Distinct from the empty-datasets check at the top: here we
        # had bags but none dispatched to the Training lane.
        diag = (
            "\n".join(
                f"  - {b.dataset_rid}: Dataset_Type={list(b.dataset_types)!r}"
                for b in leaves
            )
            or "  (no input bags after flattening)"
        )
        raise RuntimeError(
            "No bag with Dataset_Type=Training found in execution input. "
            "Cannot train. Input bags after flattening Split parents:\n"
            f"{diag}\n"
            "Add a Training-typed dataset to the execution config "
            "(see src/configs/datasets.py)."
        )

    return (
        loaders.get(_ROLE_TRAINING),
        loaders.get(_ROLE_TESTING),
        loaders.get(_ROLE_VALIDATION),
        list(CIFAR10_CLASS_NAMES),
    )


def record_predictions(
    predictions: list[dict[str, Any]],
    class_names: list[str],
    execution: Execution,
    ml_instance: DerivaML,
    source_label: str,
    emission_accuracy: tuple[float, int] | None = None,
) -> None:
    """Write predictions to the catalog as feature rows + a CSV asset.

    ``predictions`` is the output of :func:`predict_batch` — one dict per
    image with keys ``rid``, ``predicted_class``, ``confidence``, ``probs``.

    Two outputs are produced:

    1. **Catalog feature rows** — one ``Image_Classification`` per image
       carrying ``Image_Class`` and ``Confidence``. These are queryable
       via the standard feature surface.
    2. **CSV asset** ``prediction_probabilities.csv`` — wider, includes
       per-class probability columns and a ``Source_Label`` column.
       ``Source_Label`` records which model state produced these
       predictions (e.g. ``"epoch_10"``, ``"evaluation"``) so a
       downstream consumer reading the CSV can correlate against the
       matching line of ``training_log.txt``.

    The catalog feature row does NOT carry ``Source_Label`` (would
    require a schema migration); CSV is the source-label surface.

    Args:
        predictions: Output of :func:`predict_batch`.
        class_names: List of class names in index order. Used to label
            CSV probability columns.
        execution: DerivaML execution context.
        ml_instance: DerivaML instance for catalog access.
        source_label: Provenance tag describing which model state these
            predictions reflect (e.g. ``"epoch_10"``, ``"evaluation"``).
        emission_accuracy: Optional ``(accuracy_pct, n_labeled)`` tuple
            from a ground-truth-aware :func:`evaluate` on the same
            loader these predictions came from. When provided, the
            accuracy is printed alongside the "Recorded N predictions"
            line. This is the number a downstream consumer will
            reproduce by joining the committed CSV against the
            ground-truth feature — printing it here closes the
            provenance loop: if it diverges from the training log's
            same-epoch ``test_acc``, the divergence is visible at
            training time rather than from CSV archaeology later.
    """
    if not predictions:
        print("  WARNING: No predictions to record")
        return

    ImageClassification = ml_instance.feature_record_class(
        "Image", "Image_Classification"
    )
    feature_records = [
        ImageClassification(
            Image=p["rid"],
            Image_Class=p["predicted_class"],
            Confidence=p["confidence"],
        )
        for p in predictions
    ]
    execution.add_features(feature_records)
    print(
        f"  Recorded {len(feature_records)} predictions (source_label={source_label!r})"
    )
    if emission_accuracy is not None:
        acc, n_labeled = emission_accuracy
        if n_labeled > 0:
            print(
                f"    Emission-time accuracy: {acc:.2f}% "
                f"({n_labeled} labeled samples). "
                f"This is what a downstream consumer will recompute "
                f"from the committed CSV joined against the ground-truth "
                f"feature. Compare against the {source_label} line of "
                f"training_log.txt — any divergence means the two "
                f"accuracies were measured on different model state."
            )

    csv_file = execution.asset_file_path(
        MLAsset.execution_asset,
        "prediction_probabilities.csv",
        description=(
            "Per-image predicted class and probability distributions. "
            "Source_Label records the model state (e.g. epoch_N, evaluation)."
        ),
    )
    fieldnames = ["Image_RID", "Source_Label", "Predicted_Class", "Confidence"] + [
        f"prob_{c}" for c in class_names
    ]
    with csv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            row = {
                "Image_RID": p["rid"],
                "Source_Label": source_label,
                "Predicted_Class": p["predicted_class"],
                "Confidence": p["confidence"],
            }
            for j, c in enumerate(class_names):
                row[f"prob_{c}"] = p["probs"][j]
            writer.writerow(row)
    print(f"  Saved probability CSV to: {csv_file}")


def save_training_artifacts(
    execution: Execution,
    model: nn.Module,
    optimizer: optim.Optimizer,
    training_log: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Write weights checkpoint + human-readable training log as assets."""
    weights_file = execution.asset_file_path(
        MLAsset.execution_asset,
        "cifar10_cnn_weights.pt",
        ExecAssetType.model_file,
        description="Trained CNN weights, optimizer state, and per-epoch log",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "training_log": training_log,
        },
        weights_file,
    )
    print(f"  Saved weights to: {weights_file}")

    log_file = execution.asset_file_path(
        MLAsset.execution_asset,
        "training_log.txt",
        description="Per-epoch training log: loss, accuracy, hyperparameters",
    )
    with log_file.open("w") as f:
        f.write("CIFAR-10 CNN Training Log\n")
        f.write("=" * 50 + "\n\n")
        f.write("Config:\n")
        for k, v in sorted(config.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nProgress:\n")
        for entry in training_log:
            parts = [
                f"Epoch {entry['epoch']}",
                f"train_loss={entry['train_loss']:.4f}",
                f"train_acc={entry['train_acc']:.2f}%",
            ]
            for key in ("test_loss", "test_acc", "val_loss", "val_acc"):
                if key in entry:
                    if "loss" in key:
                        parts.append(f"{key}={entry[key]:.4f}")
                    else:
                        parts.append(f"{key}={entry[key]:.2f}%")
            f.write("  " + ", ".join(parts) + "\n")
    print(f"  Saved log to: {log_file}")


def _find_weights_asset(execution: Execution, filename: str):
    """Locate a named weights file in the execution's input assets, or None."""
    for paths in execution.asset_paths.values():
        for asset_path in paths:
            if asset_path.name == filename:
                return asset_path
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def cifar10_cnn(
    # Model architecture
    conv1_channels: int = 32,
    conv2_channels: int = 64,
    hidden_size: int = 128,
    dropout_rate: float = 0.0,
    # Training
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    seed: int = 42,
    # Test-only mode
    test_only: bool = False,
    weights_filename: str = "cifar10_cnn_weights.pt",
    # DerivaML integration (always passed by deriva-ml-run)
    ml_instance: DerivaML | None = None,
    execution: Execution | None = None,
) -> None:
    """Train or evaluate a 2-layer CNN on CIFAR-10 data from DerivaML datasets.

    Two modes share the same entry point so a single Hydra config can
    flip between them:

    * ``test_only=False`` (default): train ``epochs`` epochs on the
      training bag, optionally tracking test/val metrics per epoch.
      Save weights + training log; record final-epoch predictions on
      the test bag.
    * ``test_only=True``: load weights from an Execution_Asset and
      record predictions on the test bag without training.

    Args:
        conv1_channels: Output channels for the first conv layer.
        conv2_channels: Output channels for the second conv layer.
        hidden_size: Hidden FC-layer size.
        dropout_rate: Dropout probability after the hidden layer.
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
        batch_size: Batch size for all DataLoaders.
        weight_decay: L2 regularization coefficient.
        seed: RNG seed for byte-reproducible training. Drives weight
            init, training-loop shuffle order, and numpy/random calls.
        test_only: Skip training; load weights and run evaluation.
        weights_filename: Asset filename of the weights to load in
            test_only mode.
        ml_instance: DerivaML handle (supplied by deriva-ml-run).
        execution: Execution context (supplied by deriva-ml-run).
    """
    if ml_instance is None or execution is None:
        raise ValueError("ml_instance and execution are required")

    mode = "Test-only" if test_only else "Training"
    print(f"CIFAR-10 CNN {mode}")
    print(f"  Host: {ml_instance.host_name}, Catalog: {ml_instance.catalog_id}")
    print(
        f"  Architecture: conv1={conv1_channels}, conv2={conv2_channels}, hidden={hidden_size}"
    )
    if not test_only:
        print(
            f"  Training: lr={learning_rate}, epochs={epochs}, batch_size={batch_size}, seed={seed}"
        )

    # Seed BEFORE building the model so weight init is reproducible.
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = SimpleCNN(
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
    ).to(device)

    print("\nBuilding DataLoaders from execution datasets...")
    train_loader, test_loader, val_loader, class_names = build_loaders(
        execution, batch_size, require_training=not test_only, seed=seed
    )

    # -------------------- test-only mode --------------------
    if test_only:
        if test_loader is None:
            print("ERROR: No test data found (need a Dataset_Type=Testing bag).")
            return

        weights_path = _find_weights_asset(execution, weights_filename)
        if weights_path is None:
            available = [
                p.name for paths in execution.asset_paths.values() for p in paths
            ]
            print(f"ERROR: Weights file {weights_filename!r} not found.")
            print(f"  Available assets: {available}")
            return

        print(f"\nLoading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if "config" in checkpoint:
            print(f"  Checkpoint config: {checkpoint['config']}")
            # Rebuild model from checkpoint config if it differs.
            cfg = checkpoint["config"]
            model = SimpleCNN(
                conv1_channels=cfg.get("conv1_channels", conv1_channels),
                conv2_channels=cfg.get("conv2_channels", conv2_channels),
                hidden_size=cfg.get("hidden_size", hidden_size),
                dropout_rate=cfg.get("dropout_rate", dropout_rate),
            ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_acc, n_labeled = evaluate(model, test_loader, device)
        if n_labeled > 0:
            print(f"\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")
        else:
            print("\nTest set is unlabeled — skipping accuracy/loss reporting.")

        print("\nRecording test predictions to catalog...")
        predictions = predict_batch(model, test_loader, class_names, device)
        record_predictions(
            predictions,
            class_names,
            execution,
            ml_instance,
            source_label="evaluation",
            emission_accuracy=(test_acc, n_labeled),
        )

        # Compact evaluation summary as a sibling asset.
        results_file = execution.asset_file_path(
            MLAsset.execution_asset,
            "evaluation_results.txt",
            description="Test set evaluation summary: loss, accuracy, configuration",
        )
        with results_file.open("w") as f:
            f.write("CIFAR-10 CNN Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Weights file: {weights_filename}\n")
            f.write(f"Labeled test samples: {n_labeled}\n")
            f.write(f"Test loss: {test_loss:.4f}\n")
            f.write(f"Test accuracy: {test_acc:.2f}%\n")
        print(f"  Saved results to: {results_file}")
        print("\nEvaluation complete!")
        return

    # -------------------- training mode --------------------
    assert train_loader is not None  # guaranteed by require_training above
    print(f"  Training batches: {len(train_loader)}")
    if test_loader:
        print(f"  Test batches: {len(test_loader)}")
    if val_loader:
        print(f"  Validation batches: {len(val_loader)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    print("\nTraining...")
    training_log: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
        }
        line = f"  Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%"

        if test_loader is not None:
            test_loss, test_acc, n_labeled = evaluate(model, test_loader, device)
            if n_labeled > 0:
                entry["test_loss"] = test_loss
                entry["test_acc"] = test_acc
                line += f", test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%"
            else:
                line += " (test set unlabeled — no test metrics)"

        # Validation lane wiring: surfaces a generalization signal in
        # the training log. Doesn't drive save-best — this is a demo;
        # plug your early-stopping policy in here if you want one.
        if val_loader is not None:
            val_loss, val_acc, n_val = evaluate(model, val_loader, device)
            if n_val > 0:
                entry["val_loss"] = val_loss
                entry["val_acc"] = val_acc
                line += f", val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"

        print(line)
        training_log.append(entry)

    # Persist the final-epoch model and the log.
    print("\nSaving model...")
    save_training_artifacts(
        execution=execution,
        model=model,
        optimizer=optimizer,
        training_log=training_log,
        config={
            "conv1_channels": conv1_channels,
            "conv2_channels": conv2_channels,
            "hidden_size": hidden_size,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "seed": seed,
        },
    )

    # Final-epoch predictions on the test bag, tagged with the epoch
    # that produced them so a downstream reader can correlate against
    # the matching line of training_log.txt.
    #
    # Also re-evaluate on the same loader so we can print an
    # emission-time accuracy alongside the recorded predictions. This
    # is the number a downstream consumer reproduces from the
    # committed CSV; if it diverges from the final-epoch test_acc in
    # the training log, the desync is visible right here rather than
    # surfacing later when someone joins the CSV against ground truth.
    if test_loader is not None:
        print("\nRecording test predictions to catalog...")
        _, emit_acc, n_emit = evaluate(model, test_loader, device)
        predictions = predict_batch(model, test_loader, class_names, device)
        record_predictions(
            predictions,
            class_names,
            execution,
            ml_instance,
            source_label=f"epoch_{epochs}",
            emission_accuracy=(emit_acc, n_emit),
        )

    print("\nTraining complete!")
