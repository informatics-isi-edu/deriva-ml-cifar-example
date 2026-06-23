"""CIFAR-10 class label definitions.

A single source of truth for the 10 CIFAR-10 categories. Used by both the
data loader (which registers them as ``Image_Class`` vocabulary terms with
descriptions and synonyms) and the model (which uses the names list as the
fallback class ordering when a labeled-test bag isn't available).

Keep ordering stable — the index is the model's class index.
"""

from typing import Final


#: CIFAR-10 class definitions: ``(name, description, synonyms)``.
#:
#: The vocabulary loader iterates this list verbatim. The model only uses
#: ``name`` (index = class id), ignoring description and synonyms.
CIFAR10_CLASSES: Final[list[tuple[str, str, list[str]]]] = [
    ("airplane", "Fixed-wing aircraft", ["plane", "aeroplane"]),
    ("automobile", "Motor vehicle with four wheels", ["car", "auto"]),
    ("bird", "Feathered flying vertebrate", []),
    ("cat", "Small domestic feline", ["kitten"]),
    ("deer", "Hoofed ruminant mammal", []),
    ("dog", "Domestic canine", ["puppy"]),
    ("frog", "Tailless amphibian", ["toad"]),
    ("horse", "Large domesticated hoofed mammal", ["pony"]),
    ("ship", "Large watercraft", ["boat", "vessel"]),
    ("truck", "Motor vehicle for transporting cargo", ["lorry"]),
]


#: The class names in index order (just the first element of each tuple).
CIFAR10_CLASS_NAMES: Final[list[str]] = [name for name, _, _ in CIFAR10_CLASSES]


#: Reverse lookup: class name → index.
CIFAR10_CLASS_TO_IDX: Final[dict[str, int]] = {
    name: i for i, name in enumerate(CIFAR10_CLASS_NAMES)
}
