"""DerivaML Connection Configuration.

Configuration Group: ``deriva_ml``

This group specifies which Deriva catalog to connect to.

The shipped ``default_deriva`` is a **placeholder** — it points at no catalog
in particular. Edit it for your environment, or override at the CLI:

    deriva-ml-run --host <hostname> --catalog <id> ...

For multi-environment work, register additional configs (one per host/catalog)
in ``src/configs/dev/deriva.py`` and select with
``deriva_ml=<name>``.
"""

from hydra_zen import store
from deriva_ml import DerivaMLConfig

deriva_store = store(group="deriva_ml")

# REQUIRED: ``default_deriva`` is used when no connection is specified.
# Replace ``hostname`` and ``catalog_id`` with your catalog before running, or
# always pass --host/--catalog on the CLI.
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="localhost",
    catalog_id=0,  # placeholder — set to your catalog ID, or override via --catalog
    use_minid=False,
    zen_meta={
        "description": (
            "Placeholder connection. Replace hostname/catalog_id with your "
            "catalog, or override at the CLI with --host/--catalog."
        )
    },
)
