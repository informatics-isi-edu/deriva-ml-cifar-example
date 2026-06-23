"""Smoke tests for src/scripts/_cifar10_schema.py.

Most of stage 1's behavior requires a live Deriva catalog, so this
test file is intentionally sparse — it verifies the module's
public API exists and is importable. The end-to-end behavior is
exercised in the load-cifar10 smoke test in Task A13 and in
Part B of the broader test plan.
"""

from __future__ import annotations


def test_module_exposes_expected_api():
    from scripts._cifar10_schema import (
        create_or_connect_catalog,
        setup_domain_model,
        setup_workflow_types,
        setup_dataset_types,
        apply_annotations,
        run_schema_phase,
    )

    # Sanity: each is callable.
    for fn in (
        create_or_connect_catalog,
        setup_domain_model,
        setup_workflow_types,
        setup_dataset_types,
        apply_annotations,
        run_schema_phase,
    ):
        assert callable(fn)


def test_lookup_alias_target_returns_none_on_failure():
    """Resolution failures must downgrade to None, not raise.

    The helper feeds into the create-catalog banner; a transient
    network/credential error must never mis-report a fresh create as a
    retarget. We exercise the failure path by pointing at a hostname
    that won't resolve — any exception inside the helper is swallowed
    and ``None`` is returned, which is the "alias does not exist (or
    we can't tell)" sentinel the caller expects.
    """
    from scripts._cifar10_schema import _lookup_alias_target

    # A hostname that won't resolve and an alias name with no real
    # significance. The broad except inside the helper catches the
    # DNS / credential / HTTP failure and returns None.
    assert (
        _lookup_alias_target(
            "invalid-hostname-for-test.example.invalid", "no-such-alias"
        )
        is None
    )
