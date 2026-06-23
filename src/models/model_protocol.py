"""
DerivaML Model Protocol - Re-exported from deriva_ml
=====================================================

This module re-exports the DerivaMLModel protocol from deriva_ml.execution.

The protocol is now maintained in deriva-ml itself. This re-export is kept
for backwards compatibility.

See deriva_ml.execution.model_protocol for full documentation.
"""

# Re-export from deriva-ml
from deriva_ml.execution import DerivaMLModel

__all__ = ["DerivaMLModel"]
