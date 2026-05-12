"""Symbolic regions and the ASIP frontier tree.

Public API:

* :class:`Region` (abstract), :class:`EmptyRegion`, :class:`UnconstrainedRegion`,
  :class:`AxisAlignedBox`, :class:`GeneralRegion`
* :func:`build_region` — dispatch a formula to the right region kind
* :class:`SampleBatch`, :class:`Status`
* :class:`Frontier`, :class:`FrontierNode`
"""

from __future__ import annotations

from ._base import Region, SampleBatch, Status
from ._concrete import (
    AxisAlignedBox,
    EmptyRegion,
    GeneralRegion,
    UnconstrainedRegion,
    build_region,
)
from ._frontier import Frontier, FrontierNode

__all__ = [
    "AxisAlignedBox",
    "EmptyRegion",
    "Frontier",
    "FrontierNode",
    "GeneralRegion",
    "Region",
    "SampleBatch",
    "Status",
    "UnconstrainedRegion",
    "build_region",
]
