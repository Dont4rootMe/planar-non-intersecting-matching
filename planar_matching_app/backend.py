"""Thin wrapper exposing the robust matching backend to the Qt UI."""

from __future__ import annotations

# Re-use the tested solver living in the project root. This avoids the previous
# index-by-index stub that could generate intersecting segments on the canvas.
from backend import solve_complete_matching_robust

__all__ = ["solve_complete_matching_robust"]
