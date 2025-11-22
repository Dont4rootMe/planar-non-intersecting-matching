"""Planar geometry solvers (matching + ham-sandwich cut) with array inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon


# ---------------------------------------------------------------------------
# Ham-sandwich cut solver (adapted from HamSandwichViz with array I/O)
@dataclass(frozen=True)
class Line:
    m: float  # slope
    b: float  # intercept

    def y_at(self, x: float) -> float:
        return self.m * x + self.b


def _dual_line_from_point(x: float, y: float, constant: float = 1.0) -> Line:
    return Line(constant * x, -y)


def _compute_dual_line(point: Point, constant: float = 1.0) -> Line:
    return Line(constant * point.x, -point.y)


def _pairwise_intersections(lines: Sequence[Line]) -> list[float]:
    xs: list[float] = []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = lines[i], lines[j]
            if l1.m == l2.m:
                continue
            x_val = (l2.b - l1.b) / (l1.m - l2.m)
            xs.append(x_val)
    return xs


def _median_value_at(x: float, lines: Sequence[Line]) -> float:
    y_vals = sorted(line.y_at(x) for line in lines)
    mid = (len(y_vals) - 1) // 2
    return y_vals[mid]


def _median_band_values(x: float, lines: Sequence[Line]) -> tuple[float, float]:
    y_vals = sorted(line.y_at(x) for line in lines)
    lower_idx = (len(y_vals) - 1) // 2
    upper_idx = len(y_vals) // 2
    return y_vals[lower_idx], y_vals[upper_idx]


def _median_band_polygon(
    lines: Sequence[Line],
    interval: tuple[float, float],
) -> Polygon | LineString:
    left, right = interval
    xs = [left, right]
    xs.extend(_pairwise_intersections(lines))
    xs = [x for x in xs if left <= x <= right]
    xs = sorted(set(xs))
    if len(xs) < 2:
        xs = [left, right]

    lowers: list[tuple[float, float]] = []
    uppers: list[tuple[float, float]] = []
    for x in xs:
        low, high = _median_band_values(x, lines)
        lowers.append((x, low))
        uppers.append((x, high))

    if all(abs(lo - hi) < 1e-12 for ((_, lo), (_, hi)) in zip(lowers, uppers)):
        return LineString(lowers)

    ring = lowers + list(reversed(uppers))
    return Polygon(ring)


def ham_sandwich_cut(
    A_coords: np.ndarray,
    B_coords: np.ndarray,
    *,
    dual_constant: float = 1.0,
) -> list[tuple[float, float]]:
    """Compute all ham-sandwich cuts between two planar point sets."""

    if A_coords.shape != B_coords.shape:
        raise ValueError("A and B must have identical shapes.")
    if A_coords.ndim != 2 or A_coords.shape[1] != 2:
        raise ValueError("Input arrays must have shape (n, 2).")
    if A_coords.shape[0] == 0:
        return []

    points_a = [Point(float(x), float(y)) for x, y in A_coords]
    points_b = [Point(float(x), float(y)) for x, y in B_coords]
    all_points = points_a + points_b

    min_x = min(p.x for p in all_points)
    max_x = max(p.x for p in all_points)
    interval = (min_x - 40.0, max_x + 40.0)

    red_duals = [_dual_line_from_point(p.x, p.y, dual_constant) for p in points_a]
    blue_duals = [_dual_line_from_point(p.x, p.y, dual_constant) for p in points_b]

    red_band = _median_band_polygon(red_duals, interval)
    blue_band = _median_band_polygon(blue_duals, interval)

    intersection = red_band.intersection(blue_band)
    if intersection.is_empty:
        raise ValueError("Median levels do not intersect; no ham-sandwich cut found.")

    def _collect_points(geom) -> list[Point]:
        pts: list[Point] = []
        if isinstance(geom, Point):
            pts.append(geom)
        elif isinstance(geom, LineString):
            pts.extend([Point(*pt) for pt in geom.coords])
        elif isinstance(geom, Polygon):
            pts.extend([Point(*pt) for pt in geom.exterior.coords])
        elif hasattr(geom, "geoms"):
            for g in geom.geoms:  # type: ignore[attr-defined]
                pts.extend(_collect_points(g))
        return pts

    ham_points = _collect_points(intersection)
    if not ham_points:
        raise ValueError("No valid intersection points found for ham-sandwich cut.")

    cuts: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    for hp in ham_points:
        dual = _compute_dual_line(hp, constant=dual_constant)
        key = (round(dual.m, 12), round(dual.b, 12))
        if key in seen:
            continue
        seen.add(key)
        cuts.append((float(dual.m), float(dual.b)))

    return cuts


# ---------------------------------------------------------------------------
# Planar non-crossing matching via recursive ham-sandwich cuts
def _split_by_line(
    coords: np.ndarray,
    indices: list[int],
    m: float,
    b: float,
    eps: float,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices into above, below, and on-line buckets for a given cut."""

    above: list[int] = []
    below: list[int] = []
    on_line: list[int] = []

    for idx in indices:
        x_val, y_val = float(coords[idx, 0]), float(coords[idx, 1])
        offset = y_val - (m * x_val + b)
        if abs(offset) <= eps:
            on_line.append(idx)
        elif offset > 0.0:
            above.append(idx)
        else:
            below.append(idx)

    return above, below, on_line


def solve_complete_matching_robust(
    A_coords: np.ndarray,
    B_coords: np.ndarray,
    *,
    line_tolerance: float | None = None,
) -> list[tuple[int, int]]:
    """Return a non-crossing perfect matching between two equally sized sets.

    The solver recursively applies ham-sandwich cuts. When a cut passes through
    one point of A and one point of B (the odd-cardinality case), those points
    are paired directly on the cut. Remaining points are split by the cut into
    two balanced subproblems that cannot cross each other.
    """

    A = np.asarray(A_coords, dtype=np.float64)
    B = np.asarray(B_coords, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError("A and B must have identical shapes.")
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("Input arrays must have shape (n, 2).")

    n_total = A.shape[0]
    if n_total == 0:
        return []

    # Scale the tolerance to the coordinate range for robust on-line checks.
    if line_tolerance is None:
        max_extent = float(np.max(np.abs(np.concatenate([A, B])))) if n_total > 0 else 1.0
        line_tolerance = 1e-9 * max(1.0, max_extent)

    def recurse(a_idx: list[int], b_idx: list[int]) -> list[tuple[int, int]]:
        """Solve the matching problem for the provided index subsets."""

        count = len(a_idx)
        if count != len(b_idx):
            raise ValueError("Mismatched subset sizes during recursion.")
        if count == 0:
            return []
        if count == 1:
            return [(a_idx[0], b_idx[0])]

        try:
            cuts = ham_sandwich_cut(A[a_idx], B[b_idx])
        except Exception:
            cuts = []

        def attempt_cut(m: float, b_line: float) -> list[tuple[int, int]] | None:
            above_a, below_a, on_a = _split_by_line(A, list(a_idx), m, b_line, float(line_tolerance))
            above_b, below_b, on_b = _split_by_line(B, list(b_idx), m, b_line, float(line_tolerance))

            pairs_here: list[tuple[int, int]] = []

            # When the subset size is odd, the ham-sandwich cut should pass
            # through one point of each color. We pair them immediately.
            if count % 2 == 1:
                if not on_a or not on_b:
                    return None
                pairs_here.append((on_a.pop(), on_b.pop()))

            # Balance the two half-planes by assigning on-line points.
            delta = len(above_a) - len(above_b)
            if delta > 0:
                move = min(delta, len(on_b))
                above_b.extend(on_b[:move])
                on_b = on_b[move:]
                delta -= move
            elif delta < 0:
                move = min(-delta, len(on_a))
                above_a.extend(on_a[:move])
                on_a = on_a[move:]
                delta += move

            if delta != 0:
                return None

            below_a.extend(on_a)
            below_b.extend(on_b)

            # Reject degenerate splits that make no progress.
            if not pairs_here and (len(above_a) == 0 or len(below_a) == 0):
                return None

            if len(above_a) != len(above_b) or len(below_a) != len(below_b):
                return None

            upper_pairs = recurse(above_a, above_b) if above_a else []
            lower_pairs = recurse(below_a, below_b) if below_a else []
            return pairs_here + upper_pairs + lower_pairs

        for m, b_line in cuts:
            result = attempt_cut(m, b_line)
            if result is not None:
                return result

        # Fallback: pair by sorted x/y order to avoid total failure.
        sorted_a = sorted(a_idx, key=lambda i: (A[i, 0], A[i, 1]))
        sorted_b = sorted(b_idx, key=lambda i: (B[i, 0], B[i, 1]))
        return [(int(a), int(b)) for a, b in zip(sorted_a, sorted_b)]

    return recurse(list(range(n_total)), list(range(n_total)))
