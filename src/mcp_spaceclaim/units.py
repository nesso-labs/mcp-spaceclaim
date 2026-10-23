"""Length/angle conversion between the server's working unit and PyAnsys Geometry.

PyAnsys Geometry stores everything as pint quantities in SI (metres). Tools speak a
single working unit (millimetres by default), so the model never has to think about
metres and every number it sends or reads back is in the same unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ansys.geometry.core.math import Point2D, Point3D, UnitVector3D, Vector3D
from ansys.geometry.core.misc import UNITS, Angle, Distance

LENGTH_UNITS = {
    "mm": UNITS.mm,
    "cm": UNITS.cm,
    "m": UNITS.m,
    "in": UNITS.inch,
}


@dataclass(frozen=True)
class Units:
    """The working length unit; angles are always degrees."""

    name: str = "mm"

    def __post_init__(self) -> None:
        if self.name not in LENGTH_UNITS:
            raise ValueError(f"unsupported unit {self.name!r}; use one of {sorted(LENGTH_UNITS)}")

    @property
    def unit(self):
        return LENGTH_UNITS[self.name]

    @property
    def per_metre(self) -> float:
        """Working units in one metre (1000 for mm): scales raw SI tessellation data."""
        return float((1 * UNITS.m).m_as(self.unit))

    # --- into PyAnsys Geometry -------------------------------------------------

    def distance(self, value: float) -> Distance:
        return Distance(float(value), self.unit)

    def point2d(self, xy: Sequence[float]) -> Point2D:
        _check_len(xy, 2, "2D point")
        return Point2D([float(v) for v in xy], unit=self.unit)

    def point3d(self, xyz: Sequence[float]) -> Point3D:
        _check_len(xyz, 3, "3D point")
        return Point3D([float(v) for v in xyz], unit=self.unit)

    @staticmethod
    def angle(degrees: float) -> Angle:
        return Angle(float(degrees), UNITS.degree)

    # --- out of PyAnsys Geometry -----------------------------------------------

    def length_out(self, quantity) -> float:
        return round(float(_quantity(quantity).m_as(self.unit)), 6)

    def area_out(self, quantity) -> float:
        return round(float(_quantity(quantity).m_as(self.unit**2)), 6)

    def volume_out(self, quantity) -> float:
        return round(float(_quantity(quantity).m_as(self.unit**3)), 6)

    def point_out(self, point) -> list[float]:
        """A Point3D (or anything with x/y/z quantities) as a plain list in working units."""
        return [self.length_out(point.x), self.length_out(point.y), self.length_out(point.z)]


def _quantity(value):
    # Distance/Angle wrap a pint Quantity in `.value`; raw floats are SI.
    if hasattr(value, "m_as"):
        return value
    if hasattr(value, "value") and hasattr(value.value, "m_as"):
        return value.value
    return float(value) * UNITS.m


def _check_len(values: Sequence[float], n: int, what: str) -> None:
    if len(values) != n:
        raise ValueError(f"{what} needs {n} coordinates, got {len(values)}: {list(values)}")


AXES = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "-X": (-1.0, 0.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "-Z": (0.0, 0.0, -1.0),
}


def direction(value: str | Sequence[float]) -> UnitVector3D:
    """A unit direction from an axis name ("X", "-Z") or a 3-vector."""
    if isinstance(value, str):
        key = value.strip().upper()
        if key.startswith("+"):
            key = key[1:]
        if key not in AXES:
            raise ValueError(f"unknown axis {value!r}; use X, Y, Z, -X, -Y, -Z or [x, y, z]")
        return UnitVector3D(AXES[key])
    _check_len(value, 3, "direction")
    vec = Vector3D([float(v) for v in value])
    if vec.is_zero:
        raise ValueError("direction vector must not be zero")
    return UnitVector3D(vec)
