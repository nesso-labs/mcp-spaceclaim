"""Declarative sketch specs (what the model sends) -> PyAnsys Geometry sketches.

Everything here is client-side: building a ``Sketch`` needs no server, which is what
lets the test suite exercise the real PyAnsys objects without SpaceClaim.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from ansys.geometry.core.math import Plane, Point3D, UnitVector3D, Vector3D
from ansys.geometry.core.sketch import Sketch
from pydantic import BaseModel, Field

from .units import Units

Vec2 = Annotated[list[float], Field(min_length=2, max_length=2)]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]

# Local sketch axes for the three principal planes: (u axis, v axis). The extrude
# "+" direction is u x v, spelled out so the tool descriptions can promise it.
PRINCIPAL_PLANES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "XY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # normal +Z
    "XZ": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # normal -Y
    "YZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # normal +X
}


class PlaneSpec(BaseModel):
    """Where the sketch lives.

    Either a principal plane (``base`` + ``offset``/``origin``) or a planar face of an
    existing body (``face_id`` + ``body``). On a face, local (0, 0) is the face centre
    and "+" extrudes along the face's outward normal; the tool result reports the
    exact frame used.
    """

    base: Literal["XY", "XZ", "YZ"] = Field(
        "XY",
        description="Principal plane. XY: u=+X v=+Y normal +Z. XZ: u=+X v=+Z normal -Y. "
        "YZ: u=+Y v=+Z normal +X.",
    )
    offset: float = Field(0.0, description="Shift of the plane along its normal (working units).")
    origin: Vec3 | None = Field(
        None, description="Optional plane origin [x, y, z]; overrides offset when given."
    )
    face_id: str | None = Field(None, description="Sketch on this planar face instead.")
    body: str | None = Field(None, description="Body id or name that owns face_id.")


class Box(BaseModel):
    type: Literal["box"] = "box"
    center: Vec2
    width: float = Field(gt=0, description="Size along local u.")
    height: float = Field(gt=0, description="Size along local v.")
    angle: float = Field(0.0, description="Rotation in degrees, counter-clockwise.")


class Circle(BaseModel):
    type: Literal["circle"] = "circle"
    center: Vec2
    radius: float = Field(gt=0)


class Polygon(BaseModel):
    """Regular polygon. inner_radius is the apothem (half the across-flats size)."""

    type: Literal["polygon"] = "polygon"
    center: Vec2
    inner_radius: float = Field(gt=0, description="Apothem = across-flats / 2.")
    sides: int = Field(ge=3)
    angle: float = 0.0


class Slot(BaseModel):
    """Obround slot: two semicircles joined by straight sides."""

    type: Literal["slot"] = "slot"
    center: Vec2
    width: float = Field(gt=0, description="Overall length along local u (before rotation).")
    height: float = Field(gt=0, description="Overall width; the end radius is height / 2.")
    angle: float = 0.0


class Ellipse(BaseModel):
    type: Literal["ellipse"] = "ellipse"
    center: Vec2
    major_radius: float = Field(gt=0)
    minor_radius: float = Field(gt=0)
    angle: float = 0.0


class Polyline(BaseModel):
    """Closed straight-edged profile through the points (closed automatically)."""

    type: Literal["polyline"] = "polyline"
    points: Annotated[list[Vec2], Field(min_length=3)]


class LineTo(BaseModel):
    line_to: Vec2


class ArcTo(BaseModel):
    arc_to: Vec2
    center: Vec2
    clockwise: bool = False


class PathProfile(BaseModel):
    """Closed profile of lines and arcs, starting at ``start`` and ending back there."""

    type: Literal["path"] = "path"
    start: Vec2
    segments: Annotated[list[LineTo | ArcTo], Field(min_length=1)]


Shape = Annotated[
    Box | Circle | Polygon | Slot | Ellipse | Polyline | PathProfile,
    Field(discriminator="type"),
]


def principal_plane(spec: PlaneSpec, units: Units) -> Plane:
    u, v = PRINCIPAL_PLANES[spec.base]
    if spec.origin is not None:
        origin = units.point3d(spec.origin)
    else:
        normal = Vector3D(u).cross(Vector3D(v))
        origin = units.point3d([c * spec.offset for c in normal])
    return Plane(origin=origin, direction_x=u, direction_y=v)


def plane_on_face(origin: Point3D, normal: UnitVector3D) -> Plane:
    """A sketch plane on a planar face, with a deterministic in-plane u axis.

    u is the global axis least aligned with the normal, projected into the plane, so a
    face whose normal is +Z gets u = +X, like the XY plane.
    """
    n = [float(c) for c in normal]
    candidates = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    ref = min(candidates, key=lambda a: abs(sum(x * y for x, y in zip(a, n, strict=True))))
    dot = sum(x * y for x, y in zip(ref, n, strict=True))
    u = [r - dot * c for r, c in zip(ref, n, strict=True)]
    v = Vector3D(n).cross(Vector3D(u))
    return Plane(origin=origin, direction_x=u, direction_y=v)


def build_sketch(plane: Plane, shapes: list, units: Units) -> Sketch:
    if not shapes:
        raise ValueError("at least one shape is required")
    sketch = Sketch(plane)
    for shape in shapes:
        _add_shape(sketch, shape, units)
    return sketch


def _add_shape(sketch: Sketch, shape, units: Units) -> None:
    p = units.point2d
    d = units.distance
    match shape:
        case Box():
            sketch.box(
                p(shape.center), d(shape.width), d(shape.height), angle=units.angle(shape.angle)
            )
        case Circle():
            sketch.circle(p(shape.center), d(shape.radius))
        case Polygon():
            sketch.polygon(
                p(shape.center), d(shape.inner_radius), shape.sides, angle=units.angle(shape.angle)
            )
        case Slot():
            if shape.width <= shape.height:
                raise ValueError("slot width (overall length) must be larger than its height")
            sketch.slot(
                p(shape.center), d(shape.width), d(shape.height), angle=units.angle(shape.angle)
            )
        case Ellipse():
            if shape.minor_radius > shape.major_radius:
                raise ValueError("ellipse minor_radius must not exceed major_radius")
            sketch.ellipse(
                p(shape.center),
                d(shape.major_radius),
                d(shape.minor_radius),
                angle=units.angle(shape.angle),
            )
        case Polyline():
            pts = [list(pt) for pt in shape.points]
            if _close(pts[0], pts[-1]):
                pts = pts[:-1]
            if len(pts) < 3:
                raise ValueError("polyline needs at least 3 distinct points")
            for a, b in zip(pts, pts[1:] + pts[:1], strict=True):
                if _close(a, b):
                    raise ValueError(f"polyline has a zero-length edge at {a}")
                sketch.segment(p(a), p(b))
        case PathProfile():
            _add_path(sketch, shape, units)
        case _:
            raise ValueError(f"unknown shape {shape!r}")


def _add_path(sketch: Sketch, path: PathProfile, units: Units) -> None:
    p = units.point2d
    current = list(path.start)
    for seg in path.segments:
        if isinstance(seg, LineTo):
            end = list(seg.line_to)
            if _close(current, end):
                raise ValueError(f"path has a zero-length line at {current}")
            sketch.segment(p(current), p(end))
        else:
            end = list(seg.arc_to)
            r_start = math.dist(current, seg.center)
            r_end = math.dist(end, seg.center)
            if abs(r_start - r_end) > 1e-6 * max(1.0, r_start):
                raise ValueError(
                    f"arc from {current} to {end} is not centred on {seg.center}: "
                    f"radii {r_start:.6g} and {r_end:.6g} differ"
                )
            sketch.arc(p(current), p(end), p(seg.center), clockwise=seg.clockwise)
        current = end
    if not _close(current, path.start):
        raise ValueError(f"path must end where it starts ({path.start}); it ends at {current}")


def _close(a, b, tol: float = 1e-9) -> bool:
    return math.dist(a, b) <= tol
