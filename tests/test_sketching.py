"""Sketches are built client-side, so these exercise the real PyAnsys Geometry objects."""

import pytest
from pydantic import TypeAdapter, ValidationError

from mcp_spaceclaim.sketching import PlaneSpec, Shape, build_sketch, plane_on_face, principal_plane
from mcp_spaceclaim.units import Units, direction

U = Units("mm")
SHAPES = TypeAdapter(list[Shape])


def shapes(*items):
    return SHAPES.validate_python(list(items))


def vec(v):
    return [round(float(c), 9) for c in v]


@pytest.mark.parametrize(
    ("base", "normal"),
    [("XY", [0, 0, 1]), ("XZ", [0, -1, 0]), ("YZ", [1, 0, 0])],
)
def test_principal_planes_match_the_documented_normals(base, normal):
    plane = principal_plane(PlaneSpec(base=base, offset=7), U)
    assert vec(plane.direction_z) == normal
    assert U.point_out(plane.origin) == [7.0 * c for c in normal]


def test_origin_overrides_offset():
    plane = principal_plane(PlaneSpec(base="XY", offset=5, origin=[1, 2, 3]), U)
    assert U.point_out(plane.origin) == [1.0, 2.0, 3.0]


def test_plane_on_face_has_a_deterministic_u_axis():
    top = plane_on_face(U.point3d([0, 0, 10]), direction("Z"))
    assert vec(top.direction_x) == [1, 0, 0]
    assert vec(top.direction_z) == [0, 0, 1]
    side = plane_on_face(U.point3d([5, 0, 0]), direction("X"))
    assert vec(side.direction_z) == [1, 0, 0]
    assert (
        abs(sum(float(a) * float(b) for a, b in zip(side.direction_x, [1, 0, 0], strict=True)))
        < 1e-12
    )


def test_every_shape_type_builds():
    sketch = build_sketch(
        principal_plane(PlaneSpec(), U),
        shapes(
            {"type": "box", "center": [0, 0], "width": 80, "height": 50, "angle": 15},
            {"type": "circle", "center": [10, 5], "radius": 4},
            {"type": "polygon", "center": [0, 0], "inner_radius": 5, "sides": 6},
            {"type": "slot", "center": [0, 0], "width": 20, "height": 6},
            {"type": "ellipse", "center": [0, 0], "major_radius": 5, "minor_radius": 3},
            {"type": "polyline", "points": [[0, 0], [80, 0], [80, 5], [5, 5], [5, 50], [0, 50]]},
            {
                "type": "path",
                "start": [0, 0],
                "segments": [
                    {"line_to": [10, 0]},
                    {"arc_to": [10, 10], "center": [10, 5]},
                    {"line_to": [0, 10]},
                    {"line_to": [0, 0]},
                ],
            },
        ),
        U,
    )
    assert len(sketch.faces) == 5  # box, circle, polygon, slot, ellipse
    assert len(sketch.edges) == 10  # 6 polyline segments + 4 path segments


def test_polyline_accepts_an_explicitly_closed_point_list():
    sketch = build_sketch(
        principal_plane(PlaneSpec(), U),
        shapes({"type": "polyline", "points": [[0, 0], [10, 0], [10, 10], [0, 0]]}),
        U,
    )
    assert len(sketch.edges) == 3


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ({"type": "slot", "center": [0, 0], "width": 5, "height": 6}, "larger than its height"),
        (
            {"type": "ellipse", "center": [0, 0], "major_radius": 2, "minor_radius": 3},
            "minor_radius",
        ),
        ({"type": "polyline", "points": [[0, 0], [1, 0], [1, 0]]}, "zero-length edge"),
        ({"type": "polyline", "points": [[0, 0], [1, 0], [0, 0]]}, "at least 3 distinct"),
        (
            {"type": "path", "start": [0, 0], "segments": [{"line_to": [5, 0]}]},
            "must end where it starts",
        ),
        (
            {
                "type": "path",
                "start": [0, 0],
                "segments": [{"arc_to": [4, 0], "center": [1, 0]}, {"line_to": [0, 0]}],
            },
            "not centred",
        ),
    ],
)
def test_invalid_profiles_explain_themselves(shape, message):
    with pytest.raises(ValueError, match=message):
        build_sketch(principal_plane(PlaneSpec(), U), shapes(shape), U)


def test_schema_rejects_bad_input():
    with pytest.raises(ValidationError):
        shapes({"type": "circle", "center": [0, 0], "radius": -1})
    with pytest.raises(ValidationError):
        shapes({"type": "box", "center": [0, 0, 0], "width": 1, "height": 1})
    with pytest.raises(ValidationError):
        shapes({"type": "hexagon", "center": [0, 0]})


def test_empty_sketch_is_rejected():
    with pytest.raises(ValueError, match="at least one shape"):
        build_sketch(principal_plane(PlaneSpec(), U), [], U)
