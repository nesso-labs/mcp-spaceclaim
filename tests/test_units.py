import pytest
from ansys.geometry.core.misc import UNITS

from mcp_spaceclaim.units import Units, direction


def test_round_trip_in_millimetres():
    u = Units("mm")
    assert u.distance(12.5).value.m_as(UNITS.m) == pytest.approx(0.0125)
    assert u.length_out(u.distance(12.5)) == 12.5
    assert u.point_out(u.point3d([1, 2, 3])) == [1.0, 2.0, 3.0]
    assert u.volume_out((0.01 * UNITS.m) ** 3) == pytest.approx(1000.0)
    assert u.area_out((0.01 * UNITS.m) ** 2) == pytest.approx(100.0)
    assert u.per_metre == pytest.approx(1000.0)


def test_inches():
    u = Units("in")
    assert u.length_out(1 * UNITS.m) == pytest.approx(39.370079, rel=1e-6)


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError, match="unsupported unit"):
        Units("furlong")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("X", (1, 0, 0)), ("+y", (0, 1, 0)), ("-Z", (0, 0, -1)), ([0, 0, 5], (0, 0, 1))],
)
def test_direction(spec, expected):
    assert tuple(float(c) for c in direction(spec)) == pytest.approx(expected)


def test_direction_errors():
    with pytest.raises(ValueError, match="unknown axis"):
        direction("W")
    with pytest.raises(ValueError, match="must not be zero"):
        direction([0, 0, 0])
    with pytest.raises(ValueError, match="needs 3 coordinates"):
        direction([1, 0])
