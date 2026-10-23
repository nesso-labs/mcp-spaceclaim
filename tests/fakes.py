"""A tiny stand-in for the PyAnsys Geometry objects the server touches.

It models just enough (bodies with a volume, a box, faces and edges; booleans that
change volume) to test the server's orchestration: id/name resolution, change
reporting, no-op detection, error mapping. Real geometry behaviour is covered by the
live tests against a product (tests/test_live.py).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pyvista as pv
from ansys.geometry.core.math import Point3D, UnitVector3D
from ansys.geometry.core.misc import UNITS

_ids = itertools.count(1)


def new_id(prefix: str) -> str:
    return f"{prefix}{next(_ids)}"


class SurfaceType(Enum):
    SURFACETYPE_PLANE = 1
    SURFACETYPE_CYLINDER = 2


class CurveType(Enum):
    CURVETYPE_LINE = 1
    CURVETYPE_CIRCLE = 2


@dataclass
class FakeFace:
    body: FakeBody
    surface_type: SurfaceType
    center: tuple[float, float, float]  # metres
    normal_vec: tuple[float, float, float] = (0.0, 0.0, 1.0)
    area_m2: float = 1e-4
    radius_m: float | None = None
    id: str = field(default_factory=lambda: new_id("F"))

    @property
    def area(self):
        return self.area_m2 * UNITS.m**2

    def point(self, u=0.5, v=0.5):
        return Point3D(self.center)

    def normal(self, u=0.5, v=0.5):
        return UnitVector3D(self.normal_vec)

    @property
    def shape(self):
        geometry = SimpleNamespace(
            radius=None if self.radius_m is None else self.radius_m * UNITS.m,
            dir_z=UnitVector3D([0, 0, 1]),
        )
        return SimpleNamespace(geometry=geometry)

    @property
    def edges(self):
        return [e for e in self.body.edges if self.id in e.face_ids]

    def tessellate(self):
        return pv.Plane(center=self.center, direction=self.normal_vec, i_size=0.01, j_size=0.01)


@dataclass
class FakeEdge:
    body: FakeBody
    curve_type: CurveType
    start_m: tuple[float, float, float]
    end_m: tuple[float, float, float]
    face_ids: tuple[str, ...] = ()
    radius_m: float | None = None
    id: str = field(default_factory=lambda: new_id("E"))

    @property
    def length(self):
        d = sum((a - b) ** 2 for a, b in zip(self.start_m, self.end_m, strict=True)) ** 0.5
        return d * UNITS.m

    @property
    def start(self):
        return Point3D(self.start_m)

    @property
    def end(self):
        return Point3D(self.end_m)

    @property
    def shape(self):
        s, e = self.start_m, self.end_m

        def evaluate_proportion(t):
            pos = [a + (b - a) * t for a, b in zip(s, e, strict=True)]
            return SimpleNamespace(position=Point3D(pos))

        geometry = SimpleNamespace(
            radius=None if self.radius_m is None else self.radius_m * UNITS.m,
            origin=Point3D(s),
        )
        return SimpleNamespace(geometry=geometry, evaluate_proportion=evaluate_proportion)


class FakeBody:
    """An axis-aligned block; booleans only change volume and face counts."""

    def __init__(self, design: FakeDesign, name: str, lo, hi, *, inert: bool = False):
        self.design = design
        self.id = new_id("B")
        self.name = name
        self.lo, self.hi = list(lo), list(hi)
        self.volume_m3 = _box_volume(self.lo, self.hi)
        self.is_surface = False
        self.inert = inert  # booleans "succeed" without changing anything
        self.calls: list[tuple] = []
        self._build_topology()

    def _build_topology(self):
        lo, hi = self.lo, self.hi
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        top = FakeFace(self, SurfaceType.SURFACETYPE_PLANE, (cx, cy, hi[2]), (0, 0, 1))
        bottom = FakeFace(self, SurfaceType.SURFACETYPE_PLANE, (cx, cy, lo[2]), (0, 0, -1))
        side = FakeFace(
            self, SurfaceType.SURFACETYPE_PLANE, (hi[0], cy, (lo[2] + hi[2]) / 2), (1, 0, 0)
        )
        self.faces = [top, bottom, side]
        self.edges = [
            FakeEdge(
                self,
                CurveType.CURVETYPE_LINE,
                (lo[0], lo[1], hi[2]),
                (hi[0], lo[1], hi[2]),
                (top.id,),
            ),
            FakeEdge(
                self,
                CurveType.CURVETYPE_LINE,
                (hi[0], lo[1], lo[2]),
                (hi[0], lo[1], hi[2]),
                (side.id,),
            ),
        ]

    @property
    def volume(self):
        return self.volume_m3 * UNITS.m**3

    @property
    def bounding_box(self):
        return SimpleNamespace(min_corner=Point3D(self.lo), max_corner=Point3D(self.hi))

    @property
    def parent_component(self):
        return self.design

    def set_name(self, name):
        self.name = name

    def copy(self, parent, name):
        clone = FakeBody(self.design, name, self.lo, self.hi)
        clone.volume_m3 = self.volume_m3
        self.design.bodies.append(clone)
        return clone

    def translate(self, direction, distance):
        d = distance.value.m_as(UNITS.m)
        for i, c in enumerate(direction):
            self.lo[i] += float(c) * d
            self.hi[i] += float(c) * d

    def rotate(self, origin, axis, angle):
        self.calls.append(("rotate", angle.value.m_as(UNITS.degree)))

    def mirror(self, plane):
        self.calls.append(("mirror",))

    def subtract(self, other, keep_other=False):
        others = other if isinstance(other, list) else [other]
        self.calls.append(("subtract", [o.id for o in others], keep_other))
        if not self.inert:
            self.volume_m3 -= sum(o.volume_m3 for o in others) / 2
            self.faces.append(
                FakeFace(self, SurfaceType.SURFACETYPE_CYLINDER, (0, 0, 0), radius_m=0.004)
            )
        if not keep_other:
            for o in others:
                self.design.delete_body(o)

    def unite(self, other, keep_other=False):
        others = other if isinstance(other, list) else [other]
        self.calls.append(("unite", [o.id for o in others]))
        if not self.inert:
            self.volume_m3 += sum(o.volume_m3 for o in others)
        for o in others:
            self.design.delete_body(o)

    def intersect(self, other, keep_other=False):
        others = other if isinstance(other, list) else [other]
        if not self.inert:
            self.volume_m3 = min([self.volume_m3, *(o.volume_m3 for o in others)])
        if not keep_other:
            for o in others:
                self.design.delete_body(o)

    def shell_body(self, offset):
        self.volume_m3 *= 0.5
        return True

    def remove_faces(self, faces, offset):
        self.volume_m3 *= 0.4
        self.faces = [f for f in self.faces if f not in faces]
        return True

    def get_collision(self, other):
        return SimpleNamespace(name="INTERSECT")

    def tessellate(self, merge=True):
        lo, hi = self.lo, self.hi
        return pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])).triangulate()


class FakeDesign:
    def __init__(self, name: str):
        self.name = name
        self.is_active = True
        self.bodies: list[FakeBody] = []
        self.calls: list[tuple] = []
        self.named_selections: list = []

    def get_all_bodies(self):
        return list(self.bodies)

    def add_block(self, name, lo_mm, hi_mm, **kw) -> FakeBody:
        body = FakeBody(self, name, [v / 1000 for v in lo_mm], [v / 1000 for v in hi_mm], **kw)
        self.bodies.append(body)
        return body

    def extrude_sketch(self, name, sketch, distance, direction="+", cut=False):
        d = distance.value.m_as(UNITS.m)
        z0 = float(sketch.plane.origin[2])
        self.calls.append(("extrude", name, d, direction, sketch))
        lo, hi = [0.0, 0.0, z0], [0.01, 0.01, z0 + d]
        body = FakeBody(self, name, lo, hi)
        self.bodies.append(body)
        return body

    def revolve_sketch(self, name, sketch, axis, angle, rotation_origin):
        self.calls.append(("revolve", name, angle.value.m_as(UNITS.degree)))
        body = FakeBody(self, name, [0, 0, 0], [0.01, 0.01, 0.01])
        self.bodies.append(body)
        return body

    def create_sphere(self, name, center, radius):
        r = radius.value.m_as(UNITS.m)
        c = [float(v) for v in center]
        body = FakeBody(self, name, [v - r for v in c], [v + r for v in c])
        self.bodies.append(body)
        return body

    def delete_body(self, body):
        self.bodies = [b for b in self.bodies if b.id != body.id]

    def download(self, path, fmt):
        Path(path).write_bytes(b"fake " + fmt.name.encode())
        self.calls.append(("download", str(path), fmt.name))

    def create_named_selection(self, name, bodies=None, faces=None, edges=None):
        ns = SimpleNamespace(id=new_id("NS"), name=name, bodies=bodies, faces=faces, edges=edges)
        self.named_selections.append(ns)
        return ns


class FakeCommands:
    def __init__(self):
        self.calls: list[tuple] = []
        self.succeed = True

    def fillet(self, edges, radius):
        self.calls.append(("fillet", [e.id for e in edges], radius.value.m_as(UNITS.mm)))
        if self.succeed:
            edges[0].body.volume_m3 *= 0.99
        return self.succeed

    def chamfer(self, edges, distance):
        self.calls.append(("chamfer", [e.id for e in edges]))
        if self.succeed:
            edges[0].body.volume_m3 *= 0.98
        return self.succeed

    def extrude_faces(
        self,
        faces,
        distance,
        direction=None,
        extrude_type=None,
        offset_mode=None,
        pull_symmetric=False,
        copy=False,
        force_do_as_extrude=False,
    ):
        kind = extrude_type.name if extrude_type is not None else "ADD"
        self.calls.append(("extrude_faces", [f.id for f in faces], kind, force_do_as_extrude))
        body = faces[0].body
        body.hi[2] += distance.value.m_as(UNITS.m)  # signed, like the real call
        body.volume_m3 = _box_volume(body.lo, body.hi)
        return []


class FakeModeler:
    def __init__(self, version=(25, 2, 0)):
        self.client = SimpleNamespace(
            backend_version=SimpleNamespace(major=version[0], minor=version[1], patch=version[2]),
            backend_type=SimpleNamespace(name="SPACECLAIM"),
            healthy=True,
        )
        self.design: FakeDesign | None = None
        self.geometry_commands = FakeCommands()
        self.measurement_tools = SimpleNamespace(
            min_distance_between_objects=lambda a, b: SimpleNamespace(distance=0.0042 * UNITS.m)
        )
        self.closed = False
        self.scripts: list[tuple] = []

    def create_design(self, name):
        self.design = FakeDesign(name)
        return self.design

    def read_existing_design(self):
        if self.design is None:
            raise RuntimeError("no design")
        return self.design

    def get_active_design(self, sync_with_backend=True):
        return self.design

    def open_file(self, path):
        self.design = FakeDesign(Path(path).stem)
        self.design.add_block("Imported", [0, 0, 0], [10, 10, 10])
        return self.design

    def run_script_file(self, path, script_args=None, import_design=False):
        self.scripts.append((Path(path).read_text(), script_args))
        self.design.add_block("FromScript", [0, 0, 0], [1, 1, 1])
        return {"ok": "1"}, self.design if import_design else None

    def close(self, close_design=True):
        self.closed = True


def _box_volume(lo, hi) -> float:
    return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
