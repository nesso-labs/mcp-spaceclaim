"""The MCP server: tool definitions on top of one shared SpaceClaim session."""

import functools
import logging
import math
import os
import tempfile
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, ValidationError

from . import __version__, render
from .session import Session, SessionError
from .sketching import PlaneSpec, Shape, Vec3, build_sketch, plane_on_face, principal_plane
from .units import Units
from .units import direction as as_direction

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
Drive Ansys SpaceClaim / Discovery (2025 R2 or newer) through PyAnsys Geometry.

Workflow:
1. spaceclaim_connect (launches SpaceClaim with its API server, or attaches to one).
2. new_design or open_file.
3. Build: sketch_extrude / revolve_sketch / create_box / create_cylinder / create_sphere,
   then boolean, move_body, pattern_body, fillet_edges, chamfer_edges, pull_faces, shell_body.
4. Check: list_bodies, describe_body, select_faces / select_edges, measure_distance and
   render_view (look at the picture before telling the user it is done).
5. save_design (.scdocx, .step, .x_t, .igs, .stl ...).

Conventions: every length is in the server's working unit (see spaceclaim_status, default
millimetres); angles are degrees. Bodies are referenced by id (preferred) or unique name.
Face and edge ids change after topology edits: re-select them instead of reusing old ids.
Every modelling tool reports which bodies were created, deleted or modified, and fails
with no_geometry_change if the backend accepted the call but nothing changed.
For anything the typed tools cannot do, run_script runs a SpaceClaim Python script.
"""

RO = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
EDIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)


class Rotation(BaseModel):
    origin: Vec3 = Field(description="A point on the rotation axis [x, y, z].")
    axis: str | Vec3 = Field("Z", description='Axis direction: "X", "Y", "Z", "-Z" or [x, y, z].')
    angle: float = Field(description="Degrees, right-hand rule about the axis.")


class MirrorPlane(BaseModel):
    origin: Vec3 = Field([0.0, 0.0, 0.0], description="A point on the mirror plane.")
    normal: str | Vec3 = Field("X", description="Plane normal, e.g. X mirrors across x = origin.x")


class Config(BaseModel):
    units: str = "mm"
    product: Literal["spaceclaim", "discovery", "geometry-service"] = "spaceclaim"
    version: int | None = 252
    hidden: bool = False
    host: str = "localhost"
    port: int = 50051
    transport_mode: str | None = None
    allow_scripts: bool = True
    output_dir: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ
        version = env.get("SPACECLAIM_MCP_VERSION", "252")
        return cls(
            units=env.get("SPACECLAIM_MCP_UNITS", "mm"),
            product=env.get("SPACECLAIM_MCP_PRODUCT", "spaceclaim"),
            version=int(version) if version.strip() else None,
            hidden=_truthy(env.get("SPACECLAIM_MCP_HIDDEN", "0")),
            host=env.get("SPACECLAIM_MCP_HOST", "localhost"),
            port=int(env.get("SPACECLAIM_MCP_PORT", "50051")),
            transport_mode=env.get("SPACECLAIM_MCP_TRANSPORT_MODE") or None,
            allow_scripts=_truthy(env.get("SPACECLAIM_MCP_ALLOW_SCRIPTS", "1")),
            output_dir=env.get("SPACECLAIM_MCP_OUTPUT_DIR") or None,
        )


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def tool_errors(fn: Callable) -> Callable:
    """Turn every failure into a ToolError the model can read.

    MCPServer hides the message of unexpected exceptions ("Error executing tool"), which
    leaves the model guessing; a SpaceClaim error message is usually the fix.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except (SessionError, ValueError, KeyError, ValidationError) as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            log.exception("tool %s failed", fn.__name__)
            raise ToolError(f"{type(exc).__name__} from SpaceClaim/PyAnsys: {exc}") from exc

    return wrapper


def build_server(config: Config | None = None, session: Session | None = None) -> MCPServer:
    config = config or Config.from_env()
    session = session or Session(units=Units(config.units))
    units = session.units
    out_dir = Path(config.output_dir or Path.home() / "Documents" / "mcp-spaceclaim")

    mcp = MCPServer(
        "spaceclaim",
        title="Ansys SpaceClaim",
        instructions=INSTRUCTIONS,
        version=__version__,
        website_url="https://github.com/nesso-labs/mcp-spaceclaim",
    )

    def tool(annotations: ToolAnnotations):
        def register(fn):
            mcp.add_tool(tool_errors(fn), annotations=annotations)
            return fn

        return register

    def track(
        op: Callable[[], Any], focus: list[str] | None, expect_change: bool = True
    ) -> dict[str, Any]:
        """Run a modelling call under the lock and report what it changed.

        ``focus`` are the bodies the call should touch (see Session.snapshot). With
        ``expect_change`` a call that leaves every body as it was is an error: PyAnsys
        returns normally from some backend no-ops, and the model must not build on them.
        """
        with session.lock:
            before = session.snapshot(focus)
            try:
                extra = op() or {}
            except Exception:
                _discard_new_bodies(session, set(before))
                raise
            after = session.snapshot(focus)
            created = [i for i in after if i not in before]
            if created:
                details = session.snapshot(created)
                after.update({i: details[i] for i in created if i in details})
            changes = session.diff(before, after)
            if expect_change and not any(changes.values()):
                raise ToolError(
                    "no_geometry_change: the backend accepted the call but no body changed. "
                    "Check ids and dimensions (a cut that misses the body, a fillet larger "
                    "than the faces allow) and render_view to see the current state."
                )
            return {**extra, "changes": changes}

    # ---------------------------------------------------------------------------------
    # session
    # ---------------------------------------------------------------------------------

    @tool(EDIT)
    def spaceclaim_connect(
        mode: Annotated[
            Literal["launch", "attach"],
            Field(
                description="launch: start a new product instance. attach: connect to one "
                "already running with the ApiServer add-in."
            ),
        ] = "launch",
        product: Annotated[
            Literal["spaceclaim", "discovery", "geometry-service"] | None,
            Field(description="Product to launch (default from SPACECLAIM_MCP_PRODUCT)."),
        ] = None,
        version: Annotated[
            int | None, Field(description="Ansys release, e.g. 252 for 2025 R2, 261 for 2026 R1.")
        ] = None,
        hidden: Annotated[bool | None, Field(description="Launch without showing the UI.")] = None,
        host: Annotated[str | None, Field(description="attach: API server host.")] = None,
        port: Annotated[int | None, Field(description="attach: API server port.")] = None,
        transport_mode: Annotated[
            Literal["wnua", "uds", "mtls", "insecure"] | None,
            Field(description="attach: gRPC transport; wnua is the Windows default."),
        ] = None,
        timeout: Annotated[int, Field(description="Seconds to wait for a launch.")] = 300,
    ) -> dict[str, Any]:
        """Start SpaceClaim/Discovery (or attach to a running one) and make it the target.

        Launching starts the product with its API server add-in and takes 30-120 s.
        Reconnecting closes the previous connection first; a product this server launched
        exits with it.
        """
        with session.lock:
            if mode == "launch":
                session.launch(
                    product=product or config.product,
                    version=version if version is not None else config.version,
                    hidden=config.hidden if hidden is None else hidden,
                    timeout=timeout,
                )
            else:
                session.attach(
                    host=host or config.host,
                    port=port or config.port,
                    transport_mode=transport_mode or config.transport_mode,
                )
            return session.status()

    @tool(RO)
    def spaceclaim_status() -> dict[str, Any]:
        """Connection state, backend version, working unit and the open design."""
        with session.lock:
            return session.status()

    @tool(DESTRUCTIVE)
    def spaceclaim_disconnect() -> dict[str, Any]:
        """Close the connection. A product launched by this server exits (unsaved work is
        lost); an attached product keeps running."""
        with session.lock:
            session.disconnect()
            return session.status()

    # ---------------------------------------------------------------------------------
    # documents
    # ---------------------------------------------------------------------------------

    @tool(DESTRUCTIVE)
    def new_design(name: str = "Design1") -> dict[str, Any]:
        """Create an empty design. This closes the currently open design in the product;
        save_design first if it has unsaved work."""
        with session.lock:
            modeler = session.require_modeler()
            previous = session.design.name if session.design is not None else None
            session.design = modeler.create_design(name)
            return {"design": session.design.name, "closed_previous": previous}

    @tool(DESTRUCTIVE)
    def open_file(
        path: Annotated[
            str,
            Field(
                description="CAD file on this machine: .scdocx, .step/.stp, "
                ".x_t, .igs, .sat, native formats SpaceClaim can import."
            ),
        ],
    ) -> dict[str, Any]:
        """Open a CAD file as the active design (closes the current design)."""
        file = Path(path).expanduser()
        if not file.is_file():
            raise ToolError(f"file not found: {file}")
        with session.lock:
            modeler = session.require_modeler()
            # Uploading is for remote backends; for multi-file formats it also sends the
            # whole folder and fails on any subfolder.
            session.design = modeler.open_file(str(file), upload_to_server=not session.is_local)
            return {"design": session.design.name, "bodies": _bodies_summary(session)}

    @tool(EDIT)
    def save_design(
        path: Annotated[
            str,
            Field(
                description="Target file; the extension picks the format: .scdocx, .step/.stp, "
                ".x_t/.x_b (Parasolid), .igs, .fmd, .pmdb, or .stl (tessellated). "
                "A relative path is placed under the server's output folder."
            ),
        ],
    ) -> dict[str, Any]:
        """Save or export the active design. An existing file is replaced."""
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = out_dir / target
        ext = target.suffix.lower().lstrip(".")
        fmt = None
        if ext != "stl":
            from ansys.geometry.core.designer.design import DESIGN_FILE_FORMAT_EXTENSIONS

            fmt = next(
                (
                    f
                    for f, exts in DESIGN_FILE_FORMAT_EXTENSIONS.items()
                    if ext in exts and f.name != "INVALID"
                ),
                None,
            )
            if fmt is None:
                raise ToolError(f"unsupported extension .{ext}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write next to the target and move it into place, so a stale file from an
        # earlier save can never pass for this one.
        partial = target.with_name(f"{target.stem}.mcp-partial{target.suffix}")
        partial.unlink(missing_ok=True)
        via = "pyansys"
        with session.lock:
            design = session.require_design()
            if fmt is None:
                _export_stl(session, partial)
                via = "tessellation"
            else:
                error: Exception | None = None
                try:
                    design.download(partial, fmt)
                except Exception as exc:
                    error = exc
                if not _written(partial) and ext in PRODUCT_SAVE and session.has_product_ui:
                    # PyAnsys' STEP/IGES export needs 2027 R1; on older releases save
                    # through the product's own Save As instead.
                    if not session.is_local:
                        raise ToolError(
                            f"STEP/IGES export is not available through the API on this "
                            f"release, and the remote product cannot write {partial} here; "
                            "save as .scdocx or .x_t, or run the server on the CAD machine"
                        )
                    _save_in_product(session, partial)
                    via = "product save"
                if not _written(partial):
                    partial.unlink(missing_ok=True)
                    detail = f": {error}" if error else ""
                    raise ToolError(f"export to .{ext} produced no file{detail}")
        partial.replace(target)
        return {"path": str(target), "bytes": target.stat().st_size, "via": via}

    # ---------------------------------------------------------------------------------
    # inspection
    # ---------------------------------------------------------------------------------

    @tool(RO)
    def list_bodies(
        refresh: Annotated[
            bool, Field(description="Re-read the design first (after edits made in the UI).")
        ] = False,
    ) -> dict[str, Any]:
        """All bodies in the design with volume, bounding box and face/edge counts."""
        with session.lock:
            if refresh:
                session.refresh_design()
            return {"units": units.name, "bodies": _bodies_summary(session)}

    @tool(RO)
    def describe_body(
        body: Annotated[str, Field(description="Body id or unique name.")],
        include_faces: bool = True,
        include_edges: bool = True,
        limit: Annotated[int, Field(ge=1, le=2000)] = 300,
    ) -> dict[str, Any]:
        """Faces (type, area, normal, radius, centre) and edges (type, length, ends, radius)
        of one body, with the ids that the face/edge tools take."""
        with session.lock:
            b = session.body(body)
            info = _body_summary(session, b)
            if include_faces:
                faces = list(b.faces)
                info["faces"] = [_face_info(units, f) for f in faces[:limit]]
                info["faces_truncated"] = len(faces) > limit
            if include_edges:
                edges = list(b.edges)
                info["edges"] = [_edge_info(units, e) for e in edges[:limit]]
                info["edges_truncated"] = len(edges) > limit
            return info

    @tool(RO)
    def select_faces(
        body: Annotated[str, Field(description="Body id or unique name.")],
        surface_type: Annotated[
            Literal["plane", "cylinder", "cone", "torus", "sphere", "nurbs", "other"] | None,
            Field(description="Keep only this surface type."),
        ] = None,
        normal: Annotated[
            str | Vec3 | None,
            Field(
                description='Planar faces whose outward normal points this way ("Z", "-X" or '
                "[x, y, z]), within normal_tolerance degrees."
            ),
        ] = None,
        normal_tolerance: float = 5.0,
        radius: Annotated[float | None, Field(description="Cylinder/cone/sphere radius.")] = None,
        radius_tolerance: float = 1e-3,
        min_area: float | None = None,
        max_area: float | None = None,
        extreme: Annotated[
            Literal["max_x", "min_x", "max_y", "min_y", "max_z", "min_z"] | None,
            Field(
                description="Of the matches, keep those whose centre is furthest along this "
                "direction, e.g. max_z = the top face(s)."
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Find faces by geometry instead of guessing ids (top face, all Ø8 holes ...)."""
        with session.lock:
            b = session.body(body)
            want_normal = as_direction(normal) if normal is not None else None
            cos_tol = math.cos(math.radians(normal_tolerance))
            matches = []
            for f in b.faces:
                info = _face_info(units, f)
                if surface_type and info["type"] != surface_type:
                    continue
                if want_normal is not None:
                    n = info.get("normal")
                    if n is None or _dot(n, want_normal) < cos_tol:
                        continue
                if radius is not None and (
                    info.get("radius") is None or abs(info["radius"] - radius) > radius_tolerance
                ):
                    continue
                if min_area is not None and info["area"] < min_area:
                    continue
                if max_area is not None and info["area"] > max_area:
                    continue
                matches.append(info)
            matches = _keep_extreme(matches, extreme)
            return {"body": b.id, "count": len(matches), "faces": matches}

    @tool(RO)
    def select_edges(
        body: Annotated[str, Field(description="Body id or unique name.")],
        curve_type: Annotated[
            Literal["line", "circle", "ellipse", "nurbs", "other"] | None,
            Field(description="Keep only this curve type."),
        ] = None,
        parallel_to: Annotated[
            str | Vec3 | None,
            Field(description='Straight edges parallel to this axis ("Z" or [x, y, z]).'),
        ] = None,
        radius: Annotated[float | None, Field(description="Circular edges of this radius.")] = None,
        radius_tolerance: float = 1e-3,
        min_length: float | None = None,
        max_length: float | None = None,
        on_faces: Annotated[
            list[str] | None, Field(description="Only edges bounding one of these face ids.")
        ] = None,
        extreme: Annotated[
            Literal["max_x", "min_x", "max_y", "min_y", "max_z", "min_z"] | None,
            Field(description="Of the matches, keep those furthest along this direction."),
        ] = None,
    ) -> dict[str, Any]:
        """Find edges by geometry (vertical edges, edges of the top face, Ø8 hole rims ...)."""
        with session.lock:
            b = session.body(body)
            allowed: set[str] | None = None
            if on_faces:
                allowed = {e.id for f in session.faces(b.id, on_faces) for e in f.edges}
            axis = as_direction(parallel_to) if parallel_to is not None else None
            matches = []
            for e in b.edges:
                if allowed is not None and e.id not in allowed:
                    continue
                info = _edge_info(units, e)
                if curve_type and info["type"] != curve_type:
                    continue
                if axis is not None:
                    if info["type"] != "line":
                        continue
                    d = [q - p for p, q in zip(info["start"], info["end"], strict=True)]
                    length = math.sqrt(sum(c * c for c in d)) or 1.0
                    dot = sum(c * a for c, a in zip(d, axis, strict=True)) / length
                    if abs(dot) < math.cos(math.radians(1.0)):
                        continue
                if radius is not None and (
                    info.get("radius") is None or abs(info["radius"] - radius) > radius_tolerance
                ):
                    continue
                if min_length is not None and info["length"] < min_length:
                    continue
                if max_length is not None and info["length"] > max_length:
                    continue
                matches.append(info)
            matches = _keep_extreme(matches, extreme)
            return {"body": b.id, "count": len(matches), "edges": matches}

    @tool(RO)
    def measure_distance(
        body_a: Annotated[str, Field(description="Body id or unique name.")],
        body_b: Annotated[str, Field(description="Body id or unique name.")],
    ) -> dict[str, Any]:
        """Minimum distance between two bodies and how they collide (none/touch/intersect/
        contained)."""
        with session.lock:
            a, b = session.body(body_a), session.body(body_b)
            gap = session.require_modeler().measurement_tools.min_distance_between_objects(a, b)
            collision = a.get_collision(b)
            return {
                "distance": units.length_out(gap.distance),
                "collision": collision.name.lower(),
                "units": units.name,
            }

    @tool(RO)
    def render_view(
        views: Annotated[
            list[Literal["iso", "front", "top", "right", "back", "left", "bottom"]],
            Field(description="One view, or up to four for a 2x2 sheet."),
        ] = ["iso"],  # noqa: B006 - pydantic copies defaults
        bodies: Annotated[
            list[str] | None, Field(description="Only these bodies (default: all).")
        ] = None,
        highlight_body: Annotated[
            str | None, Field(description="Body owning highlight_faces / highlight_edges.")
        ] = None,
        highlight_faces: list[str] | None = None,
        highlight_edges: list[str] | None = None,
        size: Annotated[int, Field(ge=300, le=1600)] = 900,
    ) -> list:
        """A shaded picture of the model. Use it after every few edits and before reporting
        a result; highlight faces/edges to confirm a selection before filleting or cutting."""
        if len(views) > 4:
            raise ToolError("at most four views per image")
        with session.lock:
            factor = units.per_metre  # tessellation comes back in metres
            targets = [session.body(r) for r in bodies] if bodies else session.bodies()
            scene = render.Scene(unit=units.name)
            for b in targets:
                scene.meshes.append(
                    render.mesh_from_polydata(b.name, b.tessellate(merge=True), factor)
                )
            if highlight_faces or highlight_edges:
                if not highlight_body:
                    raise ToolError("highlight_body is required with highlight_faces/edges")
                for f in session.faces(highlight_body, highlight_faces or []):
                    m = render.mesh_from_polydata(f"face {f.id}", f.tessellate(), factor)
                    m.highlight = True
                    scene.meshes.append(m)
                for e in session.edges(highlight_body, highlight_edges or []):
                    scene.lines.append(_sample_edge(e, factor))
            png = render.render(scene, list(views), size=size)
        names = ", ".join(b.name for b in targets)
        return [Image(data=png, format="png"), f"{len(targets)} bodies ({names}); views: {views}"]

    # ---------------------------------------------------------------------------------
    # modelling
    # ---------------------------------------------------------------------------------

    def _plane(plane: PlaneSpec):
        with session.lock:  # reads the face from the backend
            return _plane_unlocked(plane)

    def _plane_unlocked(plane: PlaneSpec):
        if plane.face_id:
            if not plane.body:
                raise ToolError("plane.body is required with plane.face_id")
            face = session.faces(plane.body, [plane.face_id])[0]
            if _surface_name(face) != "plane":
                raise ToolError(f"face {face.id} is not planar; sketch on a planar face")
            return plane_on_face(face.point(), face.normal())
        return principal_plane(plane, units)

    def _frame(pl) -> dict[str, Any]:
        return {
            "origin": units.point_out(pl.origin),
            "u": [round(float(c), 6) for c in pl.direction_x],
            "v": [round(float(c), 6) for c in pl.direction_y],
            "normal": [round(float(c), 6) for c in pl.direction_z],
        }

    def _place(name, make_body, operation, targets):
        """Create a tool body, then keep it, unite it into targets, or cut it from them."""
        design = session.require_design()
        target_bodies = [session.body(t) for t in targets or []]
        if operation != "new" and not target_bodies:
            raise ToolError(f"operation={operation!r} needs targets (body ids or names)")
        new = make_body(design)
        if new is None:
            raise ToolError(
                "the backend did not create a body (open or self-intersecting profile?)"
            )
        if operation == "new":
            return {"body": {"id": new.id, "name": new.name}}
        # Booleans re-read the design, so look every body up again by id before each call
        # instead of reusing objects that may belong to the previous tree.
        ids = [t.id for t in target_bodies]
        # A boolean with a body it does not touch "succeeds" by deleting the tool and
        # leaving the target as it was, so refuse those up front (track() then removes
        # the tool body we just made).
        _require_contact(session.body(new.id), [session.body(i) for i in ids], name)
        if operation == "add":
            keep = session.body(ids[0])
            keep.unite([session.body(new.id), *[session.body(i) for i in ids[1:]]])
            return {"body": {"id": keep.id, "name": keep.name}}
        for n, tid in enumerate(ids):
            session.body(tid).subtract(session.body(new.id), keep_other=n < len(ids) - 1)
        return {"cut_from": ids}

    Operation = Annotated[
        Literal["new", "add", "cut"],
        Field(
            description="new: separate body. add: unite into targets[0] (with any other "
            "targets). cut: subtract from every target."
        ),
    ]
    Targets = Annotated[list[str] | None, Field(description="Body ids/names for add or cut.")]

    @tool(EDIT)
    def sketch_extrude(
        name: str,
        shapes: Annotated[
            list[Shape],
            Field(
                description="Closed profiles in sketch coordinates (u, v). Use one outer "
                "profile per body; for holes or pockets run a second call with "
                "operation='cut'."
            ),
        ],
        distance: Annotated[float, Field(gt=0)],
        plane: PlaneSpec = PlaneSpec(),  # noqa: B008 - pydantic copies defaults
        direction: Annotated[
            Literal["+", "-", "both"],
            Field(
                description="Along the plane normal, against it, or symmetric about the "
                "plane (distance is then the total)."
            ),
        ] = "+",
        operation: Operation = "new",
        targets: Targets = None,
    ) -> dict[str, Any]:
        """Sketch 2D shapes on a plane or planar face and extrude them into a solid."""
        pl = _plane(plane)

        def make(design):
            p, sign = pl, direction
            if direction == "both":
                p, sign = _offset_plane(pl, -distance / 2, units), "+"
            sketch = build_sketch(p, shapes, units)
            return design.extrude_sketch(name, sketch, units.distance(distance), direction=sign)

        result = track(lambda: _place(name, make, operation, targets), _focus(session, targets))
        return {"plane": _frame(pl), **result}

    @tool(EDIT)
    def revolve_sketch(
        name: str,
        shapes: Annotated[list[Shape], Field(description="Closed profiles to revolve.")],
        axis_origin: Annotated[Vec3, Field(description="A point on the axis [x, y, z].")],
        axis: Annotated[str | Vec3, Field(description='Axis direction ("Z" or [x, y, z]).')],
        angle: Annotated[float, Field(gt=0, le=360)] = 360.0,
        plane: PlaneSpec = PlaneSpec(base="XZ"),  # noqa: B008
        operation: Operation = "new",
        targets: Targets = None,
    ) -> dict[str, Any]:
        """Revolve sketch profiles about an axis (shafts, pulleys, turned parts). The axis
        must lie in the sketch plane and must not cross the profile."""
        pl = _plane(plane)

        def make(design):
            sketch = build_sketch(pl, shapes, units)
            return design.revolve_sketch(
                name,
                sketch,
                axis=as_direction(axis),
                angle=units.angle(angle),
                rotation_origin=units.point3d(axis_origin),
            )

        result = track(lambda: _place(name, make, operation, targets), _focus(session, targets))
        return {"plane": _frame(pl), **result}

    @tool(EDIT)
    def create_box(
        name: str,
        corner_min: Annotated[Vec3, Field(description="[x, y, z] of one corner.")],
        corner_max: Annotated[Vec3, Field(description="[x, y, z] of the opposite corner.")],
        operation: Operation = "new",
        targets: Targets = None,
    ) -> dict[str, Any]:
        """Axis-aligned block between two corners."""
        lo = [min(a, b) for a, b in zip(corner_min, corner_max, strict=True)]
        hi = [max(a, b) for a, b in zip(corner_min, corner_max, strict=True)]
        size = [h - lo_ for lo_, h in zip(lo, hi, strict=True)]
        if min(size) <= 0:
            raise ToolError(f"box has zero size: {size}")

        def make(design):
            pl = principal_plane(PlaneSpec(base="XY", offset=lo[2]), units)
            from .sketching import Box

            box = Box(
                center=[(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], width=size[0], height=size[1]
            )
            return design.extrude_sketch(
                name, build_sketch(pl, [box], units), units.distance(size[2])
            )

        return track(lambda: _place(name, make, operation, targets), _focus(session, targets))

    @tool(EDIT)
    def create_cylinder(
        name: str,
        base_center: Annotated[Vec3, Field(description="Centre of the base circle [x, y, z].")],
        radius: Annotated[float, Field(gt=0)],
        height: Annotated[float, Field(gt=0)],
        axis: Annotated[str | Vec3, Field(description='Axis direction ("Z" or [x, y, z]).')] = "Z",
        operation: Operation = "new",
        targets: Targets = None,
    ) -> dict[str, Any]:
        """Cylinder (or, with operation='cut', a round hole of that depth)."""

        def make(design):
            from .sketching import Circle

            pl = plane_on_face(units.point3d(base_center), as_direction(axis))
            sketch = build_sketch(pl, [Circle(center=[0, 0], radius=radius)], units)
            return design.extrude_sketch(name, sketch, units.distance(height))

        return track(lambda: _place(name, make, operation, targets), _focus(session, targets))

    @tool(EDIT)
    def create_sphere(
        name: str,
        center: Vec3,
        radius: Annotated[float, Field(gt=0)],
        operation: Operation = "new",
        targets: Targets = None,
    ) -> dict[str, Any]:
        """Sphere."""

        def make(design):
            return design.create_sphere(name, units.point3d(center), units.distance(radius))

        return track(lambda: _place(name, make, operation, targets), _focus(session, targets))

    @tool(EDIT)
    def boolean(
        operation: Literal["unite", "subtract", "intersect"],
        target: Annotated[str, Field(description="Body that is kept and modified.")],
        tools: Annotated[list[str], Field(min_length=1, description="Bodies combined into it.")],
        keep_tools: Annotated[
            bool, Field(description="Keep the tool bodies (subtract/intersect only).")
        ] = False,
    ) -> dict[str, Any]:
        """Unite, subtract or intersect bodies into target."""

        def op():
            t = session.body(target)
            others = [session.body(r) for r in tools]
            if any(o.id == t.id for o in others):
                raise ToolError("target must not be one of the tools")
            _require_contact(t, others, "the tool bodies")
            getattr(t, operation)(others, keep_other=keep_tools and operation != "unite")
            return {"target": t.id}

        with session.lock:
            result = track(op, _focus(session, [target, *tools]))
            target_id = result["target"]
            if not any(m["id"] == target_id for m in result["changes"]["modified"]):
                raise ToolError(
                    f"no_geometry_change: {operation} left the target {target_id} unchanged "
                    f"(changes: {result['changes']})"
                )
            return result

    @tool(EDIT)
    def move_body(
        bodies: Annotated[list[str], Field(min_length=1)],
        translate: Annotated[Vec3 | None, Field(description="Offset [dx, dy, dz].")] = None,
        rotate: Rotation | None = None,
        mirror: MirrorPlane | None = None,
        copy: Annotated[
            bool, Field(description="Transform copies and leave the originals in place.")
        ] = False,
    ) -> dict[str, Any]:
        """Translate, rotate and/or mirror bodies (applied in that order)."""
        if translate is None and rotate is None and mirror is None:
            raise ToolError("give translate, rotate or mirror")

        def op():
            moved = []
            for ref in bodies:
                b = session.body(ref)
                if copy:
                    b = b.copy(b.parent_component, f"{b.name}_copy")
                _transform(b, translate, rotate, mirror, units)
                entry = {"id": b.id, "name": b.name}
                if "/" in b.id:
                    entry["note"] = (
                        "body lives in a sub-component: the transform applies to its master "
                        "body, so every instance of that component moves"
                    )
                moved.append(entry)
            return {"bodies": moved}

        # A move shows up as a bounding-box change, but rotating a cylinder about its own
        # axis legitimately changes nothing; only a copy is certain to create a body.
        return track(op, _focus(session, bodies), expect_change=copy)

    @tool(EDIT)
    def pattern_body(
        body: str,
        count: Annotated[int, Field(ge=2, le=500, description="Total instances incl. original.")],
        kind: Literal["linear", "circular"] = "linear",
        direction: Annotated[str | Vec3, Field(description="linear: step direction.")] = "X",
        spacing: Annotated[
            float | None, Field(description="linear: distance between instances.")
        ] = None,
        axis_origin: Annotated[Vec3 | None, Field(description="circular: point on axis.")] = None,
        axis: Annotated[str | Vec3, Field(description="circular: axis direction.")] = "Z",
        total_angle: Annotated[
            float, Field(description="circular: span in degrees; 360 spreads evenly.")
        ] = 360.0,
        unite: Annotated[bool, Field(description="Merge all instances into the original.")] = False,
    ) -> dict[str, Any]:
        """Copy a body in a row or around an axis (bolt circles, rib arrays)."""
        if kind == "linear" and not spacing:
            raise ToolError("linear pattern needs spacing")
        if kind == "circular" and axis_origin is None:
            raise ToolError("circular pattern needs axis_origin")

        def op():
            src = session.body(body)
            copies = []
            step = (
                total_angle / count
                if math.isclose(total_angle, 360.0)
                else (total_angle / (count - 1))
            )
            for i in range(1, count):
                c = src.copy(src.parent_component, f"{src.name}_{i + 1}")
                if kind == "linear":
                    c.translate(as_direction(direction), units.distance(spacing * i))
                else:
                    c.rotate(units.point3d(axis_origin), as_direction(axis), units.angle(step * i))
                copies.append(c)
            if unite:
                src.unite(copies)
                return {"body": src.id}
            return {"bodies": [src.id, *[c.id for c in copies]]}

        return track(op, _focus(session, [body]))

    @tool(EDIT)
    def fillet_edges(
        body: str,
        edge_ids: Annotated[list[str], Field(min_length=1)],
        radius: Annotated[float, Field(gt=0)],
    ) -> dict[str, Any]:
        """Round edges with a constant radius (get ids from select_edges)."""

        def op():
            edges = session.edges(body, edge_ids)
            ok = session.require_modeler().geometry_commands.fillet(edges, units.distance(radius))
            if not ok:
                raise ToolError("fillet failed: radius too large for the adjacent faces?")

        return track(op, _focus(session, [body]))

    @tool(EDIT)
    def chamfer_edges(
        body: str,
        edge_ids: Annotated[list[str], Field(min_length=1)],
        distance: Annotated[float, Field(gt=0)],
    ) -> dict[str, Any]:
        """Bevel edges by an equal distance (get ids from select_edges)."""

        def op():
            edges = session.edges(body, edge_ids)
            ok = session.require_modeler().geometry_commands.chamfer(
                edges, units.distance(distance)
            )
            if not ok:
                raise ToolError("chamfer failed: distance too large for the adjacent faces?")

        return track(op, _focus(session, [body]))

    @tool(EDIT)
    def pull_faces(
        body: str,
        face_ids: Annotated[list[str], Field(min_length=1)],
        distance: Annotated[
            float,
            Field(
                description="Along the outward normal: positive adds material, negative removes it."
            ),
        ],
    ) -> dict[str, Any]:
        """SpaceClaim Pull: offset planar faces to change a size (make a plate thicker,
        deepen a pocket). To resize a hole, cut a larger cylinder instead."""
        from ansys.geometry.core.designer.geometry_commands import ExtrudeType, OffsetMode

        if distance == 0:
            raise ToolError("distance must not be zero")

        def op():
            faces = session.faces(body, face_ids)
            commands = session.require_modeler().geometry_commands
            if distance > 0:
                commands.extrude_faces(faces, units.distance(distance))
            else:
                # Inward pull as PyAnsys' own tests do it: negative distance, forced cut,
                # done as an extrusion rather than a face offset.
                commands.extrude_faces(
                    faces,
                    units.distance(distance),
                    None,
                    ExtrudeType.FORCE_CUT,
                    OffsetMode.IGNORE_RELATIONSHIPS,
                    False,
                    False,
                    True,
                )

        return track(op, _focus(session, [body]))

    @tool(EDIT)
    def shell_body(
        body: str,
        thickness: Annotated[float, Field(gt=0)],
        open_faces: Annotated[
            list[str] | None, Field(description="Faces to remove, leaving an open shell.")
        ] = None,
        side: Annotated[
            Literal["inward", "outward"],
            Field(description="inward keeps the outer size; outward keeps the inner size."),
        ] = "inward",
    ) -> dict[str, Any]:
        """Hollow a solid to a wall thickness, optionally opening some faces."""
        # PyAnsys offsets along the outward normal: a positive value grows the part.
        offset = -thickness if side == "inward" else thickness

        def op():
            b = session.body(body)
            if open_faces:
                if "/" in b.id:
                    raise ToolError(
                        f"{b.name} sits in a sub-component; PyAnsys cannot open faces of "
                        "instanced bodies. Use run_script, or shell without open_faces."
                    )
                ok = b.remove_faces(session.faces(body, open_faces), units.distance(offset))
            else:
                ok = b.shell_body(units.distance(offset))
            if not ok:
                raise ToolError("shell failed: thickness too large for the body?")

        return track(op, _focus(session, [body]))

    @tool(EDIT)
    def rename_body(body: str, new_name: str) -> dict[str, Any]:
        """Rename a body."""
        with session.lock:
            b = session.body(body)
            b.set_name(new_name)
            return {"id": b.id, "name": new_name}

    @tool(DESTRUCTIVE)
    def delete_body(body: str) -> dict[str, Any]:
        """Delete a body from the design (there is no undo through the API)."""

        def op():
            b = session.body(body)
            b.parent_component.delete_body(b)

        return track(op, _focus(session, [body]))

    @tool(EDIT)
    def create_named_selection(
        name: str,
        bodies: Annotated[list[str] | None, Field(description="Whole bodies.")] = None,
        body: Annotated[str | None, Field(description="Body owning face_ids/edge_ids.")] = None,
        face_ids: list[str] | None = None,
        edge_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Group bodies/faces/edges under a name (inlet/outlet/wall for CFD, fixed supports
        for FEA); Fluent and Mechanical pick these up."""
        if not (bodies or face_ids or edge_ids):
            raise ToolError("give bodies, face_ids or edge_ids")
        if (face_ids or edge_ids) and not body:
            raise ToolError("body is required with face_ids/edge_ids")
        with session.lock:
            design = session.require_design()
            ns = design.create_named_selection(
                name,
                bodies=[session.body(r) for r in bodies or []] or None,
                faces=session.faces(body, face_ids) if face_ids else None,
                edges=session.edges(body, edge_ids) if edge_ids else None,
            )
            return {"id": ns.id, "name": ns.name}

    @tool(DESTRUCTIVE)
    def run_script(
        code: Annotated[
            str,
            Field(
                description="SpaceClaim Python (IronPython) script using the SpaceClaim API "
                "of the running release. Read inputs from the dict argsDict; put outputs "
                "in a dict named result (string values)."
            ),
        ],
        args: Annotated[dict[str, str] | None, Field(description="Passed as argsDict.")] = None,
    ) -> dict[str, Any]:
        """Escape hatch: run a script inside the product for anything the typed tools do not
        cover. The design is re-read afterwards."""
        if not config.allow_scripts:
            raise ToolError("scripts are disabled on this server (SPACECLAIM_MCP_ALLOW_SCRIPTS=0)")
        script = textwrap.dedent(code)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mcp_script.py"
            path.write_text(script, encoding="utf-8")

            def op():
                modeler = session.require_modeler()
                result, _ = modeler.run_script_file(path, script_args=args or {})
                # Re-read separately: with import_design=True a script that leaves no
                # document open would raise after it ran, and its result would be lost.
                session.design = _safe(modeler.read_existing_design) or session.design
                return {"result": result}

            with session.lock:
                session.require_modeler()
                if session.design is None:
                    return op()  # nothing to compare against; the script may create one
                ids = [b.id for b in session.bodies()]
                return track(op, ids[:50] or None, expect_change=False)

    return mcp


# --- helpers ---------------------------------------------------------------------------


def _require_contact(tool, targets: list, tool_name: str) -> None:
    misses = [t for t in targets if _collision(tool, t) == "none"]
    if misses:
        names = ", ".join(f"{t.name} ({t.id})" for t in misses)
        raise ToolError(
            f"{tool_name} does not touch {names}: nothing to unite with or cut. Check the "
            "sketch plane, offset and direction (render_view helps)."
        )


def _collision(a, b) -> str | None:
    kind = _safe(lambda: a.get_collision(b))
    return None if kind is None else kind.name.lower()


def _discard_new_bodies(session: Session, known: set[str]) -> None:
    """After a failed edit, delete bodies it created (tool bodies, BoolOpCopy_* copies).

    A failing boolean raises after PyAnsys has already copied the tool bodies, and
    skips its own design refresh, so re-read the design first.
    """
    _safe(session.refresh_design)
    for b in _safe(session.bodies, default=[]):
        if b.id not in known:
            log.info("removing %s (%s) left behind by a failed edit", b.name, b.id)
            _safe(lambda b=b: b.parent_component.delete_body(b))


def _focus(session: Session, refs: list[str] | None) -> list[str] | None:
    """Ids of the bodies whose geometry an edit is expected to touch."""
    if not refs:
        return []
    with session.lock:
        return [session.body(r).id for r in refs]


def _offset_plane(plane, distance: float, units: Units):
    from ansys.geometry.core.math import Plane

    origin = units.point_out(plane.origin)
    shifted = [o + float(c) * distance for o, c in zip(origin, plane.direction_z, strict=True)]
    return Plane(
        origin=units.point3d(shifted),
        direction_x=plane.direction_x,
        direction_y=plane.direction_y,
    )


def _transform(body, translate, rotate, mirror, units: Units) -> None:
    from ansys.geometry.core.math import Plane

    if translate is not None:
        length = math.sqrt(sum(c * c for c in translate))
        if length > 0:
            body.translate(as_direction(translate), units.distance(length))
    if rotate is not None:
        body.rotate(
            units.point3d(rotate.origin), as_direction(rotate.axis), units.angle(rotate.angle)
        )
    if mirror is not None:
        n = as_direction(mirror.normal)
        tmp = plane_on_face(units.point3d(mirror.origin), n)
        body.mirror(
            Plane(origin=tmp.origin, direction_x=tmp.direction_x, direction_y=tmp.direction_y)
        )


PRODUCT_SAVE = {"stp", "step", "igs"}

_SAVE_SCRIPT = """\
options = ExportOptions.Create()
DocumentSave.Execute(argsDict["path"], options)
result = {"saved": argsDict["path"]}
"""


def _save_in_product(session: Session, path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "mcp_save.py"
        script.write_text(_SAVE_SCRIPT, encoding="utf-8")
        session.require_modeler().run_script_file(script, script_args={"path": str(path)})


def _written(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _export_stl(session: Session, target: Path) -> None:
    import pyvista as pv

    meshes = [b.tessellate(merge=True) for b in session.bodies()]
    if not meshes:
        raise ToolError("nothing to export: the design has no bodies")
    merged = pv.merge(meshes) if len(meshes) > 1 else meshes[0]
    surface = merged.extract_surface(algorithm="dataset_surface").triangulate()
    # STL has no units; write it in the working unit, which is what slicers expect (mm).
    surface.points *= session.units.per_metre
    surface.save(str(target), binary=True)


def _bodies_summary(session: Session) -> list[dict[str, Any]]:
    return [_body_summary(session, b) for b in session.bodies()]


def _body_summary(session: Session, b) -> dict[str, Any]:
    units = session.units
    info: dict[str, Any] = {
        "id": b.id,
        "name": b.name,
        "component": _safe(lambda: b.parent_component.name),
        "is_surface": _safe(lambda: bool(b.is_surface)),
        "volume": _safe(lambda: units.volume_out(b.volume)),
        "faces": _safe(lambda: len(b.faces)),
        "edges": _safe(lambda: len(b.edges)),
    }
    box = _safe(lambda: b.bounding_box)
    if box is not None:
        lo, hi = units.point_out(box.min_corner), units.point_out(box.max_corner)
        info["bbox"] = {
            "min": lo,
            "max": hi,
            "size": [round(h - lo_, 6) for lo_, h in zip(lo, hi, strict=True)],
        }
    return info


_SURFACES = {
    "SURFACETYPE_PLANE": "plane",
    "SURFACETYPE_CYLINDER": "cylinder",
    "SURFACETYPE_CONE": "cone",
    "SURFACETYPE_TORUS": "torus",
    "SURFACETYPE_SPHERE": "sphere",
    "SURFACETYPE_NURBS": "nurbs",
}
_CURVES = {
    "CURVETYPE_LINE": "line",
    "CURVETYPE_CIRCLE": "circle",
    "CURVETYPE_ELLIPSE": "ellipse",
    "CURVETYPE_NURBS": "nurbs",
}


def _surface_name(face) -> str:
    return _SURFACES.get(face.surface_type.name, "other")


def _face_info(units: Units, f) -> dict[str, Any]:
    kind = _surface_name(f)
    info: dict[str, Any] = {"id": f.id, "type": kind, "area": _safe(lambda: units.area_out(f.area))}
    center = _safe(f.point)
    if center is not None:
        info["center"] = units.point_out(center)
    if kind == "plane":
        n = _safe(f.normal)
        if n is not None:
            info["normal"] = [round(float(c), 6) for c in n]
    if kind in ("cylinder", "cone", "sphere"):
        r = _safe(lambda: f.shape.geometry.radius)
        if r is not None:
            info["radius"] = units.length_out(r)
    if kind == "torus":
        for key in ("major_radius", "minor_radius"):
            r = _safe(lambda key=key: getattr(f.shape.geometry, key))
            if r is not None:
                info[key] = units.length_out(r)
    if kind == "cylinder":
        ax = _safe(lambda: f.shape.geometry.dir_z)
        if ax is not None:
            info["axis"] = [round(float(c), 6) for c in ax]
    return info


def _edge_info(units: Units, e) -> dict[str, Any]:
    kind = _CURVES.get(e.curve_type.name, "other")
    info: dict[str, Any] = {
        "id": e.id,
        "type": kind,
        "length": _safe(lambda: units.length_out(e.length)),
    }
    start, end = _safe(lambda: e.start), _safe(lambda: e.end)
    if start is not None and end is not None:
        info["start"], info["end"] = units.point_out(start), units.point_out(end)
    if kind == "circle":
        r = _safe(lambda: e.shape.geometry.radius)
        if r is not None:
            info["radius"] = units.length_out(r)
        c = _safe(lambda: e.shape.geometry.origin)
        if c is not None:
            info["center"] = units.point_out(c)
    if "center" not in info and "start" in info:
        info["center"] = [
            round((a + b) / 2, 6) for a, b in zip(info["start"], info["end"], strict=True)
        ]
    return info


def _dot(a, b) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b, strict=True))


def _keep_extreme(items: list[dict[str, Any]], extreme: str | None) -> list[dict[str, Any]]:
    if not extreme or not items:
        return items
    which, axis = extreme.split("_")
    idx = "xyz".index(axis)
    values = [it["center"][idx] for it in items if "center" in it]
    if not values:
        return items
    best = max(values) if which == "max" else min(values)
    span = (max(values) - min(values)) or 1.0
    return [it for it in items if "center" in it and abs(it["center"][idx] - best) <= 1e-6 * span]


def _sample_edge(edge, factor: float, n: int = 48):
    import numpy as np

    shape = edge.shape
    pts = []
    for i in range(n + 1):
        pos = shape.evaluate_proportion(i / n).position
        pts.append([float(c) * factor for c in pos])
    return np.array(pts)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default
