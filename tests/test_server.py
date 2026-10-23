"""Tool behaviour through a real MCP client, against the fake backend in fakes.py."""

from pathlib import Path

import anyio
import pytest
from mcp import Client

from mcp_spaceclaim.server import Config, build_server
from mcp_spaceclaim.session import Session
from mcp_spaceclaim.units import Units

from .fakes import FakeModeler


@pytest.fixture
def modeler():
    m = FakeModeler()
    m.create_design("Test")
    return m


@pytest.fixture
def server(modeler, tmp_path):
    session = Session(units=Units("mm"), modeler=modeler, design=modeler.design, product="fake")
    return build_server(Config(output_dir=str(tmp_path)), session=session)


def call(server, name, args=None):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, args or {})

    return anyio.run(go)


def ok(server, name, args=None):
    result = call(server, name, args)
    assert not result.is_error, result.content[0].text
    return result.structured_content


def error(server, name, args=None) -> str:
    result = call(server, name, args)
    assert result.is_error, result.structured_content
    return result.content[0].text


def list_tools(server):
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    return anyio.run(go)


# --- surface --------------------------------------------------------------------------


def test_tool_surface_and_hints(server):
    tools = {t.name: t for t in list_tools(server)}
    for name in ("spaceclaim_connect", "sketch_extrude", "render_view", "run_script"):
        assert name in tools
    assert tools["list_bodies"].annotations.read_only_hint is True
    assert tools["render_view"].annotations.read_only_hint is True
    assert tools["delete_body"].annotations.destructive_hint is True
    assert tools["new_design"].annotations.destructive_hint is True
    assert tools["sketch_extrude"].annotations.read_only_hint is False


def test_not_connected_is_a_readable_error(tmp_path):
    srv = build_server(Config(output_dir=str(tmp_path)), session=Session())
    assert "call spaceclaim_connect first" in error(srv, "list_bodies")
    assert ok(srv, "spaceclaim_status") == {"connected": False, "units": "mm"}


def test_old_backend_is_refused():
    session = Session(modeler=None)
    session.modeler = FakeModeler(version=(24, 2, 0))
    with pytest.raises(Exception, match="too old"):
        session._check_backend()
    assert session.modeler is None


def test_status_reports_backend(server):
    status = ok(server, "spaceclaim_status")
    assert status["connected"] is True
    assert status["backend_version"] == "25.2.0"
    assert status["design"] == "Test"


# --- modelling ------------------------------------------------------------------------


def test_sketch_extrude_creates_a_body_and_reports_it(server, modeler):
    out = ok(
        server,
        "sketch_extrude",
        {
            "name": "Plate",
            "shapes": [{"type": "box", "center": [0, 0], "width": 80, "height": 50}],
            "distance": 5,
        },
    )
    created = out["changes"]["created"]
    assert [c["name"] for c in created] == ["Plate"]
    assert created[0]["volume"] == pytest.approx(10 * 10 * 5)
    assert out["plane"]["normal"] == [0, 0, 1]
    (_, name, distance, direction, _sketch) = modeler.design.calls[-1]
    assert (name, distance, direction) == ("Plate", pytest.approx(0.005), "+")


def test_symmetric_extrude_starts_half_a_distance_below_the_plane(server, modeler):
    ok(
        server,
        "sketch_extrude",
        {
            "name": "Rib",
            "shapes": [{"type": "circle", "center": [0, 0], "radius": 3}],
            "distance": 10,
            "direction": "both",
            "plane": {"base": "XY", "offset": 20},
        },
    )
    sketch = modeler.design.calls[-1][-1]
    assert float(sketch.plane.origin[2]) == pytest.approx(0.015)


def test_cut_subtracts_from_targets_and_removes_the_tool_body(server, modeler):
    plate = modeler.design.add_block("Plate", [0, 0, 0], [80, 50, 5])
    out = ok(
        server,
        "sketch_extrude",
        {
            "name": "Hole",
            "shapes": [{"type": "circle", "center": [15, 15], "radius": 4}],
            "distance": 5,
            "operation": "cut",
            "targets": ["Plate"],
        },
    )
    assert out["cut_from"] == [plate.id]
    assert [m["id"] for m in out["changes"]["modified"]] == [plate.id]
    assert out["changes"]["modified"][0]["volume_change"] < 0
    assert [b.name for b in modeler.design.bodies] == ["Plate"]


def test_a_cut_that_changes_nothing_is_an_error(server, modeler):
    modeler.design.add_block("Plate", [0, 0, 0], [80, 50, 5], inert=True)
    msg = error(
        server,
        "sketch_extrude",
        {
            "name": "Hole",
            "shapes": [{"type": "circle", "center": [500, 500], "radius": 4}],
            "distance": 5,
            "operation": "cut",
            "targets": ["Plate"],
        },
    )
    assert "no_geometry_change" in msg


def test_a_cut_that_misses_is_refused_and_cleaned_up(server, modeler):
    modeler.design.add_block("Far", [500, 500, 500], [510, 510, 510])
    msg = error(
        server,
        "create_cylinder",
        {
            "name": "Drill",
            "base_center": [0, 0, 0],
            "radius": 3,
            "height": 5,
            "operation": "cut",
            "targets": ["Far"],
        },
    )
    assert "does not touch" in msg
    assert [b.name for b in modeler.design.bodies] == ["Far"]


def test_boolean_with_disjoint_tool_is_refused(server, modeler):
    modeler.design.add_block("A", [0, 0, 0], [1, 1, 1])
    modeler.design.add_block("B", [5, 5, 5], [6, 6, 6])
    msg = error(server, "boolean", {"operation": "subtract", "target": "A", "tools": ["B"]})
    assert "does not touch" in msg
    assert len(modeler.design.bodies) == 2


def test_operation_add_needs_targets(server):
    msg = error(
        server,
        "create_box",
        {"name": "B", "corner_min": [0, 0, 0], "corner_max": [1, 1, 1], "operation": "add"},
    )
    assert "needs targets" in msg


def test_create_box_normalises_corners(server, modeler):
    ok(server, "create_box", {"name": "B", "corner_min": [10, 0, 5], "corner_max": [0, 20, 0]})
    _, _, distance, _, sketch = modeler.design.calls[-1]
    assert distance == pytest.approx(0.005)
    assert float(sketch.plane.origin[2]) == pytest.approx(0.0)


def test_boolean_rejects_target_in_tools(server, modeler):
    a = modeler.design.add_block("A", [0, 0, 0], [1, 1, 1])
    assert "must not be one of the tools" in error(
        server, "boolean", {"operation": "unite", "target": a.id, "tools": [a.id]}
    )


def test_boolean_unite(server, modeler):
    a = modeler.design.add_block("A", [0, 0, 0], [10, 10, 10])
    b = modeler.design.add_block("B", [10, 0, 0], [20, 10, 10])
    out = ok(server, "boolean", {"operation": "unite", "target": "A", "tools": ["B"]})
    assert [d["id"] for d in out["changes"]["deleted"]] == [b.id]
    assert out["changes"]["modified"][0]["volume"] == pytest.approx(2000)
    assert a.calls[-1][0] == "unite"


def test_ambiguous_names_ask_for_an_id(server, modeler):
    modeler.design.add_block("Twin", [0, 0, 0], [1, 1, 1])
    modeler.design.add_block("Twin", [2, 0, 0], [3, 1, 1])
    assert "use the body id" in error(server, "describe_body", {"body": "Twin"})


def test_unknown_body_lists_what_exists(server, modeler):
    modeler.design.add_block("Plate", [0, 0, 0], [1, 1, 1])
    assert "Plate (" in error(server, "describe_body", {"body": "Nope"})


def test_move_translates_and_copy_creates(server, modeler):
    plate = modeler.design.add_block("Plate", [0, 0, 0], [10, 10, 1])
    out = ok(server, "move_body", {"bodies": ["Plate"], "translate": [5, 0, 0]})
    assert out["changes"]["modified"][0]["bbox"]["min"] == [5.0, 0.0, 0.0]
    assert plate.lo[0] == pytest.approx(0.005)
    out = ok(server, "move_body", {"bodies": ["Plate"], "translate": [0, 20, 0], "copy": True})
    assert out["changes"]["created"][0]["name"] == "Plate_copy"
    assert plate.lo[1] == pytest.approx(0.0)


def test_linear_pattern(server, modeler):
    modeler.design.add_block("Rib", [0, 0, 0], [2, 10, 10])
    out = ok(server, "pattern_body", {"body": "Rib", "count": 4, "spacing": 15})
    assert len(out["bodies"]) == 4
    xs = sorted(round(b.lo[0] * 1000, 6) for b in modeler.design.bodies)
    assert xs == [0, 15, 30, 45]


def test_circular_pattern_spreads_evenly(server, modeler):
    modeler.design.add_block("Bolt", [40, 0, 0], [45, 5, 5])
    ok(
        server,
        "pattern_body",
        {"body": "Bolt", "count": 6, "kind": "circular", "axis_origin": [0, 0, 0]},
    )
    angles = [b.calls[-1][1] for b in modeler.design.bodies[1:]]
    assert angles == pytest.approx([60, 120, 180, 240, 300])


def test_fillet_uses_edge_ids_and_detects_failure(server, modeler):
    body = modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    edge = body.edges[0].id
    out = ok(server, "fillet_edges", {"body": "Block", "edge_ids": [edge], "radius": 2})
    assert out["changes"]["modified"]
    assert modeler.geometry_commands.calls[-1] == ("fillet", [edge], pytest.approx(2))
    modeler.geometry_commands.succeed = False
    assert "fillet failed" in error(
        server, "fillet_edges", {"body": "Block", "edge_ids": [edge], "radius": 50}
    )


def test_stale_edge_id_points_to_reselect(server, modeler):
    modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    msg = error(server, "chamfer_edges", {"body": "Block", "edge_ids": ["E999"], "distance": 1})
    assert "select_edges" in msg


def test_pull_face_negative_distance_cuts(server, modeler):
    body = modeler.design.add_block("Plate", [0, 0, 0], [10, 10, 5])
    top = body.faces[0].id
    ok(server, "pull_faces", {"body": "Plate", "face_ids": [top], "distance": -2})
    assert modeler.geometry_commands.calls[-1][2:] == ("FORCE_CUT", True)
    assert body.hi[2] == pytest.approx(0.003)
    ok(server, "pull_faces", {"body": "Plate", "face_ids": [top], "distance": 4})
    assert modeler.geometry_commands.calls[-1][2:] == ("ADD", False)
    assert body.hi[2] == pytest.approx(0.007)


# --- selection ------------------------------------------------------------------------


def test_select_faces_by_normal_and_extreme(server, modeler):
    body = modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    out = ok(server, "select_faces", {"body": "Block", "normal": "Z"})
    assert [f["id"] for f in out["faces"]] == [body.faces[0].id]
    out = ok(server, "select_faces", {"body": "Block", "surface_type": "plane", "extreme": "min_z"})
    assert [f["id"] for f in out["faces"]] == [body.faces[1].id]


def test_select_edges_parallel_and_on_face(server, modeler):
    body = modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    out = ok(server, "select_edges", {"body": "Block", "parallel_to": "Z"})
    assert [e["id"] for e in out["edges"]] == [body.edges[1].id]
    out = ok(server, "select_edges", {"body": "Block", "on_faces": [body.faces[0].id]})
    assert [e["id"] for e in out["edges"]] == [body.edges[0].id]


def test_describe_body(server, modeler):
    modeler.design.add_block("Block", [0, 0, 0], [10, 20, 30])
    info = ok(server, "describe_body", {"body": "Block"})
    assert info["volume"] == pytest.approx(6000)
    assert info["bbox"]["size"] == [10, 20, 30]
    assert info["faces"][0]["normal"] == [0, 0, 1]
    assert info["edges"][0]["length"] == pytest.approx(10)


def test_measure_distance(server, modeler):
    modeler.design.add_block("A", [0, 0, 0], [1, 1, 1])
    modeler.design.add_block("B", [5, 0, 0], [6, 1, 1])
    out = ok(server, "measure_distance", {"body_a": "A", "body_b": "B"})
    assert out == {"distance": pytest.approx(4.2), "collision": "none", "units": "mm"}


# --- output ---------------------------------------------------------------------------


def test_render_view_returns_an_image(server, modeler):
    body = modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    result = call(
        server,
        "render_view",
        {
            "views": ["iso", "top"],
            "highlight_body": "Block",
            "highlight_faces": [body.faces[0].id],
            "highlight_edges": [body.edges[0].id],
            "size": 400,
        },
    )
    assert not result.is_error, result.content
    kinds = [c.type for c in result.content]
    assert kinds[0] == "image"
    assert result.content[0].mime_type == "image/png"


def test_save_picks_format_from_extension(server, modeler, tmp_path):
    modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    out = ok(server, "save_design", {"path": "part.step"})
    assert Path(out["path"]) == tmp_path / "part.step"
    assert modeler.design.calls[-1][2] == "STEP"
    assert "unsupported extension" in error(server, "save_design", {"path": "part.docx"})


def test_step_without_api_export_refuses_on_a_remote_product(server, modeler):
    modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])

    def no_step(path, fmt):
        raise RuntimeError("STEP export needs 27.1")

    modeler.design.download = no_step
    # The fixture session has no host: the product's own save would write on its machine.
    assert "remote product" in error(server, "save_design", {"path": "part.step"})


def test_step_fallback_on_a_local_product(modeler, tmp_path):
    modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    modeler.design.download = lambda path, fmt: (_ for _ in ()).throw(RuntimeError("27.1"))
    session = Session(modeler=modeler, design=modeler.design, host="localhost")
    srv = build_server(Config(output_dir=str(tmp_path)), session=session)
    out = ok(srv, "save_design", {"path": "part.step"})
    assert out["via"] == "product save"
    assert (tmp_path / "part.step").read_bytes().startswith(b"ISO-10303-21")
    assert not list(tmp_path.glob("*.mcp-partial*"))


def test_empty_export_is_an_error(server, modeler, tmp_path):
    modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    modeler.design.download = lambda path, fmt: Path(path).write_bytes(b"")
    assert "produced no file" in error(server, "save_design", {"path": "part.x_t"})


def test_shell_goes_inward_by_default(server, modeler):
    body = modeler.design.add_block("Box", [0, 0, 0], [10, 10, 10])
    ok(server, "shell_body", {"body": "Box", "thickness": 1.5})
    assert body.calls[-1] == ("shell", pytest.approx(-1.5))
    ok(server, "shell_body", {"body": "Box", "thickness": 1.5, "side": "outward"})
    assert body.calls[-1] == ("shell", pytest.approx(1.5))


def test_save_stl_from_tessellation(server, modeler, tmp_path):
    modeler.design.add_block("Block", [0, 0, 0], [10, 10, 10])
    out = ok(server, "save_design", {"path": str(tmp_path / "print.stl")})
    assert out["bytes"] > 84  # binary STL header + count


def test_named_selection(server, modeler):
    body = modeler.design.add_block("Duct", [0, 0, 0], [10, 10, 100])
    out = ok(
        server,
        "create_named_selection",
        {"name": "inlet", "body": "Duct", "face_ids": [body.faces[1].id]},
    )
    assert out["name"] == "inlet"
    assert modeler.design.named_selections[0].faces[0].id == body.faces[1].id


def test_run_script_passes_args_and_refreshes(server, modeler):
    out = ok(server, "run_script", {"code": "result = {'ok': '1'}", "args": {"n": "3"}})
    assert out["result"] == {"ok": "1"}
    assert modeler.scripts[-1] == ("result = {'ok': '1'}", {"n": "3"})
    assert [c["name"] for c in out["changes"]["created"]] == ["FromScript"]


def test_run_script_can_be_disabled(modeler, tmp_path):
    session = Session(modeler=modeler, design=modeler.design)
    srv = build_server(Config(allow_scripts=False, output_dir=str(tmp_path)), session=session)
    assert "disabled" in error(srv, "run_script", {"code": "pass"})


def test_failed_cut_does_not_leave_the_tool_body(server, modeler):
    plate = modeler.design.add_block("Plate", [0, 0, 0], [80, 50, 5])

    def boom(other, keep_other=False):
        raise RuntimeError("backend refused the boolean")

    plate.subtract = boom
    msg = error(
        server,
        "create_cylinder",
        {
            "name": "Drill",
            "base_center": [10, 10, 0],
            "radius": 3,
            "height": 5,
            "operation": "cut",
            "targets": ["Plate"],
        },
    )
    assert "backend refused the boolean" in msg
    assert [b.name for b in modeler.design.bodies] == ["Plate"]


def test_run_script_without_a_design(modeler, tmp_path):
    session = Session(modeler=modeler, design=None)
    modeler.design = None
    srv = build_server(Config(output_dir=str(tmp_path)), session=session)
    modeler.create_design("ScriptMade")
    session.design = None
    out = ok(srv, "run_script", {"code": "result = {}"})
    assert out["result"] == {"ok": "1"}


def test_new_design_reports_what_it_closed(server):
    out = ok(server, "new_design", {"name": "Bracket"})
    assert out == {"design": "Bracket", "closed_previous": "Test"}
