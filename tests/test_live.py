"""End-to-end check against a real product. Skipped unless SPACECLAIM_MCP_LIVE=1.

Run it on the Windows machine that has SpaceClaim/Discovery installed:

    set SPACECLAIM_MCP_LIVE=1
    uv run pytest -m live -v

By default it launches SpaceClaim (SPACECLAIM_MCP_VERSION, default 252). Set
SPACECLAIM_MCP_LIVE_ATTACH=1 to use an already running session instead
(SPACECLAIM_MCP_HOST / SPACECLAIM_MCP_PORT / SPACECLAIM_MCP_TRANSPORT_MODE).

It builds an L-bracket with two holes and filleted corners, checks every volume
against the hand-computed value, renders it and exports STEP/SCDOCX/STL.
"""

import math
import os

import anyio
import pytest
from mcp import Client

from mcp_spaceclaim.server import Config, build_server
from mcp_spaceclaim.session import Session
from mcp_spaceclaim.units import Units

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("SPACECLAIM_MCP_LIVE") != "1", reason="set SPACECLAIM_MCP_LIVE=1"
    ),
]


def test_l_bracket_end_to_end(tmp_path):
    config = Config.from_env()
    config.output_dir = str(tmp_path)
    session = Session(units=Units("mm"))
    server = build_server(config, session=session)
    attach = os.environ.get("SPACECLAIM_MCP_LIVE_ATTACH") == "1"

    async def run():
        async with Client(server, read_timeout_seconds=600) as c:

            async def ok(name, args=None):
                result = await c.call_tool(name, args or {})
                assert not result.is_error, f"{name}: {result.content[0].text}"
                return result

            async def data(name, args=None):
                return (await ok(name, args)).structured_content

            await data("spaceclaim_connect", {"mode": "attach" if attach else "launch"})
            await data("new_design", {"name": "LiveBracket"})

            base = await data(
                "sketch_extrude",
                {
                    "name": "Base",
                    "shapes": [{"type": "box", "center": [40, 25], "width": 80, "height": 50}],
                    "distance": 5,
                },
            )
            base_id = base["body"]["id"]
            assert base["changes"]["created"][0]["volume"] == pytest.approx(20_000, rel=1e-6)

            await data(
                "create_box",
                {
                    "name": "Wall",
                    "corner_min": [0, 0, 5],
                    "corner_max": [5, 50, 50],
                    "operation": "add",
                    "targets": [base_id],
                },
            )
            bodies = (await data("list_bodies"))["bodies"]
            assert len(bodies) == 1
            assert bodies[0]["volume"] == pytest.approx(20_000 + 11_250, rel=1e-6)

            # Of the upward faces (plate top at z = 5, wall top at z = 50) the lowest is
            # the plate top, where the holes go.
            up = await data("select_faces", {"body": base_id, "normal": "Z", "extreme": "min_z"})
            assert up["count"] == 1 and up["faces"][0]["center"][2] == pytest.approx(5), up
            hole = await data(
                "sketch_extrude",
                {
                    "name": "Holes",
                    "shapes": [
                        {"type": "circle", "center": [55, 15], "radius": 4},
                        {"type": "circle", "center": [55, 35], "radius": 4},
                    ],
                    "plane": {"base": "XY", "offset": 5},
                    "distance": 5,
                    "direction": "-",
                    "operation": "cut",
                    "targets": [base_id],
                },
            )
            removed = 2 * math.pi * 4**2 * 5
            after_holes = hole["changes"]["modified"][0]["volume"]
            assert after_holes == pytest.approx(31_250 - removed, rel=1e-5)

            holes = await data(
                "select_faces", {"body": base_id, "surface_type": "cylinder", "radius": 4}
            )
            assert holes["count"] == 2

            vertical = await data(
                "select_edges", {"body": base_id, "parallel_to": "Z", "extreme": "max_x"}
            )
            assert vertical["count"] == 2  # the two outer corners of the plate
            await data(
                "fillet_edges",
                {"body": base_id, "edge_ids": [e["id"] for e in vertical["edges"]], "radius": 6},
            )
            final = (await data("list_bodies"))["bodies"][0]["volume"]
            fillet_loss = 2 * (6**2 - math.pi * 6**2 / 4) * 5
            assert final == pytest.approx(after_holes - fillet_loss, rel=1e-5)

            # A full revolve: PyAnsys implements it as a sweep and only tests 90 degrees
            # upstream. Washer: outer R 12, inner R 6, 3 thick, revolved about Z.
            washer = await data(
                "revolve_sketch",
                {
                    "name": "Washer",
                    "shapes": [{"type": "box", "center": [9, 101.5], "width": 6, "height": 3}],
                    "plane": {"base": "XZ"},
                    "axis_origin": [0, 0, 0],
                    "axis": "Z",
                    "angle": 360,
                },
            )
            expected = math.pi * (12**2 - 6**2) * 3
            assert washer["changes"]["created"][0]["volume"] == pytest.approx(expected, rel=1e-5)

            await data(
                "create_named_selection",
                {
                    "name": "mounting_holes",
                    "body": base_id,
                    "face_ids": [f["id"] for f in holes["faces"]],
                },
            )

            picture = await ok("render_view", {"views": ["iso", "top"], "size": 600})
            assert picture.content[0].type == "image"

            for name in ("bracket.step", "bracket.scdocx", "bracket.stl"):
                saved = await data("save_design", {"path": name})
                assert saved["bytes"] > 0

            if not attach:
                await data("spaceclaim_disconnect")

    anyio.run(run)
