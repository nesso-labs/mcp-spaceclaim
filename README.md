# mcp-spaceclaim

An [MCP](https://modelcontextprotocol.io) server for **Ansys SpaceClaim** and **Ansys
Discovery** (2025 R2 or newer). It lets Claude, or any other MCP client, model parts,
inspect geometry, look at shaded views of the result and export CAD files. It runs
through [PyAnsys Geometry](https://geometry.docs.pyansys.com), the official Python client
for the SpaceClaim/Discovery modelling API.

```text
you:    Make an L-bracket: 80×50×5 mm base, 45 mm wall on the short edge, two Ø8 holes
        15 mm from the free edge, R6 on the outer corners. Save it as STEP.
claude: spaceclaim_connect → new_design → sketch_extrude (base) → create_box (wall, add)
        → sketch_extrude (holes, cut) → select_edges → fillet_edges → render_view
        → save_design bracket.step
```

> **Status: alpha.** The unit tests run the real PyAnsys Geometry client objects and a
> fake backend. The end-to-end test against a live SpaceClaim (`tests/test_live.py`) has
> **not yet been run on 2025 R2**. If you have SpaceClaim, please run it (see
> [Development](#development)) and open an issue with the result.

Not affiliated with or endorsed by Ansys, Inc. *Ansys*, *SpaceClaim* and *Discovery* are
trademarks of ANSYS, Inc.

## What it can do

| Area | Tools |
|---|---|
| Session | `spaceclaim_connect` (launch or attach), `spaceclaim_status`, `spaceclaim_disconnect` |
| Documents | `new_design`, `open_file` (SCDOCX, STEP, Parasolid, IGES, ...), `save_design` (`.scdocx` `.step` `.x_t` `.x_b` `.igs` `.fmd` `.pmdb` `.dsco` `.stl`) |
| Sketch + solid | `sketch_extrude`, `revolve_sketch` (boxes, circles, regular polygons, slots, ellipses, polylines, line/arc paths, on a principal plane or on a planar face), `create_box`, `create_cylinder`, `create_sphere`. Each one can make a new body, `add` to bodies or `cut` from them |
| Edit | `boolean` (unite/subtract/intersect), `move_body` (translate/rotate/mirror, optionally as a copy), `pattern_body` (linear/circular), `fillet_edges`, `chamfer_edges`, `pull_faces` (SpaceClaim *Pull*), `shell_body`, `rename_body`, `delete_body`, `create_named_selection` (for Fluent/Mechanical) |
| Inspect | `list_bodies`, `describe_body`, `select_faces` / `select_edges` (select by type, normal, radius, length or position, e.g. "the top face", "the Ø8 holes", "the vertical edges"), `measure_distance` |
| See | `render_view`: a shaded PNG (iso/front/top/right/...) that the model looks at, with optional face/edge highlighting to check a selection before an operation |
| Anything else | `run_script` runs a SpaceClaim Python script inside the product (can be disabled) |

How it keeps the model honest:

- **Every modelling call reports what changed**: bodies created, deleted and modified,
  with volume and bounding box.
- **Silent no-ops become errors.** Some backend operations return success and change
  nothing: a cut that misses, a fillet that is too large. The server compares the
  geometry before and after, and fails with `no_geometry_change` so the model does not
  build on a step that did not happen.
- **One working unit** for every number in and out (millimetres by default), and degrees
  for angles. PyAnsys works in metres internally.
- **Errors keep their message.** SpaceClaim's own error text reaches the model, instead
  of the SDK's generic "Error executing tool".

## Requirements

- Windows with **SpaceClaim or Discovery 2025 R2+** installed and licensed (both ship
  the API server add-in that PyAnsys Geometry talks to). A headless Ansys Geometry
  Service also works.
- Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

## Install

Claude Code:

```powershell
claude mcp add spaceclaim -- uvx --from git+https://github.com/nesso-labs/mcp-spaceclaim mcp-spaceclaim
```

Claude Desktop (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "spaceclaim": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/nesso-labs/mcp-spaceclaim", "mcp-spaceclaim"],
      "env": { "SPACECLAIM_MCP_VERSION": "252" }
    }
  }
}
```

Any other MCP client: run `mcp-spaceclaim` (stdio).

Then ask for a part. The model calls `spaceclaim_connect` first; by default that launches
SpaceClaim 2025 R2 with its UI visible, so you can watch the model build.

### Attaching to a running session

`spaceclaim_connect` with `mode: "attach"` connects to a product that is already running
with the API server add-in (`host`, `port`, `transport_mode`). On Windows the default
transport is `wnua`. See PyAnsys Geometry's
[guide to existing sessions](https://geometry.docs.pyansys.com/version/stable/getting_started/existing/index.html).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SPACECLAIM_MCP_UNITS` | `mm` | Working length unit: `mm`, `cm`, `m`, `in` |
| `SPACECLAIM_MCP_PRODUCT` | `spaceclaim` | What `launch` starts: `spaceclaim`, `discovery`, `geometry-service` |
| `SPACECLAIM_MCP_VERSION` | `252` | Ansys release to launch (`252` = 2025 R2, `261` = 2026 R1); empty = newest installed |
| `SPACECLAIM_MCP_HIDDEN` | `0` | Launch without showing the UI |
| `SPACECLAIM_MCP_HOST` / `_PORT` / `_TRANSPORT_MODE` | `localhost` / `50051` / auto | Defaults for `attach` |
| `SPACECLAIM_MCP_ALLOW_SCRIPTS` | `1` | `0` removes the ability to run scripts (`run_script` returns an error) |
| `SPACECLAIM_MCP_OUTPUT_DIR` | `~/Documents/mcp-spaceclaim` | Where relative `save_design` paths go |

CLI flags override the same settings: `--units`, `--product`, `--ansys-version`,
`--no-scripts`, and `--transport streamable-http --http-host 127.0.0.1 --http-port 8765`
to serve over HTTP.

**Over HTTP the server has no authentication**, and anyone who can reach it can drive
SpaceClaim and run scripts. Keep it on loopback, or put it behind something that
authenticates. A machine running SpaceClaim is usually also the machine running the
MCP client, and stdio is the right choice there.

## Conventions

Sketch planes (`plane.base`) and what "+" extrudes along:

| Plane | local u | local v | `+` direction |
|---|---|---|---|
| `XY` | +X | +Y | +Z |
| `XZ` | +X | +Z | −Y |
| `YZ` | +Y | +Z | +X |
| planar face (`face_id` + `body`) | stable in-plane axis | normal × u | outward normal |

`offset` shifts the plane along its normal. Every sketch tool returns the exact frame it
used. Bodies are referenced by id, or by name when the name is unique. Face and edge ids
can change after a topology edit, so select them again rather than reusing old ones.

## Safety

- `new_design` and `open_file` close the current document. `delete_body` and
  `spaceclaim_disconnect` are marked destructive. There is no undo through the API, so
  save first.
- `run_script` executes arbitrary SpaceClaim Python inside the product with your user's
  rights. MCP clients ask before running it. Set `SPACECLAIM_MCP_ALLOW_SCRIPTS=0` if you
  don't want it.
- The server never sends data anywhere; it talks only to the local product.

## Development

```powershell
uv sync --extra dev
uv run pytest                 # unit tests: real PyAnsys client objects + fake backend
uv run ruff check src tests
```

Live end-to-end test (Windows + SpaceClaim). It builds the bracket above, checks every
volume against the hand-computed value, renders it and exports STEP/SCDOCX/STL:

```powershell
$env:SPACECLAIM_MCP_LIVE = "1"
uv run pytest -m live -v
```

### Design notes

- **Typed tools, with scripting as the exception.** The official
  [`ansys-common-mcp`](https://github.com/ansys/pyansys-common-mcp) servers mostly let the
  model write and run PyAnsys code. That is flexible, but every call is untyped code
  running on the host. Here the common operations are typed tools with validated
  arguments and checked results. `run_script` runs inside SpaceClaim's own scripting
  sandbox for everything else.
- The before/after geometry check follows the idea in
  [ansys-agent-bridge](https://github.com/1622352030/ansys-agent-bridge), which found that
  some backend booleans report success without changing anything.
- `render_view` draws the backend's tessellation with matplotlib on the Agg backend. It
  needs no OpenGL context, so it also works on a headless VM.

## License

MIT. See [LICENSE](LICENSE).
