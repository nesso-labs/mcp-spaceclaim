"""Start the server as a subprocess over stdio, the way an MCP client does, and poke it."""

import sys

import anyio
from mcp import Client, StdioServerParameters

EXPECTED = {"spaceclaim_connect", "sketch_extrude", "render_view", "save_design", "run_script"}


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_spaceclaim"])
    async with Client(params) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        missing = EXPECTED - names
        assert not missing, f"missing tools: {missing}"
        status = await client.call_tool("spaceclaim_status", {})
        assert status.structured_content == {"connected": False, "units": "mm"}, status
        print(f"ok: {len(names)} tools, status {status.structured_content}")


anyio.run(main)
