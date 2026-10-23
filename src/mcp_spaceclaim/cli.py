"""Command line entry point: ``mcp-spaceclaim [--transport stdio|streamable-http]``."""

import argparse
import logging
import sys

from . import __version__
from .server import Config, build_server
from .session import Session
from .units import Units


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mcp-spaceclaim",
        description="MCP server for Ansys SpaceClaim / Discovery (2025 R2 or newer).",
    )
    parser.add_argument("--version", action="version", version=f"mcp-spaceclaim {__version__}")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio (default) for a local MCP client; streamable-http to serve over HTTP.",
    )
    parser.add_argument(
        "--http-host",
        default="127.0.0.1",
        help="Bind address for streamable-http. The server has no authentication: keep it on "
        "loopback or behind something that authenticates.",
    )
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--units", choices=["mm", "cm", "m", "in"], help="Working length unit.")
    parser.add_argument(
        "--product", choices=["spaceclaim", "discovery", "geometry-service"], help="To launch."
    )
    parser.add_argument("--ansys-version", type=int, help="Ansys release to launch, e.g. 252.")
    parser.add_argument("--no-scripts", action="store_true", help="Disable the run_script tool.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    # stdout carries the MCP protocol on stdio: logs must go to stderr.
    logging.basicConfig(stream=sys.stderr, level=args.log_level.upper())

    config = Config.from_env()
    if args.units:
        config.units = args.units
    if args.product:
        config.product = args.product
    if args.ansys_version:
        config.version = args.ansys_version
    if args.no_scripts:
        config.allow_scripts = False

    # One session for the whole process. It is closed here rather than in an MCP lifespan
    # hook because the SDK enters the lifespan once per client connection: closing there
    # would shut SpaceClaim down whenever one HTTP client went away.
    session = Session(units=Units(config.units))
    server = build_server(config, session=session)
    try:
        if args.transport == "stdio":
            server.run("stdio")
        else:
            server.run("streamable-http", host=args.http_host, port=args.http_port)
    finally:
        with session.lock:
            session.disconnect()


if __name__ == "__main__":
    main()
