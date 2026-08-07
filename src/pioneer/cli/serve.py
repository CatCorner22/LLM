"""Serving CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from pioneer.core.config import get_settings
from pioneer.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pioneer ML serving CLI")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port

    try:
        import uvicorn

        from pioneer.serving.app import create_app
    except ImportError:
        logger.error("serve_deps_missing", hint="pip install pioneer-ml[serve]")
        return 1

    logger.info("serve_cli_start", host=host, port=port)
    uvicorn.run(create_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
