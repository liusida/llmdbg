"""Command-line entry point for llmdbg."""

import argparse

from llmdbg import __version__
from llmdbg.dap import main as dap_main


def main() -> None:
    """Run the llmdbg command-line interface."""
    parser = argparse.ArgumentParser(
        prog="llmdbg",
        description="Debug language models layer by layer.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("dap", help="Run the Debug Adapter Protocol server on stdio.")
    args = parser.parse_args()
    if args.command == "dap":
        dap_main()


if __name__ == "__main__":
    main()
