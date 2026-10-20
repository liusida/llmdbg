"""Command-line entry point for llmdbg."""

import argparse

from llmdbg import __version__


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
    parser.parse_args()


if __name__ == "__main__":
    main()
