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
    terminal = subparsers.add_parser('debug', help='Interactive/scriptable terminal DAP client.')
    terminal.add_argument('config', nargs='?', help='Path to a .llmdbg.py program')
    terminal.add_argument('--connect', metavar='HOST:PORT', help='Use an existing model host instead of a private host')
    terminal.add_argument('--json', action='store_true', help='One JSON result per command on stdout; progress on stderr')
    terminal.add_argument('--batch', action='store_true', help='Read commands from stdin; exit on first error')
    terminal.add_argument('-c', '--command', dest='commands', action='append', default=[], help='Execute a command (repeatable)')
    terminal.add_argument('--timeout', type=float, default=1800)
    terminal.add_argument('--model')
    terminal.add_argument('--device')
    terminal.add_argument('--dtype')
    terminal.add_argument('--max-new-tokens', type=int)
    terminal.add_argument('--demo', action='store_true', help='Use the built-in model-free demo')
    args = parser.parse_args()
    if args.command == "dap":
        dap_main()
    elif args.command == 'debug':
        from llmdbg.terminal import run
        run(args)


if __name__ == "__main__":
    main()
