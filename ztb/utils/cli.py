from typing import Callable


def run_main(main_fn: Callable[[], int]):
    """
    Run a main function that returns an exit code and ensure that the script
    process terminates with `sys.exit(code)` when executed directly.

    Use it from scripts like so:
        def main():
            ...
            return 0

        if __name__ == '__main__':
            from ztb.utils.cli import run_main
            run_main(main)
    """
    import sys

    exit_code = 0
    try:
        exit_code = main_fn()
    except SystemExit as se:
        # Allow the main function to raise SystemExit; re-raise to bubble out code
        raise se
    except Exception:
        # Do not swallow exceptions; print a minimal traceback and return 1
        import traceback

        traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)


def add_common_cli_args(parser):
    """Add common script CLI arguments like log-level"""
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    return parser


def configure_logging_from_args(args):
    """Set up basic logging formatting from parsed args"""
    import logging

    # Set basic configuration with timestamp and level if not yet configured
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
