""" Implementation of the command line interface.
"""
from argparse import ArgumentParser

from .__version__ import __version__
from .run_prediction import add_args as pred_args
from .run_prediction import predict
from .run_trainer import add_args as train_args
from .run_trainer import train

__all__ = "main",
 

def main(argv=None) -> int:
    """ Parse command line arguments. Then execute the application CLI.
    :param argv: argument list to parse
    :return: exit status
    """
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", action="version",
            version=f"T5Chem {__version__}",
            help="print version and exit")
    subparsers = parser.add_subparsers(title="subcommands")
    common = ArgumentParser(add_help=False)  # common subcommand arguments
    _execute(subparsers, common)
    args = parser.parse_args(argv)    
    if not hasattr(args, "command") or not args.command:
        # No sucommand was specified.
        parser.print_help()
        raise SystemExit(1)
    command = args.command
    try:
        command(args)
    except RuntimeError as err:
        return 1
    return 0
 

def _execute(subparsers, common):
    """ CLI adaptor for the api.hello command.
    :param subparsers: subcommand parsers
    :param common: parser for common subcommand arguments
    """
    train_parser = subparsers.add_parser("train", parents=[common])
    train_args(train_parser)
    train_parser.set_defaults(command=train)
    pred_parser = subparsers.add_parser("predict", parents=[common])
    pred_args(pred_parser)
    pred_parser.set_defaults(command=predict)
    return

# Make the module executable.

if __name__ == "__main__":
    main()
