import argparse
from pathlib import Path


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent
