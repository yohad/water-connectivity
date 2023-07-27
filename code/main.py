import argparse

from pathlib import Path
from os.path import abspath

from gilad_model import run_gilad_model


def main() -> None:
    parser = argparse.ArgumentParser()

    # Generic Parameters
    parser.add_argument(
        "--no-print",
        dest="print",
        help="print",
        action="store_false",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="output folder",
        default=abspath("."),
        type=Path,
    )

    # Model Parameters
    parser.add_argument(
        "-L", dest="length", help="domain size", type=int, default=20
    )
    parser.add_argument(
        "-N",
        dest="resolution",
        help="resolution of the domain",
        type=int,
        default=128,
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="time",
        help="simulation time",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-m",
        "--slope",
        dest="slope",
        help="slope of the terrain",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-p",
        "--precipitation",
        dest="precipitation",
        help="precipitation",
        type=float,
        default=1.1,
    )

    args = parser.parse_args()
    run_gilad_model(
        output=args.output,
        length=args.length,
        resolution=args.resolution,
        sim_time=args.time,
        slope=args.slope,
        precipitation=args.precipitation,
    )


if __name__ == "__main__":
    main()
