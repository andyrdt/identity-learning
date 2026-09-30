import argparse
from importlib import import_module

EXPERIMENTS = (
    "figure01_scalar",
    "figures02_10_initialization",
    "figure03_fourthroot",
    "figures04_06_depth",
    "figures05_07_table01_factorization",
    "figure08_covariance",
    "figure09_initialization_disk",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*", metavar="experiment", help="default: all")
    parser.add_argument("--fresh", action="store_true", help="rerun experiments even if results are cached")
    args = parser.parse_args()
    unknown = set(args.experiments) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiments: {', '.join(sorted(unknown))}")
    for name in args.experiments or EXPERIMENTS:
        experiment = import_module(f"experiments.{name}.run")
        if args.fresh or not all(path.is_file() for path in experiment.cache_files()):
            print(f"generate: {name}", flush=True)
            experiment.generate()
        print(f"plot: {name}", flush=True)
        experiment.plot()


if __name__ == "__main__":
    main()
