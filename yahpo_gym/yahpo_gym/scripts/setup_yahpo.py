#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path
from yahpo_gym import benchmark_set
from yahpo_gym.local_config import local_config
import yahpo_gym.benchmarks.lcbench  # noqa: F401#


def setup(dest_dir: Path | str):
    """
    Clone yahpo data into <dest_dir>/yahpo_data
    """
    # Define the repository URL
    yahpo_data_url = "https://github.com/slds-lmu/yahpo_data.git@v2"

    # Run the git clone command
    dest_dir = Path(dest_dir).joinpath("yahpo_data")
    subprocess.run(["git", "clone", yahpo_data_url, dest_dir])

    local_config.init_config()
    local_config.set_data_path(dest_dir)


def test():
    bench = benchmark_set.BenchmarkSet("lcbench")
    bench.instances
    bench.set_instance("3945")
    value = bench.config_space.sample_configuration(1).get_dictionary()
    result = bench.objective_function(value)
    print(f"Eval objective for {value}: {result}")
    if result is not None:
        print("Setup successfull!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Setup script for yahpo-gym.")
    parser.add_argument(
        "dest_dir",
        help="Destination directory for cloning meta-data required for yahpo-gym.",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    print(args)
    setup(args.dest_dir)
    test()


if __name__ == "__main__":
    main()
