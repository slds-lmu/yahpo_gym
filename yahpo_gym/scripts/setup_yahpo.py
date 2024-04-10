#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path
from yahpo_gym import benchmark_set
from yahpo_gym import local_config
import yahpo_gym.benchmarks.lcbench  # noqa: F401


def setup(dest_dir: Path | str):
    # Define the repository URL
    yahpo_data_url = "https://github.com/slds-lmu/yahpo_data.git"

    # Run the git clone command
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup script for yahpo-gym.")
    parser.add_argument(
        "dest_dir",
        help="Destination directory for cloning meta-data required for yahpo-gym.",
    )
    args = parser.parse_args()

    setup(args.destination_directory)
    test()
