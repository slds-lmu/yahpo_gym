Getting Started
************************



Installation (Python)
=======================

`YAHPO Gym` can be installed directly from GitHub using `pip`:

.. code-block:: bash

    pip install "git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"


Setup
=======================

To run a benchmark you need to obtain the ONNX model (`model.onnx`), `ConfigSpace <https://automl.github.io/ConfigSpace>`_ (`config_space.json`) and some encoding info (`encoding.json`).

You can download all files for all scenarios in a single folder `here (Github) <https://github.com/slds-lmu/yahpo_data>`_.

You should pertain the folder structure as on the hosting site (i.e., create a `"path-to-data"` directory, for example named `"multifidelity_data"`, containing the individual, e.g., `"lcench"`, directories).

.. code-block:: python

    # Initialize the local config & set path for surrogates and metadata
    from yahpo_gym import local_config
    local_config.init_config()
    local_config.set_data_path("path-to-data")


Usage
=======================

This example showcases the simplicity of `YAHPO Gym`'s API.
Additional, more in-depth examples can be found in the `Examples <https://slds-lmu.github.io/yahpo_gym/examples.html>`_.

.. code-block:: python

    from yahpo_gym import benchmark_set
    import yahpo_gym.benchmarks.lcbench
    # Select a Benchmark
    bench = benchmark_set.BenchmarkSet("lcbench")
    # List available instances
    bench.instances
    # Set an instance
    bench.set_instance("3945")
    # Sample a point from the configspace (containing parameters for the instance and budget)
    value = bench.config_space.sample_configuration(1).get_dictionary()
    # Evaluate
    print(bench.objective_function(value))


