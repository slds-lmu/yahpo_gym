# YAHPO-GYM

Surrogate based benchmarks for HPO problems


### Setup 

```py
# Initialize the local config & set save path for surrogates and metadata
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```

### Run Inference

```py
from yahpo_gym import BenchmarkSet
# Select a Benchmark
bench = BenchmarkSet("lcbench")
# List available instances
bench.instances
# Set an instance
bench.set_instance("3945")
value = {'epoch':1, 'batch_size':1, 'learning_rate':.1, 'momentum':.1, 'weight_decay':.1, 'num_layers':1, 'max_units':1, 'max_dropout':.1}
print(bench.objective_function(value))
```