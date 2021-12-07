# Installation instructions: https://github.com/pfistfl/yahpo_gym/tree/main/yahpo_gym_r
reticulate::use_condaenv("yahpo_gym", required=TRUE)
setwd("yahpo_gym_r")
devtools::load_all()

# Instiate the benchmark
bs = BenchmarkSet$new("nb301")

# Available targets
bs$codomain
# Available hyperparameters
bs$domain

bs$subset_codomain("val_accuracy")


# Available: bs$instances
obj = bs$get_objective("CIFAR10")


library("bbotk")

ins = bb_optimize(obj, max_evals = 100)
