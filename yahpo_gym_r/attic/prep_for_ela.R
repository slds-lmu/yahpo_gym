# Installation instructions: https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym_r
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
# Run 100 random evaluations
ins = bb_optimize(obj, max_evals = 100)
# Obtain results
df = ins$instance$archive$data[, c(names(bs$domain$params), names(bs$codomain$params)), with = FALSE] 
str(df, 1)

# Compute gower distance
library("gower")
gower_dist(df[1,], df[2,])

