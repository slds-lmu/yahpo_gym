cluster.functions = batchtools::makeClusterFunctionsSlurm("/home/lschnei8/slurm_wyoming.tmpl", array.jobs = FALSE)
default.resources = list(walltime = 300L, memory = 512L, ntasks = 1L, ncpus = 1L, nodes = 1L, clusters = "teton")

max.concurrent.jobs = 9999L
