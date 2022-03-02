library(batchtools)
library(data.table)
library(mlr3misc)
library(mlr3hyperband)
reticulate::use_virtualenv("mf_env/", required = TRUE)  # virtualenv where yahpo gym ist installed
library(reticulate)
yahpo_gym = import("yahpo_gym")

packages = c("data.table", "mlr3misc", "mlr3hyperband")

# FIXME: clean up logs/no logging?

#reg = makeExperimentRegistry(file.dir = "/gscratch/lschnei8/registry_yahpo_mf", packages = packages)
reg = makeExperimentRegistry(file.dir = NA, conf.file = NA, packages = packages)  # interactive session
saveRegistry(reg)
# reg = loadRegistry("registry_yahpo_mf_clean")  # to inspect the original registry on the cluster
# tab = getJobTable()

hb_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(reticulate)
  yahpo_gym = import("yahpo_gym")

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario, instance = instance$instance)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  if (grepl("rbv2_", instance$scenario)) {
    fidelity_param_id = "trainsize"
  } else {
    fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  }
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  schedule = hyperband_schedule(r_min = min_budget, r_max = max_budget, eta = 3, integer_budget = instance$on_integer_scale)
  n_iterations = ceiling(instance$budget / sum(schedule$budget * schedule$n)) * (max(schedule$bracket) + 3)

  py_run_file("hb_bohb_wrapper.py")
  res = py$run_hb(scenario = instance$scenario, instance = instance$instance, target = instance$target, minimize = instance$minimize, on_integer_scale = instance$on_integer_scale, n_iterations = n_iterations, seed = job$seed)
  res = as.data.table(res)
  res = res[cumsum(get(fidelity_param_id)) <= instance$budget, ]
  res[, method := "hb"]
  res[, scenario := instance$scenario]
  res[, target := instance$target]
  res[, instance := instance$instance]
  res[, repl := job$repl]
  res
}

bohb_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(reticulate)
  yahpo_gym = import("yahpo_gym")

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario, instance = instance$instance)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  if (grepl("rbv2_", instance$scenario)) {
    fidelity_param_id = "trainsize"
  } else {
    fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  }
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  schedule = hyperband_schedule(r_min = min_budget, r_max = max_budget, eta = 3, integer_budget = instance$on_integer_scale)
  n_iterations = ceiling(instance$budget / sum(schedule$budget * schedule$n)) * (max(schedule$bracket) + 3)

  py_run_file("hb_bohb_wrapper.py")
  res = py$run_bohb(scenario = instance$scenario, instance = instance$instance, target = instance$target, minimize = instance$minimize, on_integer_scale = instance$on_integer_scale, n_iterations = n_iterations, seed = job$seed)
  res = as.data.table(res)
  res = res[cumsum(get(fidelity_param_id)) <= instance$budget, ]
  res[, method := "bohb"]
  res[, scenario := instance$scenario]
  res[, target := instance$target]
  res[, instance := instance$instance]
  res[, repl := job$repl]
  res
}

optuna_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(reticulate)
  yahpo_gym = import("yahpo_gym")

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario, instance = instance$instance)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  if (grepl("rbv2_", instance$scenario)) {
    fidelity_param_id = "trainsize"
  } else {
    fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  }
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  schedule = hyperband_schedule(r_min = min_budget, r_max = max_budget, eta = 3, integer_budget = instance$on_integer_scale)
  # n_trials is more like a heuristic because optuna prunes tirals on its own we just make sure that we evaluate enough cumulative budget
  n_trials = ceiling(instance$budget / sum(schedule$budget * schedule$n)) * sum(schedule[, .(n = max(n)), by = .(bracket)]$n)

  py_run_file("optuna_wrapper.py")
  res = py$run_optuna(scenario = instance$scenario, instance = instance$instance, target = instance$target, minimize = instance$minimize, on_integer_scale = instance$on_integer_scale, n_trials = n_trials, seed = job$seed)
  res = as.data.table(res)
  res = res[cumsum(get(fidelity_param_id)) <= instance$budget, ]
  res[, method := "optuna"]
  res[, scenario := instance$scenario]
  res[, target := instance$target]
  res[, instance := instance$instance]
  res[, repl := job$repl]
  res
}

dehb_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(reticulate)
  yahpo_gym = import("yahpo_gym")

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario, instance = instance$instance)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  if (grepl("rbv2_", instance$scenario)) {
    fidelity_param_id = "trainsize"
  } else {
    fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  }
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  schedule = hyperband_schedule(r_min = min_budget, r_max = max_budget, eta = 3, integer_budget = instance$on_integer_scale)
  n_trials = ceiling(instance$budget / sum(schedule$budget * schedule$n)) * sum(schedule$n)

  py_run_string('sys.path.append("/home/lschnei8/DEHB/")')  # FIXME:
  py_run_file("dehb_wrapper.py")
  res = py$run_dehb(scenario = instance$scenario, instance = instance$instance, target = instance$target, minimize = instance$minimize, on_integer_scale = instance$on_integer_scale, n_trials = n_trials, seed = job$seed)
  res = as.data.table(res)
  res = res[cumsum(get(fidelity_param_id)) <= instance$budget, ]
  res[, method := "dehb"]
  res[, scenario := instance$scenario]
  res[, target := instance$target]
  res[, instance := instance$instance]
  res[, repl := job$repl]
  res
}

smac_mf_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(reticulate)
  yahpo_gym = import("yahpo_gym")

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario, instance = instance$instance)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  if (grepl("rbv2_", instance$scenario)) {
    fidelity_param_id = "trainsize"
  } else {
    fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  }
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  schedule = hyperband_schedule(r_min = min_budget, r_max = max_budget, eta = 3, integer_budget = instance$on_integer_scale)
  n_trials = ceiling(instance$budget / sum(schedule$budget * schedule$n)) * sum(schedule$n)

  py_run_file("smac_wrapper.py")
  res = py$run_smac4mf(scenario = instance$scenario, instance = instance$instance, target = instance$target, minimize = instance$minimize, on_integer_scale = instance$on_integer_scale, n_trials = n_trials, seed = job$seed)
  res = as.data.table(res)
  res = res[cumsum(get(fidelity_param_id)) <= instance$budget, ]
  res[, method := "smac4mf"]
  res[, scenario := instance$scenario]
  res[, target := instance$target]
  res[, instance := instance$instance]
  res[, repl := job$repl]
  res
}

smac_hpo_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(reticulate)
  yahpo_gym = import("yahpo_gym")

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario, instance = instance$instance)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  if (grepl("rbv2_", instance$scenario)) {
    fidelity_param_id = "trainsize"
  } else {
    fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  }
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  schedule = hyperband_schedule(r_min = min_budget, r_max = max_budget, eta = 3, integer_budget = instance$on_integer_scale)
  n_trials = as.integer(ceiling(instance$budget / max_budget))  # full budget

  py_run_file("smac_wrapper.py")
  res = py$run_smac4hpo(scenario = instance$scenario, instance = instance$instance, target = instance$target, minimize = instance$minimize, on_integer_scale = instance$on_integer_scale, n_trials = n_trials, seed = job$seed)
  res = as.data.table(res)
  res = res[cumsum(get(fidelity_param_id)) <= instance$budget, ]
  res[, method := "smac4hpo"]
  res[, scenario := instance$scenario]
  res[, target := instance$target]
  res[, instance := instance$instance]
  res[, repl := job$repl]
  res
}

random_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(reticulate)
  yahpo_gym = import("yahpo_gym")

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario, instance = instance$instance)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  if (grepl("rbv2_", instance$scenario)) {
    fidelity_param_id = "trainsize"
  } else {
    fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  }
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  n_trials = as.integer(ceiling(instance$budget / max_budget))  # full budget

  py_run_file("random_wrapper.py")
  res = py$run_random(scenario = instance$scenario, instance = instance$instance, n_trials = n_trials, seed = job$seed)
  res = as.data.table(res)
  res = res[cumsum(get(fidelity_param_id)) <= instance$budget, ]
  res[, method := "random"]
  res[, scenario := instance$scenario]
  res[, target := instance$target]
  res[, instance := instance$instance]
  res[, repl := job$repl]
  res
}


# add algorithms
addAlgorithm("hb", fun = hb_wrapper)
addAlgorithm("bohb", fun = bohb_wrapper)
addAlgorithm("optuna", fun = optuna_wrapper)
addAlgorithm("dehb", fun = dehb_wrapper)
addAlgorithm("smac_mf", fun = smac_mf_wrapper)
addAlgorithm("smac_hpo", fun = smac_hpo_wrapper)
addAlgorithm("random", fun = random_wrapper)

# setup scenarios and instances
get_nb301_setup = function(budget_factor = 40L) {
  scenario = "nb301"
  bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario, instance = "CIFAR10")
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  ndim = length(bench$config_space$get_hyperparameter_names()) - 1L  # NOTE: instance is not part of

  instances = "CIFAR10"
  target = "val_accuracy"
  budget = ceiling(20L * max_budget + sqrt(ndim) * max_budget * budget_factor)
  on_integer_scale = TRUE
  minimize = bench$config$config$y_minimize[match(target, bench$config$config$y_names)]
  setup = setDT(expand.grid(scenario = scenario, instance = instances, target = target, ndim = ndim, max_budget = max_budget, budget = budget, on_integer_scale = on_integer_scale, minimize = minimize, stringsAsFactors = FALSE))
  setup
}

get_lcbench_setup = function(budget_factor = 40L) {
  scenario = "lcbench"
  bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario, instance = "167168")
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  ndim = length(bench$config_space$get_hyperparameter_names()) - 2L

  instances = c("167168", "189873", "189906")
  target = "val_accuracy"
  budget = ceiling(20L * max_budget + sqrt(ndim) * max_budget * budget_factor)
  on_integer_scale = TRUE
  minimize = bench$config$config$y_minimize[match(target, bench$config$config$y_names)]
  setup = setDT(expand.grid(scenario = scenario, instance = instances, target = target, ndim = ndim, max_budget = max_budget, budget = budget, on_integer_scale = on_integer_scale, minimize = minimize, stringsAsFactors = FALSE))
  setup
}

get_rbv2_setup = function(budget_factor = 40L) {
  setup = map_dtr(c("rbv2_glmnet", "rbv2_rpart", "rbv2_ranger", "rbv2_xgboost", "rbv2_super"), function(scenario) {
    bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario, instance = "1040")
    fidelity_space = bench$get_fidelity_space()
    fidelity_param_id = "trainsize"
    min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
    max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
    ndim = length(bench$config_space$get_hyperparameter_names()) - 3L  # repl and trainsize and instance

    instances = switch(scenario, rbv2_glmnet = c("375", "458"), rbv2_rpart = c("14", "40499"), rbv2_ranger = c("16", "42"), rbv2_xgboost = c("12", "1501", "16", "40499"), rbv2_super = c("1053", "1457", "1063", "1479", "15", "1468"))
    target = "acc"
    budget = ceiling(20L * max_budget + sqrt(ndim) * max_budget * budget_factor)
    on_integer_scale = FALSE
    minimize = bench$config$config$y_minimize[match(target, bench$config$config$y_names)]
    setup = setDT(expand.grid(scenario = scenario, instance = instances, target = target, ndim = ndim, max_budget = max_budget, budget = budget, on_integer_scale = on_integer_scale, minimize = minimize, stringsAsFactors = FALSE))
  })
}

setup = rbind(get_nb301_setup(), get_lcbench_setup(), get_rbv2_setup())

setup[, id := seq_len(.N)]

# add problems
prob_designs = map(seq_len(nrow(setup)), function(i) {
  prob_id = paste0(setup[i, ]$scenario, "_", setup[i, ]$instance, "_", setup[i, ]$target)
  addProblem(prob_id, data = list(scenario = setup[i, ]$scenario, instance = setup[i, ]$instance, target = setup[i, ]$target, ndim = setup[i, ]$ndim, max_budget = setup[i, ]$max_budget, budget = setup[i, ]$budget, on_integer_scale = setup[i, ]$on_integer_scale, minimize = setup[i, ]$minimize))
  setNames(list(setup[i, ]), nm = prob_id)
})
nn = sapply(prob_designs, names)
prob_designs = unlist(prob_designs, recursive = FALSE, use.names = FALSE)
names(prob_designs) = nn

# add jobs for optimizers
optimizers = data.table(algorithm = c("hb", "bohb", "optuna", "dehb", "smac_mf", "smac_hpo", "random"))

for (i in seq_len(nrow(optimizers))) {
  algo_designs = setNames(list(optimizers[i, ]), nm = optimizers[i, ]$algorithm)

  ids = addExperiments(
    prob.designs = prob_designs,
    algo.designs = algo_designs,
    repls = 30L
  )
  addJobTags(ids, as.character(optimizers[i, ]$algorithm))
}

jobs = findJobs()
resources.default = list(walltime = 3600 * 3L, memory = 2048L, ntasks = 1L, ncpus = 1L, nodes = 1L, clusters = "teton", max.concurrent.jobs = 9999L)
submitJobs(jobs, resources = resources.default)

done = findDone()
results = reduceResultsList(done, function(x, job) {
  budget_var = if (job$instance$scenario %in% c("lcbench", "nb301")) "epoch" else "trainsize"
  target_var = job$instance$target
  if (!job$instance$minimize) {
    x[, (target_var) := - get(target_var)]
  }
  
  tmp = x[, c(target_var, budget_var, "method", "scenario", "instance", "repl"), with = FALSE]
  tmp[, iter := seq_len(.N)]
  colnames(tmp) = c("target", "budget", "method", "scenario", "instance", "repl", "iter")
  tmp
})
results = rbindlist(results, fill = TRUE)
saveRDS(results, "results_mf.rds")


tab = getJobTable()
as.numeric(sum(tab$time.running), units = "hours")  # 388.0598 CPUh for our benchmark (optimizers + yahpo overhead which is negligable)

running_time = reduceResultsList(done, function(x, job) {
  time_var = if (job$instance$scenario == "lcbench") "time" else if (job$instance$scenario == "nb301") "runtime" else "timetrain"
  sum(x[[time_var]])
})

sum(unlist(running_time)) / 3600 # 133996.3 CPUh for training time so 133996.3 + 388.0598 = 134384.4 for real

