library(batchtools)
library(data.table)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3misc)
library(mlr3mbo)  # @moo
library(bbotk)  # @focussearch
reticulate::use_virtualenv("mf_env/", required = TRUE)
#reticulate::use_virtualenv("/home/lps/.local/share/virtualenvs/yahpo_gym-4ygV7ggv/", required = TRUE)

packages = c("data.table", "mlr3", "mlr3learners", "mlr3pipelines", "mlr3misc", "mlr3mbo", "bbotk")

RhpcBLASctl::blas_set_num_threads(1L)
RhpcBLASctl::omp_set_num_threads(1L)

reg = makeExperimentRegistry(file.dir = "/gscratch/lschnei8/registry_yahpo_mo", packages = packages)
#reg = makeExperimentRegistry(file.dir = NA, conf.file = NA)
saveRegistry(reg)

make_optim_instance = function(instance) {
  benchmark = BenchmarkSet$new(as.character(instance$scenario), download = FALSE)
  benchmark$subset_codomain(instance$subset[[1L]])
  objective = benchmark$get_objective(as.character(instance$instance), multifidelity = FALSE, check_values = FALSE)
  budget = 30 * benchmark$get_search_space(drop_fidelity_params = TRUE)$length
  optim_instance = OptimInstanceMultiCrit$new(objective, search_space = benchmark$get_search_space(drop_fidelity_params = TRUE), terminator = trm("evals", n_evals = budget), check_values = FALSE)
  optim_instance
}

random_search_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(yahpogym)
  logger = lgr::get_logger("bbotk")
  logger$set_threshold("warn")
  future::plan("sequential")

  optim_instance = make_optim_instance(instance)
  optimizer = opt("random_search")
  optimizer$optimize(optim_instance)
  optim_instance
}

parego_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(yahpogym)
  logger = lgr::get_logger("bbotk")
  logger$set_threshold("warn")
  future::plan("sequential")

  optim_instance = make_optim_instance(instance)
  acq_budget = 500L * optim_instance$search_space$length

  surrogate = default_surrogate(optim_instance, n_learner = 1L)
  acq_function = AcqFunctionEI$new()
  acq_optimizer = AcqOptimizer$new(opt("focus_search", n_points = 1000L, maxit = ceiling((acq_budget - 1000) / 1000)), terminator = trm("evals", n_evals = acq_budget))

  design = generate_design_lhs(optim_instance$search_space, 4L * optim_instance$search_space$length)$data
  optim_instance$eval_batch(design)

  bayesopt_parego(optim_instance, surrogate = surrogate, acq_function = acq_function, acq_optimizer = acq_optimizer)
  optim_instance
}


bohb_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(reticulate)
  yahpo_gym = import("yahpo_gym")

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
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

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
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

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
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

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
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

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
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

  bench = yahpo_gym$benchmark_set$BenchmarkSet(instance$scenario)
  bench$set_instance(instance$instance)
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
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
get_nb301_setup = function(budget_factor = 30) {
  scenario = "nb301"
  bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario)
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  ndim = length(bench$config_space$get_hyperparameter_names()) - 1L  # NOTE: instance is not part of

  instances = "CIFAR10"
  target = "val_accuracy"
  budget = ndim * max_budget * budget_factor
  on_integer_scale = TRUE
  minimize = bench$config$config$y_minimize[match(target, bench$config$config$y_names)]
  setup = setDT(expand.grid(scenario = scenario, instance = instances, target = target, ndim = ndim, max_budget = max_budget, budget = budget, on_integer_scale = on_integer_scale, minimize = minimize, stringsAsFactors = FALSE))
  setup
}

get_lcbench_setup = function(budget_factor = 30) {
  scenario = "lcbench"
  bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario)
  fidelity_space = bench$get_fidelity_space()
  fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
  min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
  max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
  ndim = length(bench$config_space$get_hyperparameter_names()) - 2L

  instances = c("167168", "189873", "189906")
  target = "val_accuracy"
  budget = ndim * max_budget * budget_factor
  on_integer_scale = TRUE
  minimize = bench$config$config$y_minimize[match(target, bench$config$config$y_names)]
  setup = setDT(expand.grid(scenario = scenario, instance = instances, target = target, ndim = ndim, max_budget = max_budget, budget = budget, on_integer_scale = on_integer_scale, minimize = minimize, stringsAsFactors = FALSE))
  setup
}

get_iaml_setup = function(budget_factor = 30) {
  setup = map_dtr(c("iaml_glmnet", "iaml_rpart", "iaml_ranger", "iaml_xgboost", "iaml_super"), function(scenario) {
    bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario)
    fidelity_space = bench$get_fidelity_space()
    fidelity_param_id = fidelity_space$get_hyperparameter_names()[1]
    min_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$lower
    max_budget = fidelity_space$get_hyperparameter(fidelity_param_id)$upper
    ndim = length(bench$config_space$get_hyperparameter_names()) - 2L

    instances = c("40981", "41146", "1489", "1067")
    target = "mmce"
    budget = ndim * max_budget * budget_factor
    on_integer_scale = FALSE
    minimize = bench$config$config$y_minimize[match(target, bench$config$config$y_names)]
    setup = setDT(expand.grid(scenario = scenario, instance = instances, target = target, ndim = ndim, max_budget = max_budget, budget = budget, on_integer_scale = on_integer_scale, minimize = minimize, stringsAsFactors = FALSE))
  })
}

setup = rbind(get_nb301_setup(), get_lcbench_setup(), get_iaml_setup())

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
    repls = 1L
  )
  addJobTags(ids, as.character(optimizers[i, ]$algorithm))
}

jobs = findJobs()
resources.default = list(walltime = 3600 * 12L, memory = 2048L, ntasks = 1L, ncpus = 1L, nodes = 1L, clusters = "teton", max.concurrent.jobs = 9999L)
submitJobs(jobs, resources = resources.default)

done = findDone()
results = reduceResultsList(done, function(x, job) {
  budget_var = if (job$instance$scenario %in% c("lcbench", "nb301")) "epoch" else "trainsize"
  target_var = job$instance$target
  # FIXME: minimize direction of target
  if (!job$instance$minimize) {
    x[, (target_var) := - get(target_var)]
  }
  
  tmp = x[, c(target_var, budget_var, "method", "scenario", "instance", "repl"), with = FALSE]
  tmp[, iter := seq_len(.N)]
  colnames(tmp) = c("target", "budget", "method", "scenario", "instance", "repl", "iter")
  tmp
})
results = rbindlist(results, fill = TRUE)
saveRDS(results, "results.rds")

