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
library(reticulate)
yahpo_gym = import("yahpo_gym")

packages = c("data.table", "mlr3", "mlr3learners", "mlr3pipelines", "mlr3misc", "mlr3mbo", "bbotk")

RhpcBLASctl::blas_set_num_threads(1L)
RhpcBLASctl::omp_set_num_threads(1L)

reg = makeExperimentRegistry(file.dir = "/gscratch/lschnei8/registry_yahpo_mo", packages = packages)
#reg = makeExperimentRegistry(file.dir = NA, conf.file = NA)
saveRegistry(reg)

make_optim_instance = function(instance) {
  benchmark = BenchmarkSet$new(as.character(instance$scenario), download = FALSE)
  benchmark$subset_codomain(instance$targets[[1L]])
  objective = benchmark$get_objective(as.character(instance$instance), multifidelity = FALSE, check_values = FALSE)
  budget = instance$budget
  optim_instance = OptimInstanceMultiCrit$new(objective, search_space = benchmark$get_search_space(drop_fidelity_params = TRUE), terminator = trm("evals", n_evals = budget), check_values = FALSE)
  optim_instance
}

random_wrapper = function(job, data, instance, ...) {
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
  acq_optimizer = AcqOptimizer$new(opt("focus_search", n_points = 1000L, maxit = 5L), terminator = trm("evals", n_evals = acq_budget))

  design = generate_design_lhs(optim_instance$search_space, 4L * optim_instance$search_space$length)$data
  optim_instance$eval_batch(design)

  bayesopt_parego(optim_instance, surrogate = surrogate, acq_function = acq_function, acq_optimizer = acq_optimizer)
  optim_instance
}

smsego_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(yahpogym)
  logger = lgr::get_logger("bbotk")
  logger$set_threshold("warn")
  future::plan("sequential")

  optim_instance = make_optim_instance(instance)
  acq_budget = 500L * optim_instance$search_space$length

  surrogate = default_surrogate(optim_instance)
  acq_function = AcqFunctionSmsEgo$new()
  acq_optimizer = AcqOptimizer$new(opt("focus_search", n_points = 1000L, maxit = 5L), terminator = trm("evals", n_evals = acq_budget))

  design = generate_design_lhs(optim_instance$search_space, 4L * optim_instance$search_space$length)$data
  optim_instance$eval_batch(design)

  bayesopt_smsego(optim_instance, surrogate = surrogate, acq_function = acq_function, acq_optimizer = acq_optimizer)
  optim_instance
}

ehvi_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(yahpogym)
  logger = lgr::get_logger("bbotk")
  logger$set_threshold("warn")
  future::plan("sequential")

  optim_instance = make_optim_instance(instance)
  acq_budget = 500L * optim_instance$search_space$length

  surrogate = default_surrogate(optim_instance)
  acq_function = AcqFunctionEHVI$new()
  acq_optimizer = AcqOptimizer$new(opt("focus_search", n_points = 1000L, maxit = 5L), terminator = trm("evals", n_evals = acq_budget))

  design = generate_design_lhs(optim_instance$search_space, 4L * optim_instance$search_space$length)$data
  optim_instance$eval_batch(design)

  bayesopt_ehvi(optim_instance, surrogate = surrogate, acq_function = acq_function, acq_optimizer = acq_optimizer)
  optim_instance
}

mego_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(yahpogym)
  logger = lgr::get_logger("bbotk")
  logger$set_threshold("warn")
  future::plan("sequential")

  optim_instance = make_optim_instance(instance)
  acq_budget = 500L * optim_instance$search_space$length

  acq_optimizer = AcqOptimizer$new(opt("random_search", batch_size = 1000L), terminator = trm("evals", n_evals = acq_budget))

  design = generate_design_lhs(optim_instance$search_space, 4L * optim_instance$search_space$length)$data
  optim_instance$eval_batch(design)

  bayesopt_mego(optim_instance, acq_optimizer = acq_optimizer)
  optim_instance
}

# add algorithms
addAlgorithm("random", fun = random_wrapper)
addAlgorithm("parego", fun = parego_wrapper)
addAlgorithm("smsego", fun = smsego_wrapper)
addAlgorithm("ehvi", fun = ehvi_wrapper)
addAlgorithm("mego", fun = mego_wrapper)

# setup scenarios and instances
get_iaml_setup = function(budget_factor = 30) {
  setup = map_dtr(c("iaml_glmnet", "iaml_rpart", "iaml_ranger", "iaml_xgboost", "iaml_super"), function(scenario) {
    bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario)
    ndim = length(bench$config_space$get_hyperparameter_names()) - 2L
    instances = c("40981", "41146", "1489", "1067")
    targets = list(c("mmce", "nf"))
    budget = ndim * budget_factor
    minimize = bench$config$config$y_minimize[match(targets[[1L]], bench$config$config$y_names)]

    setup = setDT(expand.grid(scenario = scenario, instance = instances, targets = targets, budget = budget, ndim = ndim, minimize = list(minimize), stringsAsFactors = FALSE))
  })
}

setup = rbind(get_iaml_setup())

setup[, id := seq_len(.N)]

# add problems
prob_designs = map(seq_len(nrow(setup)), function(i) {
  prob_id = paste0(setup[i, ]$scenario, "_", setup[i, ]$instance, "_", paste0(setup[i, ]$targets[[1L]], collapse = "_"))
  addProblem(prob_id, data = list(scenario = setup[i, ]$scenario, instance = setup[i, ]$instance, targets = setup[i, ]$targets, ndim = setup[i, ]$ndim, budget = setup[i, ]$budget, minimize = setup[i, ]$minimize))
  setNames(list(setup[i, ]), nm = prob_id)
})
nn = sapply(prob_designs, names)
prob_designs = unlist(prob_designs, recursive = FALSE, use.names = FALSE)
names(prob_designs) = nn

# add jobs for optimizers
optimizers = data.table(algorithm = c("random", "parego", "smsego", "ehvi", "mego"))

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

