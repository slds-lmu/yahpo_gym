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

root = here::here()
experiments_dir = file.path(root, "experiments")

source_files = map_chr("helpers.R", function(x) file.path(experiments_dir, x))
for (sf in source_files) {
  source(sf)
}

reg = makeExperimentRegistry(file.dir = "/gscratch/lschnei8/registry_yahpo_mo", packages = packages, source = source_files)
#reg = makeExperimentRegistry(file.dir = NA, conf.file = NA, source = source_files)
saveRegistry(reg)

# FIXME: acq_budget and n_points and maxit for focussearch
# FIXME: ehvi easier for computation
# FIXME: random interleaving in all methods?

make_optim_instance = function(instance) {
  benchmark = BenchmarkSet$new(as.character(instance$scenario), instance = as.character(instance$instance), download = FALSE)
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

randomx4_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(yahpogym)
  logger = lgr::get_logger("bbotk")
  logger$set_threshold("warn")
  future::plan("sequential")

  instance$budget = 4L * instance$budget

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
  acq_optimizer = AcqOptimizer$new(opt("focus_search", n_points = 500L, maxit = 5L), terminator = trm("evals", n_evals = acq_budget))

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
  acq_optimizer = AcqOptimizer$new(opt("focus_search", n_points = 500L, maxit = 5L), terminator = trm("evals", n_evals = acq_budget))

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
  acq_optimizer = AcqOptimizer$new(opt("focus_search", n_points = 500L, maxit = 5L), terminator = trm("evals", n_evals = acq_budget))

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
addAlgorithm("randomx4", fun = randomx4_wrapper)
addAlgorithm("parego", fun = parego_wrapper)
addAlgorithm("smsego", fun = smsego_wrapper)
addAlgorithm("ehvi", fun = ehvi_wrapper)
addAlgorithm("mego", fun = mego_wrapper)

# setup scenarios and instances
get_lcbench_setup = function(budget_factor = 30) {
  bench = yahpo_gym$benchmark_set$BenchmarkSet("lcbench", instance = "167152")
  ndim = length(bench$config_space$get_hyperparameter_names()) - 2L
  instances = c("167152", "167185", "189873")
  targets = list(c("val_accuracy", "val_cross_entropy"))
  budget = ndim * budget_factor
  setup = setDT(expand.grid(scenario = "lcbench", instance = instances, targets = targets, ndim = ndim, budget = budget, stringsAsFactors = FALSE))
  setup[, minimize := map(targets, function(x) bench$config$config$y_minimize[match(x, bench$config$config$y_names)])]
  setup
}

get_iaml_setup = function(budget_factor = 30) {
  setup = map_dtr(c("iaml_rpart", "iaml_ranger", "iaml_xgboost", "iaml_super"), function(scenario) {
    if (scenario == "iaml_super") budget_factor = 20L
    bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario, instance = "1489")
    ndim = length(bench$config_space$get_hyperparameter_names()) - 2L
    instances = switch(scenario, iaml_rpart = c("1489", "1067"), iaml_ranger = c("1489", "1067"), iaml_xgboost = c("40981", "1489"), iaml_super = c("1489", "1067"))
    targets = if (scenario == "iaml_xgboost") list(c("mmce", "nf"), c("mmce", "nf", "ias")) else list(c("mmce", "nf"))
    budget = ndim * budget_factor
    setup = setDT(expand.grid(scenario = scenario, instance = instances, targets = targets, ndim = ndim, budget = budget, stringsAsFactors = FALSE))
    setup[, minimize := map(targets, function(x) bench$config$config$y_minimize[match(x, bench$config$config$y_names)])]
  })
}

setup = rbind(get_lcbench_setup(), get_iaml_setup())

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
optimizers = data.table(algorithm = c("random", "randomx4", "parego", "smsego", "ehvi", "mego"))

for (i in seq_len(nrow(optimizers))) {
  algo_designs = setNames(list(optimizers[i, ]), nm = optimizers[i, ]$algorithm)

  ids = addExperiments(
    prob.designs = prob_designs,
    algo.designs = algo_designs,
    repls = 1L
  )
  addJobTags(ids, as.character(optimizers[i, ]$algorithm))
}

tab = getJobTable()
tab[, walltime := map_dbl(prob.pars, function(x) x$budget) * 30]
tab["ehvi" %in% tags, walltime := walltime * 10]
jobs = tab[, c("job.id", "walltime")]
resources.default = list(memory = 2048L, ntasks = 1L, ncpus = 1L, nodes = 1L, clusters = "teton", max.concurrent.jobs = 9999L)
submitJobs(jobs, resources = resources.default)

done = findDone()
results = reduceResultsList(done, function(x, job) {
  # FIXME:
})
results = rbindlist(results, fill = TRUE)
saveRDS(results, "results_mo.rds")

