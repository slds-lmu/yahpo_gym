library(batchtools)
library(data.table)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3misc)
library(mlr3mbo)  # @moo
library(bbotk)  # @focussearch
library(paradox)  # @expression_params
library(miesmuschel)  # @ yahpo_mo
reticulate::use_virtualenv("mf_env/", required = TRUE)
#reticulate::use_virtualenv("/home/lps/.local/share/virtualenvs/yahpo_gym-4ygV7ggv/", required = TRUE)
library(reticulate)
yahpo_gym = import("yahpo_gym")

packages = c("data.table", "mlr3", "mlr3learners", "mlr3pipelines", "mlr3misc", "mlr3mbo", "bbotk", "paradox", "miesmuschel")

RhpcBLASctl::blas_set_num_threads(1L)
RhpcBLASctl::omp_set_num_threads(1L)

root = here::here()
experiments_dir = file.path(root, "experiments")

source_files = map_chr("helpers.R", function(x) file.path(experiments_dir, x))
for (sf in source_files) {
  source(sf)
}

#reg = makeExperimentRegistry(file.dir = "/gscratch/lschnei8/registry_yahpo_mo", packages = packages, source = source_files)
reg = makeExperimentRegistry(file.dir = NA, conf.file = NA, packages = packages, source = source_files)  # interactive session
saveRegistry(reg)
# reg = loadRegistry("registry_yahpo_mo_clean")  # to inspect the original registry on the cluster
# tab = getJobTable()

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

mies_wrapper = function(job, data, instance, ...) {
  reticulate::use_virtualenv("mf_env/", required = TRUE)
  library(yahpogym)
  logger = lgr::get_logger("bbotk")
  logger$set_threshold("warn")
  future::plan("sequential")

  optim_instance = make_optim_instance(instance)
  mutator_numeric = mut("cmpmaybe", mut("gauss"), p = 0.2)
  mutator_categorical = mut("cmpmaybe", mut("unif", can_mutate_to_same = FALSE), p = 0.2)
  mutator = mut("combine", list(ParamDbl = mutator_numeric, ParamInt = mutator_numeric, ParamFct = mutator_categorical, ParamLgl = mutator_categorical))
  recombinator = rec("maybe", rec("xounif", p = 0.2), p = 1)
  parent_selector_main = sel("tournament", scl("nondom"))
  survival_selector = sel("best", scl("nondom"))
  mu = floor(instance$budget / 6)
  lambda = floor(mu / 4)
  optimizer = opt("mies", lambda = lambda, mu = mu, survival_strategy = "plus", mutator = mutator, recombinator = recombinator, parent_selector = parent_selector_main, survival_selector)
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
addAlgorithm("mies", fun = mies_wrapper)
addAlgorithm("parego", fun = parego_wrapper)
addAlgorithm("smsego", fun = smsego_wrapper)
addAlgorithm("ehvi", fun = ehvi_wrapper)
addAlgorithm("mego", fun = mego_wrapper)

# setup scenarios and instances
get_lcbench_setup = function(budget_factor = 40L) {
  bench = yahpo_gym$benchmark_set$BenchmarkSet("lcbench", instance = "167152")
  ndim = length(bench$config_space$get_hyperparameter_names()) - 2L
  instances = c("167152", "167185", "189873")
  targets = list(c("val_accuracy", "val_cross_entropy"))
  budget = ceiling(20L + sqrt(ndim) * budget_factor)
  setup = setDT(expand.grid(scenario = "lcbench", instance = instances, targets = targets, ndim = ndim, budget = budget, stringsAsFactors = FALSE))
  setup[, minimize := map(targets, function(x) bench$config$config$y_minimize[match(x, bench$config$config$y_names)])]
  setup
}

get_iaml_setup = function(budget_factor = 40L) {
  setup = map_dtr(c("iaml_glmnet", "iaml_ranger", "iaml_xgboost", "iaml_super"), function(scenario) {
    bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario, instance = "1489")
    ndim = length(bench$config_space$get_hyperparameter_names()) - 2L
    instances = switch(scenario, iaml_glmnet = c("1489", "1067"), iaml_ranger = c("1489", "1067"), iaml_xgboost = c("40981", "1489"), iaml_super = c("1489", "1067"))
    targets = if (scenario == "iaml_xgboost") list(c("mmce", "nf", "ias"), c("mmce", "nf", "ias", "rammodel")) else if (scenario == "iaml_glmnet") list(c("mmce", "nf")) else list(c("mmce", "nf", "ias"))
    budget = ceiling(20L + sqrt(ndim) * budget_factor)
    setup = setDT(expand.grid(scenario = scenario, instance = instances, targets = targets, ndim = ndim, budget = budget, stringsAsFactors = FALSE))
    setup[, minimize := map(targets, function(x) bench$config$config$y_minimize[match(x, bench$config$config$y_names)])]
  })
}

get_rbv2_setup = function(budget_factor = 40L) {
  setup = map_dtr(c("rbv2_rpart", "rbv2_ranger", "rbv2_xgboost", "rbv2_super"), function(scenario) {
    bench = yahpo_gym$benchmark_set$BenchmarkSet(scenario, instance = "1040")
    ndim = length(bench$config_space$get_hyperparameter_names()) - 3L
    instances = switch(scenario, rbv2_rpart = c("41163", "1476", "40499"), rbv2_ranger = c("6", "40979", "375"), rbv2_xgboost = c("28", "182", "12"), rbv2_super = c("1457", "6", "1053"))
    targets = list(c("acc", "memory"))
    budget = ceiling(20L + sqrt(ndim) * budget_factor)
    setup = setDT(expand.grid(scenario = scenario, instance = instances, targets = targets, ndim = ndim, budget = budget, stringsAsFactors = FALSE))
    setup[, minimize := map(targets, function(x) bench$config$config$y_minimize[match(x, bench$config$config$y_names)])]
  })
}

setup = rbind(get_lcbench_setup(), get_iaml_setup(), get_rbv2_setup())

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
optimizers = data.table(algorithm = c("random", "randomx4", "mies", "parego", "smsego", "ehvi", "mego"))

for (i in seq_len(nrow(optimizers))) {
  algo_designs = setNames(list(optimizers[i, ]), nm = optimizers[i, ]$algorithm)

  ids = addExperiments(
    prob.designs = prob_designs,
    algo.designs = algo_designs,
    repls = 30L
  )
  addJobTags(ids, as.character(optimizers[i, ]$algorithm))
}

tab = getJobTable()
tab[, walltime := map_dbl(prob.pars, function(x) x$budget) * 40L]
tab["ehvi" %in% tags, walltime := walltime * 10]
jobs = tab[, c("job.id", "walltime")]
resources.default = list(memory = 2048L, ntasks = 1L, ncpus = 1L, nodes = 1L, clusters = "teton", max.concurrent.jobs = 9999L)
submitJobs(jobs, resources = resources.default)

done = findDone()
results = reduceResultsList(done, function(x, job) {
  pars = job$pars
  tmp = x$archive$data[, pars$prob.pars$targets, with = FALSE]
  tmp[, method := pars$algo.pars$algorithm]
  tmp[, scenario := pars$prob.pars$scenario]
  tmp[, instance := pars$prob.pars$instance]
  tmp[, targets := paste0(pars$prob.pars$targets, collapse = "_")]
  tmp[, iter := seq_len(.N)]
  tmp[, repl := job$repl]
  for (i in seq_along(pars$prob.pars$targets)) {
    target = pars$prob.pars$targets[i]
    minimize = pars$prob.pars$minimize[i]
    tmp[[target]] = (if (minimize) 1 else -1) * tmp[[target]]
  }
  tmp
})
results = rbindlist(results, fill = TRUE)
saveRDS(results, "results_mo.rds")

tab = getJobTable()
as.numeric(sum(tab$time.running), units = "hours")  # 2940.432 CPUh for our benchmark (optimizers + yahpo overhead which is negligable)

