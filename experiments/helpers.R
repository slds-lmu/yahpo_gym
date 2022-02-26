make_optim_instance = function(instance) {
  benchmark = BenchmarkSet$new(as.character(instance$scenario), instance = as.character(instance$instance))
  benchmark$subset_codomain(instance$targets[[1L]])
  objective = benchmark$get_objective(as.character(instance$instance), multifidelity = FALSE, check_values = FALSE)
  budget = instance$budget
  optim_instance = OptimInstanceMultiCrit$new(objective, search_space = benchmark$get_search_space(drop_fidelity_params = TRUE), terminator = trm("evals", n_evals = budget), check_values = FALSE)
  optim_instance
}

