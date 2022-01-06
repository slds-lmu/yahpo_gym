ObjectiveYAHPO = R6::R6Class("ObjectiveYAHPO",
  inherit = bbotk::ObjectiveRFun,
  public = list(
    timed = NULL,
    logging = NULL,

    initialize = function(instance, multifidelity = TRUE, py_instance, domain, codomain = NULL, check_values = TRUE, timed = FALSE, logging = FALSE, multithread = FALSE) {
      assert_flag(multifidelity)
      self$timed = assert_flag(timed)
      assert_flag(check_values)
      self$timed = assert_flag(timed)
      assert_flag(logging)
      assert_flag(multithread)
      if (is.null(codomain)) {
        codomain = ps(y = p_dbl(tags = "minimize"))
      }
      private$.py_instance = py_instance

      # set constant instance / fidelities and define domain over all other values
      instance_param = py_instance$config$instance_names
      fidelity_params = if (!multifidelity) py_instance$config$fidelity_params else NULL
      pars = setdiff(domain$ids(), c(instance_param, fidelity_params))
      domain_new = ParamSet$new(domain$params[pars])
      if (domain$has_trafo) {
        domain_new$trafo = domain$trafo
      }
      if (domain$has_deps) {
        domain_new$deps = domain$deps
      }

      # define constants param_set
      cst = ps()
      if (length(instance_param)) {
        cst$add(domain$params[[instance_param]])
        cst$values = insert_named(cst$values, y = setNames(list(instance), nm = instance_param))
      }
      if (length(fidelity_params)) {
        for (fidelity_param in fidelity_params) {
          cst$add(domain$params[[fidelity_param]])
          cst$values = insert_named(cst$values, y = setNames(list(domain$params[[fidelity_param]]$upper), nm = fidelity_param))
        }
      }

      if (self$timed) {
        fun = function(xs, ...) {self$py_instance$objective_function_timed(preproc_xs(xs, ...), logging = logging, multithread = multithread)[self$codomain$ids()]}
      } else {
        fun = function(xs, ...) {self$py_instance$objective_function(preproc_xs(xs, ...), logging = logging, multithread = multithread)[self$codomain$ids()]}
      }

      # asserts id, domain, codomain, properties
      super$initialize(
        id = paste0("YAHPO_", py_instance$config$config_id),
        domain = domain_new,
        codomain = codomain,
        properties = character(),
        constants = cst,
        check_values = assert_flag(check_values),
        fun = fun
      )
    }
  ),

  private = list(
    .py_instance = NULL
  ),

  active = list(
    py_instance = function() {
      private$.py_instance
    }
  )
)

#' @title Preprocess r object for use with python's YAHPO GYM
#' @param xs `list` \cr
#'   List of hyperparams
#' @param ... `any` \cr
#'   Named params, appended to `xs`.
#' @export
preproc_xs = function(xs, ...) {
  csts = list(...)
  xs = map(as.list(xs), function(x) {
    if (is.logical(x)) {
      as.character(x)  # NOTE: logical parameters are represented as categoricals in ConfigSpace and we fix this here
    } else {
      x
    }
  })
  keep(c(xs, csts), Negate(is.na))
}
