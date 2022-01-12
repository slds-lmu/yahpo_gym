ObjectiveYAHPO = R6::R6Class("ObjectiveYAHPO",
  inherit = bbotk::Objective,
  public = list(
    timed = NULL,
    logging = NULL,
    multithread = NULL,

    initialize = function(instance, multifidelity = TRUE, py_instance_args, domain, codomain = NULL, check_values = TRUE, timed = FALSE, logging = FALSE, multithread = FALSE) {
      assert_flag(multifidelity)
      assert_flag(check_values)
      self$timed = assert_flag(timed)
      self$logging = assert_flag(logging)
      self$multithread = assert_flag(multithread)
      if (is.null(codomain)) {
        codomain = ps(y = p_dbl(tags = "minimize"))
      }
      private$.py_instance_args = assert_list(py_instance_args)

      # set constant instance / fidelities and define domain over all other values
      instance_param = self$py_instance$config$instance_names
      fidelity_params = if (!multifidelity) self$py_instance$config$fidelity_params else NULL
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

      # asserts id, domain, codomain, properties
      super$initialize(
        id = paste0("YAHPO_", py_instance_args$config_id),
        domain = domain_new,
        codomain = codomain,
        properties = character(),
        constants = cst,
        check_values = assert_flag(check_values)
      )

    },
    eval = function(xs) {
      if (self$check_values) self$domain$assert(xs)
      if (is.null(private$.fun)) {
        private$.set_fun()
      }
      res = invoke(private$.fun, xs, .args = self$constants$values)
      if (self$check_values) self$codomain$assert(as.list(res)[self$codomain$ids()])
      return(res)
    },
    export = function() {
      private$.export()
    }
  ),

  private = list(
    .fun = NULL,
    .py_instance = NULL,
    .py_instance_args = NULL,
    .export = function() {
      private$.py_instance = NULL
      private$.fun = NULL
      private$.py_instance_args$onnx_session = NULL
      private$.py_instance_args$active_session = FALSE
    },
    .set_fun = function() {
      if (self$timed) {
        private$.fun = function(xs, ...) {self$py_instance$objective_function_timed(preproc_xs(xs, ...), logging = self$logging, multithread = self$multithread)[self$codomain$ids()]}
      } else {
        private$.fun = function(xs, ...) {self$py_instance$objective_function(preproc_xs(xs, ...), logging = self$logging, multithread = self$multithread)[self$codomain$ids()]}
      }
    }
  ),


  active = list(
    #' @field py_instance (`yahpogym.BenchmarkInstnace`)\cr
    #' Python object.
    py_instance = function() {
      if (is.null(private$.py_instance) | is0x0ptr(private$.py_instance)) {
        gym = reticulate::import("yahpo_gym")
        args = private$.py_instance_args
        private$.py_instance = gym$benchmark_set$BenchmarkSet(
          args$config_id, session=args$onnx_session, active_session = args$active_session,
          download = args$download, check = args$check
        )
      }
      return(private$.py_instance)
    },
    #' @field fun (`function`)\cr
    #' Objective function.
    fun = function(lhs) {
      if (!missing(lhs) && !identical(lhs, private$.fun)) stop("fun is read-only")
      private$.fun
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
