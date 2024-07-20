ObjectiveYAHPO = R6::R6Class("ObjectiveYAHPO",
  inherit = bbotk::Objective,
  public = list(
    timed = NULL,
    logging = NULL,
    multithread = NULL,
    seed = NULL,
    check_codomain = NULL,

    initialize = function(instance, multifidelity = TRUE, py_instance_args, domain, codomain = NULL, check_values = TRUE, timed = FALSE, logging = FALSE, multithread = FALSE, seed = 0L, check_codomain = FALSE) {
      assert_flag(multifidelity)
      assert_flag(check_values)
      self$timed = assert_flag(timed)
      self$logging = assert_flag(logging)
      self$multithread = assert_flag(multithread)
      self$seed = assert_int(seed, null.ok = TRUE)
      self$check_codomain = assert_flag(check_codomain)

      if (is.null(codomain)) {
        codomain = ps(y = p_dbl(tags = "minimize"))
      }
      private$.py_instance_args = assert_list(py_instance_args, names = "named")

      # set constant instance / fidelities and define domain over all other values
      instance_param = self$py_instance$config$instance_names
      fidelity_params = if (!multifidelity) self$py_instance$config$fidelity_params else NULL
      pars = setdiff(domain$ids(), c(instance_param, fidelity_params))
      domain_new = domain$subset(pars)  # this also handles trafo and dependencies

      # define constants param_set
      cst = ps()
      if (length(instance_param)) {
        cst = ps_union(list(cst,
          domain$subset(instance_param)
        ))
        cst$set_values(.values = structure(list(instance), names = instance_param))
      }
      if (length(fidelity_params)) {
        cst = ps_union(list(cst,
          domain$subset(fidelity_params)
        ))
        cst$set_values(.values = structure(as.list(domain$upper[fidelity_params]), names = fidelity_params))
      }

      noise = ifelse(py_instance_args$noisy, "noisy", "deterministic")
      # asserts id, domain, codomain, properties
      super$initialize(
        id = paste0("YAHPO_", py_instance_args$scenario),
        domain = domain_new,
        codomain = codomain,
        properties = noise,
        constants = cst,
        check_values = assert_flag(check_values)
      )
    },
    eval = function(xs) {
      if (self$check_values) self$domain$assert(xs)
      if (is.null(private$.fun)) {
        private$.set_fun()
      }
      res = invoke(private$.fun, list(xs), .args = self$constants$values)
      res = res[[1]][self$codomain$ids()]
      if (self$check_codomain) self$codomain$assert(as.list(res)[self$codomain$ids()])
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
        private$.fun = function(xs, ...) {
          self$py_instance$objective_function_timed(
            preproc_xs(xs, ...), seed = self$seed,
            logging = self$logging, multithread = self$multithread
          )
        }
      } else {
        private$.fun = function(xs, ...) {
          self$py_instance$objective_function(
            preproc_xs(xs, ...), seed = self$seed,
            logging = self$logging, multithread = self$multithread
          )
        }
      }
    },
    .eval_many = function(xs, ...) {
      if (is.null(private$.fun)) {
        private$.set_fun()
      }
      res = invoke(private$.fun, xs = xs, .args = self$constants$values)
      data.table::rbindlist(res)[, self$codomain$ids(), with = FALSE]
    }
  ),


  active = list(
    #' @field py_instance (`yahpogym.BenchmarkInstance`)\cr
    #' Python object.
    py_instance = function() {
      if (is.null(private$.py_instance) | is0x0ptr(private$.py_instance)) {
        gym = reticulate::import("yahpo_gym")
        args = private$.py_instance_args
        private$.py_instance = gym$benchmark_set$BenchmarkSet(
          args$scenario, session=args$onnx_session, active_session = args$active_session,
          check = args$check, noisy = args$noisy, multithread = args$multithread
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

#' @title Preprocess r object for use with python's YAHPO Gym
#' @param xss `list` of `list` \cr
#'   List of hyperparams
#' @param ... `any` \cr
#'   Named params, appended to `xss`.
#' @export
preproc_xs = function(xss, ...) {
  csts = list(...)
  map(xss, function(xs) {
    xs = map(as.list(xs), function(x) {
      if (is.logical(x)) {
        as.character(x)  # NOTE: logical parameters are represented as categoricals in ConfigSpace and we fix this here
      } else {
        x
      }
    })
    keep(c(xs, csts), Negate(is.na))
  })
}
