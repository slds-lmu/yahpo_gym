##' Benchmark Set
#'
#' R interface to a YAHPO GYM BenchmarkSet.
#'
#' @details
#' Allows exporting the objective function from a `YAHPO GYM` BenchmarkSet.
#' and additional helper functionality.
#'
#' @section Methods:
#'   * new(key, onnx_session): Initialize the class.
#'   * get_objective_function(): Obtain the [`bbotk::ObjectiveFunction`].
#'   * get_param_set(): Obtain the [`paradox::ParamSet`].
#'
#' @section Fields:
#'   * session: [`onnx.InferenceSession`] \cr
#'   * instances: [`character`] \cr
#'
#' @examples
#' @export
BenchmarkSet = R6::R6Class("BenchmarkSet",
  public = list(
    py_instance = NULL,
    initialize = function(key, onnx_session = NULL) {
      # Initialize python instance
      gym = reticulate::import("yahpo_gym")
      self$id = assert_string(key)
      self$py_instance = gym$benchmark_set$BenchmarkSet(key, download = TRUE)

    },
    #' @description
    #' Get the objective function
    #'
    #' @return
    #'  A [`bbotk::Objective`] containing a "domain" and "codomain".
    get_objective = function(instance, drop_fidelity_params = TRUE) {
      assert_choice(instance, self$instances)
      doms = private$.load_r_domains(drop_fidelity_params)
      ObjectiveYAHPO$new(
        self$py_instance,
        doms$domain,
        doms$codomain
      )
    },

    #' @description
    #' Get Optimization ConfigSpace
    #'
    #' @param instance [`instance`] \cr
    #'   A valid instance. See `instances`.
    #' @param drop_fidelity_params [`logical`] \cr
    #'   Should fidelity params be dropped?
    #' @return
    #'  A [`paradox::ParamSet`] containing the search space to optimize over.
    get_opt_space_py = function(instance) {
      assert_choice(instance, self$instances)
      self$py_instance$get_opt_space(instance, drop_fidelity_params)
    }
  ),
  active = list(
    #' @description
    #' Set/Get the ONNX session.
    #'
    #' @param sess `onnxruntime.InferenceSession`\cr
    #'   A matching `onnxruntime.InferenceSession`. Please make sure
    #'   that the session matches the selected benchmark scenario, as no
    #'   additional checks are performed and inference will fail during
    #'   evaluation of the objective function.
    #' @return `onnxruntime.InferenceSession`
    session = function(sess) {
      if (missing(sess)) {
        return(self$py_instance$session)
      } else {
        self$py_instance$set_session(sess)
      }
    },
    instances = function() {
      #' @description
      #' A character vector of available instances for the scenario.
      self$py_instance$instances
    }
  ),
  private = list(
    .load_r_domains = function() {
      ps_path = self$py_instance$config$get_path("param_set")
      source(ps_path, local = environment())
      list(domain = domain, codomain = codomain)
    }
  )
)
