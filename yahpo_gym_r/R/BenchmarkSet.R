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
      self$py_instance = gym$benchmark_set$BenchmarkSet('lcbench', download = TRUE)

    },
    #' @description
    #' Get the objective function
    #'
    #' @return
    #'  A [`bbotk::Objective`].
    get_objective_function = function(instance, drop_fidelity_params = TRUE) {
      ObjectiveYAHPO$new(
        private$.py_instance,
        self$get_opt_param_set(instance, drop_fidelity_params)
      )
    },

    #' @description
    #' Evaluate the objective function
    #'
    #' @param xs [`instance`] \cr
    #'   A valid configuration. See `get_opt_param_set`.
    #' @return
    #'  A numeric vector, prediction results.
    eval_objective_function = function(xs) {
      private$.py_instance$objective_function(xs)
    },

    #' @description
    #' Get Optimization Param Set
    #'
    #' @param instance [`instance`] \cr
    #'   A valid instance. See `instances`.
    #' @param drop_fidelity_params [`logical`] \cr
    #'   Should fidelity params be dropped?
    #' @return
    #'  A [`paradox::ParamSet`] containing the search space to optimize over.
    get_opt_space = function(instance, drop_fidelity_params = TRUE) {
      assert_character(instance)
      assert_flag(drop_fidelity_params)
      self$py_instance$get_opt_space(instance, drop_fidelity_params)
    }
  ),
  active = list(
    session = function(sess) {
      if (missing(sess)) {
        return(self$py_instance$session)
      } else {
        self$py_instance$set_session(sess)
      }
    },
    instances = function() {
      self$py_instance$instances
    }
  )
)
