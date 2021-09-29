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
    get_objective_function = function() {

    },
    get_param_set = function() {

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
