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
#' \dontrun{
#' b = BenchmarkSet$new("lcbench")
#' b$instances
#' }
#' @export
BenchmarkSet = R6::R6Class("BenchmarkSet",
  public = list(

    #' @field py_instance [`BenchmarkSet`] \cr
    #'   A python `yahpo_gym.BenchmarkSet`.
    py_instance = NULL,

    #' @description
    #' Initialize a new object
    #'
    #' @param key `character` \cr
    #'   Key for a benchmark scenario. See [`list_benchmarks`] for more information.
    #' @param onnx_session `onnxruntime.InferenceSession` \cr
    #'   A matching `onnxruntime.InferenceSession`. See `session` for more information.
    #'   If no session is provided, new session is created.
    #' @param active_session `logical` \cr
    #'   Should the benchmark run in an active `onnxruntime.InferenceSession`? Initialized to `FALSE`.
    #' @param download `logical` \cr
    #'   Download the required data on instantiation? Default `TRUE`.
    initialize = function(key, onnx_session = NULL,  download = TRUE, active_session = FALSE) {
      # Initialize python instance
      gym = reticulate::import("yahpo_gym")
      self$id = assert_string(key)
      self$py_instance = gym$benchmark_set$BenchmarkSet(key, session=onnx_session, active_session = active_session)
      # Download files
      if (assert_flag(download)) {
        self$py_instance$config$download_files(files = list("param_set.R"))
      }
    },

    #' @description
    #' Get the objective function
    #'
    #' @param instance [`instance`] \cr
    #'   A valid instance. See `instances`.
    #' @return
    #'  A [`Objective`][bbotk::Objective] containing "domain", "codomain" and a
    #'  functionality to evaluate the surrogates.
    get_objective = function(instance) {
      assert_choice(instance, self$instances)
      doms = private$.load_r_domains(instance)
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
    #' @field session `onnxruntime.InferenceSession` \cr
    #' Set/Get the ONNX session.
    #'
    #' The param `sess` is a `onnxruntime.InferenceSession`\cr
    #'   A matching `onnxruntime.InferenceSession`. Please make sure
    #'   that the session matches the selected benchmark scenario, as no
    #'   additional checks are performed and inference will fail during
    #'   evaluation of the objective function.
    session = function(sess) {
      if (missing(sess)) {
        return(self$py_instance$session)
      } else {
        self$py_instance$set_session(sess)
      }
    },
    #' @field instances `character` \cr
    #' A character vector of available instances for the scenario.
    instances = function() {
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
