##' Benchmark Set
#'
#' R interface to a YAHPO GYM BenchmarkSet.
#'
#' @details
#' Allows exporting the objective function from a `YAHPO GYM` BenchmarkSet.
#' and additional helper functionality.
#'
#' @section Methods:
#'   * new(key, onnx_session, active_session, download): Initialize the class.
#'   * get_objective(): Obtain the [`bbotk::Objective`].
#'   * get_opt_space_py(): Obtain the [`ConfigSpace`].
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

    #' @field id `character` \cr
    #'   The scenario identifier.
    id = NULL,

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
    #'   Download the required data on instantiation? Default `FALSE`.
    #' @param check `logical` \cr
    #'   Check inputs for validity before passing to surrogate model? Default `FALSE`.
    initialize = function(key, onnx_session = NULL, active_session = FALSE, download = FALSE, check = FALSE) {
      # Initialize python instance
      gym = reticulate::import("yahpo_gym")
      self$id = assert_string(key)
      self$py_instance = gym$benchmark_set$BenchmarkSet(key, session=onnx_session,
        active_session = assert_flag(active_session), download = FALSE, check = assert_flag(check))
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
    #' @param check_values (`logical`) \cr
    #'   Should values be checked by bbotk? Initialized to `FALSE`.
    #' @param timed (`logical`) \cr
    #'   Should function evaluation simulate runtime? Initialized to `FALSE`.
    #' @param logging (`logical`) \cr
    #'   Should function evaluationd be logged? Initialized to `FALSE`.
    #' @return
    #'  A [`Objective`][bbotk::Objective] containing "domain", "codomain" and a
    #'  functionality to evaluate the surrogates.
    get_objective = function(instance, check_values = FALSE, timed = FALSE, logging = FALSE) {
      assert_choice(instance, self$instances)
      assert_flag(check_values)
      ObjectiveYAHPO$new(
        instance,
        self$py_instance,
        self$domain,
        self$codomain,
        check_values = check_values,
        timed = timed
      )
    },

    #' @description
    #' Get Optimization ConfigSpace
    #'
    #' @param instance [`instance`] \cr
    #'   A valid instance. See `instances`.
    #' @param drop_fidelity_params [`logical`] \cr
    #'   Should fidelity params be dropped? Defaults to `FALSE`.
    #' @return
    #'  A [`paradox::ParamSet`] containing the search space to optimize over.
    get_opt_space_py = function(instance, drop_fidelity_params = FALSE) {
      assert_choice(instance, self$instances)
      self$py_instance$get_opt_space(instance, drop_fidelity_params)
    },

    #' @description
    #' Subset the codomain
    #'
    #' @param keep (`character`) \cr
    #'   Vector of co-domain target names to keep.
    #' @return
    #'  A [`paradox::ParamSet`] containing the output space (codomain).
    subset_codomain = function(keep) {
      codomain = self$codomain
      assert_choice(keep, names(codomain$params))
      new_domain = ParamSet$new(codomain$params[names(codomain$params) %in% keep])
      private$.domains$codomain = new_domain
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
    },

    #' @field domain `ParamSet` \cr
    #' A [`paradox::ParamSet`] describing the domain to be optimized over.
    domain = function(){
      private$.load_r_domains()$domain
    },

    #' @field codomain `ParamSet` \cr
    #' A [`paradox::ParamSet`] describing the output domain.
    codomain = function() {
      private$.load_r_domains()$codomain
    },
    #' @field quant `numeric` \cr
    #' Multiply runtime by this factor. Defaults to 0.01.
    quant = function(val) {
      if (missing(val)) {
        return(quant)
      }
      assert_number(quant)
      self$py_instance$quant = val
    }
  ),
  private = list(
    .domains = NULL,

    .load_r_domains = function(instance) {
      if (is.null(private$.domains)) {
        ps_path = self$py_instance$config$get_path("param_set")
        source(ps_path, local = environment())
        private$.domains = list(domain = domain, codomain = codomain)
      }
      private$.domains
    }
  )
)