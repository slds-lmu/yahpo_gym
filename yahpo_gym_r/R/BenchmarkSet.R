##' Benchmark Set
#'
#' R interface to a YAHPO Gym BenchmarkSet.
#'
#' @details
#' Allows exporting the objective function from a `YAHPO Gym` BenchmarkSet.
#' and additional helper functionality.
#'
#' @section Methods:
#'   * new(key, onnx_session, active_session): Initialize the class.
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

    #' @field id `character` \cr
    #'   The scenario identifier.
    id = NULL,

    #' @field instance `character` \cr
    #'   The instance identifier.
    instance = NULL,

    #' @field onnx_session `onnxruntime.InferenceSession` \cr
    #'   A session to use for the predict. If `NULL` a new session is initialized.
    onnx_session = NULL,

    #' @field active_session `logical` \cr
    #'   Should the benchmark run in an active `onnxruntime.InferenceSession`? Initialized to `FALSE`.
    active_session = NULL,

    #' @field multithread `logical` \cr
    #'   Should the ONNX session be allowed to leverage multithreading capabilities?
    multithread = NULL,

    #' @field check `logical` \cr
    #'   Check whether values coincide with `domain`.
    check = NULL,

    #' @field check_codomain `logical` \cr
    #'   Check whether returned values coincide with `codomain`.
    check_codomain = NULL,

    #' @field noisy `logical` \cr
    #'   Whether noisy surrogates should be used.
    noisy = NULL,

    #' @description
    #' Initialize a new object
    #'
    #' @param scenario `character` \cr
    #'   Key for a benchmark scenario. See [`list_benchmarks`] for more information.
    #' @param instance [`character`] \cr
    #'   A valid instance. See `instances`.
    #' @param onnx_session `onnxruntime.InferenceSession` \cr
    #'   A matching `onnxruntime.InferenceSession`. See `session` for more information.
    #'   If no session is provided, new session is created.
    #' @param active_session `logical` \cr
    #'   Should the benchmark run in an active `onnxruntime.InferenceSession`? Initialized to `FALSE`.
    #' @param multithread `logical` \cr
    #'   Should the ONNX session be allowed to leverage multithreading capabilities? Default `FALSE`.
    #' @param check `logical` \cr
    #'   Check inputs for validity before passing to surrogate model? Default `FALSE`.
    #' @param noisy `logical` \cr
    #'   Should noisy surrogates be used instead of deterministic ones?
    #' @param check_codomain `logical` \cr
    #'   Check outputs of surrogate model for validity? Default `FALSE`.
    initialize = function(scenario, instance = NULL, onnx_session = NULL, active_session = FALSE, multithread = FALSE, check = FALSE, noisy = FALSE, check_codomain = FALSE) {
      self$id = assert_string(scenario)
      self$instance = assert_string(instance, null.ok = TRUE)
      self$onnx_session = onnx_session
      self$active_session = assert_flag(active_session)
      self$multithread = assert_flag(multithread)
      self$check = assert_flag(check)
      self$noisy = assert_flag(noisy)
      self$check_codomain = assert_flag(check_codomain)
    },
    #' @description
    #' Printer with some additional information.
    print = function() {
      cat(format(self))
      cat("\n\n Targets: (Codomain)\n")
      print(self$codomain)
      cat("\nHyperparameters: (Domain)\n")
      print(self$domain)
      cat("\nAvailable instances:\n")
      print(self$instances)
    },

    #' @description
    #' Get the objective function
    #'
    #' @param instance [`character`] \cr
    #'   A valid instance. See `instances`.
    #' @param multifidelity (`logical`) \cr
    #'   Should the objective function respect multifidelity?
    #'   If `FALSE`, fidelity parameters are set as constants with their max fidelity in the domain.
    #' @param check_values (`logical`) \cr
    #'   Should values be checked by bbotk? Initialized to `TRUE`.
    #' @param timed (`logical`) \cr
    #'   Should function evaluation simulate runtime? Initialized to `FALSE`.
    #' @param logging (`logical`) \cr
    #'   Should function evaluation be logged? Initialized to `FALSE`.
    #' @param multithread `logical` \cr
    #'   Should the ONNX session be allowed to leverage multithreading capabilities? Default `FALSE`.
    #' @param seed `integer` \cr
    #'   Initial seed for the `onnxruntime.runtime`. Only relevant if `noisy = TRUE`. Default `NULL` (no seed).
    #' @param check_codomain `logical` \cr
    #'   Check outputs of surrogate model for validity? Default `FALSE`.
    #' @return
    #'  A [`Objective`][bbotk::Objective] containing "domain", "codomain" and a
    #'  functionality to evaluate the surrogates.
    get_objective = function(instance, multifidelity = TRUE, check_values = TRUE, timed = FALSE, logging = FALSE, multithread = FALSE, seed = NULL, check_codomain = FALSE) {
      assert_choice(instance, self$instances)
      assert_flag(check_values)
      assert_int(seed, null.ok = TRUE)
      assert_flag(check_codomain)
      ObjectiveYAHPO$new(
        instance,
        multifidelity,
        list(
          scenario = self$id,
          session = self$onnx_session,
          active_session = self$active_session,
          check = self$check,
          multithread = self$multithread,
          noisy = self$noisy
        ),
        self$domain,
        self$codomain,
        check_values = check_values,
        timed = timed,
        logging = logging,
        multithread = multithread,
        seed = seed,
        check_codomain = check_codomain
      )
    },
    #' @description
    #' Get Optimization Search Space
    #'
    #' A [`paradox::ParamSet`] describing the search_space used during optimization.
    #' Typically, this is the same as the domain but, e.g., with some parameters on log scale.
    #' This is the same space as the one returned by `get_opt_space_py` (with the instance param dropped).
    #' Typically this search_space should be provided when creating an [bbotk::OptimInstance].
    #' @param drop_instance_param [`logical`] \cr
    #'   Should the instance param (e.g., task id) be dropped? Defaults to `TRUE`.
    #' @param drop_fidelity_params [`logical`] \cr
    #'   Should fidelity params be dropped? Defaults to `FALSE`.
    #' @return
    #'  A [`paradox::ParamSet`] containing the search space to optimize over.
    get_search_space = function(drop_instance_param = TRUE, drop_fidelity_params = FALSE) {
      search_space = private$.load_r_domains()$search_space
      params = search_space$ids()
      if (drop_instance_param) {
        params = setdiff(params, self$py_instance$config$instance_names)
      }
      if (drop_fidelity_params) {
        params = setdiff(params, self$py_instance$config$fidelity_params)
      }
      search_space_new = search_space$subset(params)  # subset() handles trafo & dependencies

      search_space_new
    },

    #' @description
    #' Get Optimization ConfigSpace
    #'
    #' @param drop_fidelity_params [`logical`] \cr
    #'   Should fidelity params be dropped? Defaults to `TRUE`.
    #' @return
    #'  A configspace containing the search space to optimize over.
    get_opt_space_py = function(drop_fidelity_params = TRUE) {
      self$py_instance$get_opt_space(drop_fidelity_params)
    },

    #' @description
    #' Subset the codomain. Sets a new domain.
    #'
    #' @param keep (`character`) \cr
    #'   Vector of co-domain target names to keep.
    #' @return
    #'  A [`paradox::ParamSet`] containing the output space (codomain).
    subset_codomain = function(keep) {
      codomain = self$codomain
      assert_subset(keep, codomain$ids())
      new_codomain = codomain$subset(keep)
      private$.domains$codomain = new_codomain
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
    domain = function() {
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
    },

    #' @field py_instance [`BenchmarkSet`] \cr
    #'   A python `yahpo_gym.BenchmarkSet`.
    py_instance = function() {
      if (is.null(private$.py_instance)) {
        gym = reticulate::import("yahpo_gym")
        tmp = reticulate::py_capture_output({
          private$.py_instance = gym$benchmark_set$BenchmarkSet(
            scenario = self$id, instance = self$instance, session = self$onnx_session, active_session = self$active_session,
            multithread = self$multithread, noisy = self$noisy
          )
        })
      }
      return(private$.py_instance)
    }
  ),

  private = list(
    .py_instance = NULL,

    .domains = NULL,

    .load_r_domains = function(instance) {
      if (is.null(private$.domains)) {
        ps_path = self$py_instance$config$get_path("param_set")
        source(ps_path, local = environment())
        private$.domains = list(search_space = search_space, domain = domain, codomain = codomain)
      }
      private$.domains
    }
  )
)
