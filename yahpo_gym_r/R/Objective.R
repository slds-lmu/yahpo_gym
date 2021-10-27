ObjectiveYAHPO = R6::R6Class("ObjectiveYAHPO",
  inherit = bbotk::ObjectiveRFun,
  public = list(
    initialize = function(instance, py_instance, domain, codomain = NULL, check_values = FALSE) {
      if (is.null(codomain)) {
        codomain = ps(y = p_dbl(tags = "minimize"))
      }
      private$.py_instance = py_instance

      # Set constant "instance" and define search space over all other values
      instance_param = names(which(map_lgl(domain$params, function(x) "task_id" %in% x$tags)))
      pars = setdiff(domain$ids(), instance_param)
      domain$values = setNames(map(pars, function(x) to_tune()), pars)

      # Define constants param_set
      cst = domain$params[instance_param]
      cst = invoke(ps, .args = cst)
      cst$values = setNames(list(instance), instance_param)

      # asserts id, domain, codomain, properties
      super$initialize(
        id = paste0("YAHPO_", private$.py_instance$config$config_id),
        domain = domain,
        codomain = codomain,
        properties = character(),
        constants = cst,
        check_values = assert_flag(check_values),
        fun = function(xs, ...) {
          self$py_instance$objective_function(preproc_xs(xs, ...))
        }
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
#' @export
preproc_xs = function(xs, ...) {
  csts = list(...)
  keep(c(as.list(xs), csts), Negate(is.na))
}

