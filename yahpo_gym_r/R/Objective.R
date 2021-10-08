ObjectiveYAHPO = R6::R6Class("ObjectiveYAHPO",
  inherit = bbotk::ObjectiveRFun,
  public = list(
    initialize = function(py_instance, domain, codomain = NULL, check_values = FALSE) {
      if (is.null(codomain)) {
        codomain = ps(y = p_dbl(tags = "minimize"))
      }
      private$.py_instance = py_instance
      # asserts id, domain, codomain, properties
      super$initialize(
        id = paste0("YAHPO_", private$.py_instance$config$config_id),
        domain = domain,
        codomain = codomain,
        properties = character(),
        constants = ps(),
        check_values = check_values,
        fun = function(xs) {
          self$py_instance$objective_function(preproc_xs(xs))
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
preproc_xs = function(xs) {
  keep(as.list(xs), Negate(is.na))
}