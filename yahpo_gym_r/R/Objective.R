ObjectiveYAHPO = R6::R6Class("ObjectiveYAHPO",
  inherit = bbotk::ObjectiverFun,
  public = list(
    initialize = function(py_instance, domain, codomain = NULL) {
      if (is.null(codomain)) {
        codomain = ps(y = p_dbl(tags = "minimize"))
      }
      private$.py_instance = py_instance
      # asserts id, domain, codomain, properties
      super$initialize(
        id = paste0("YAHPO", private$.py_instance$config$config_id),
        domain = domain,
        codomain = codomain,
        properties = character(),
        constants = ps(),
        check_values = TRUE
      )
    },

    # @description
    # Evaluates input value(s) on the objective function. Calls the R function
    # supplied by the user.
    # @param xs Input values.
    eval = function(xs) {
      if (self$check_values) self$domain$assert(xs)
      res = invoke(private$.py_instance, xs, .args = self$constants$values)

    }
  ),
  private = list(
    .py_instance = NULL
  )
)
