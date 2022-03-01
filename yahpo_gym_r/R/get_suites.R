#' Get versioned task sets
#' @description
#' Get Optimization ConfigSpace
#'
#' @param type [`character`] \cr
#'   Type of task set to get. For now 'single' (single-crit) and 'multi' (multi-crit) are supported.
#' @param version [`numeric`] \cr
#'   Version of the task set to get. Default is `1.0`.
#' @return
#'  A data.frame containing a list of tasks.
get_suites = function(type, version = 1.0) {
    version = assert_numeric(version)
    gym = reticulate::import("yahpo_gym")
    gym$get_tasks$get_tasks(type, version)
}