#' Get versioned task sets
#' @description
#' Get Optimization ConfigSpace
#'
#' @param type [`character`] \cr
#'   Type of task set to get. For now 'single' (single-crit) and 'multi' (multi-crit) are supported.
#' @param version [`integer`] \cr
#'   Version of the task set to get. Default is 0.
#' @return
#'  A data.frame containing a list of tasks.
get_tasks = function(type, version = 0) {
    gym = reticulate::import("yahpo_gym")
    gym$get_tasks$get_tasks(type, version)
}