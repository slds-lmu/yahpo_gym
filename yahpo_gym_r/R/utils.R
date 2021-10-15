#' @title Initialize a local configuration
#'
#' @details
#'  Allows setting a path the required meta-data for YAHPO Gym is downloaded to.
#' @param path `character`\cr
#'  Path to save meta-data to.
#' @param settings_path `character`\cr
#'  Path to save configuration object including "path" to.
#' @export
init_local_config = function(path = NULL, settings_path = "~/.config/yahpo_gym") {
  assert_string(path, null.ok = TRUE)
  assert_string(settings_path, null.ok = TRUE)
  lc = reticulate::import("yahpo_gym")$local_config
  lc$init_config()
  if (!is.null(path)) {
      lc$set_data_path(path)
  }
}

#' @title Print available benchmarks
#' @return A [`Configuration`] (Python Object)
#' @export
list_benchmarks = function() {
  yp = reticulate::import("yahpo_gym.configuration")
  yp$cfg()
}
