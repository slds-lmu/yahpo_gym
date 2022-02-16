#' @title Initialize a local configuration
#'
#' @details
#'  Allows setting a path the required meta-data for YAHPO Gym is installed to.
#' @param data_path `character`\cr
#'  Path to save meta-data to.
#' @param settings_path `character`\cr
#'  Path to save configuration object including "path" to.
#' @export
init_local_config = function(data_path = NULL, settings_path = "~/.config/yahpo_gym") {
  assert_string(data_path, null.ok = TRUE)
  assert_string(settings_path, null.ok = TRUE)
  lc = reticulate::import("yahpo_gym.local_config")$LocalConfiguration(settings_path)
  lc$init_config(data_path = data_path)
  if (!is.null(data_path)) {
      lc$set_data_path(data_path)
  }
}

#' @title Print available benchmarks
#' @return A [`Configuration`] (Python Object)
#' @export
list_benchmarks = function() {
  yp = reticulate::import("yahpo_gym.configuration")
  yp$cfg()
}

is0x0ptr = function(pointer) {
  out = utils::capture.output(pointer)
  return(out == "<pointer: 0x0>")
}
