#' @export
init_local_config = function(config, path = NULL) {
  lc = reticulate::import("yahpo_gym")$local_config
  lc$init_config()
  if (!is.null(path)) {
      ls$set_data_path("path")
  }
}