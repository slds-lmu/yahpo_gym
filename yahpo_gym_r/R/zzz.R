#' @import checkmate
#' @import mlr3misc
#' @import paradox
#' @import bbotk
#' @import R6
#' @description
#' A package that connects YAHPO GYM to R.
"_PACKAGE"

.onLoad = function(libname, pkgname) { # nocov start
    reticulate::configure_environment(pkgname)
} # nocov end

leanify_package()