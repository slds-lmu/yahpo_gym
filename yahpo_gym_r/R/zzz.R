#' @import checkmate
#' @import mlr3misc
#' @import paradox
#' @import bbotk
#' @import R6
#' @import data.table
#' @importFrom stats setNames
#' @description
#' A package that connects YAHPO Gym to R.
"_PACKAGE"

.onLoad = function(libname, pkgname) { # nocov start
    reticulate::configure_environment(pkgname)
} # nocov end

# static code checks should not complain about commonly used data.table columns
utils::globalVariables(c("search_space", "domain", "codomain", "quant"))

leanify_package()
