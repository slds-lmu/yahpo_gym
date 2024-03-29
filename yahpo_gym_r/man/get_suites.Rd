% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/get_suites.R
\name{get_suites}
\alias{get_suites}
\title{Get versioned task sets}
\usage{
get_suites(type, version = 1)
}
\arguments{
\item{type}{[`character`] \cr
Type of task set to get. For now 'single' (single-crit) and 'multi' (multi-crit) are supported.}

\item{version}{[`numeric`] \cr
Version of the task set to get. Default is `1.0`.}
}
\value{
A data.frame containing a list of tasks.
}
\description{
Get Optimization ConfigSpace
}
