test_that("Objective eval and eval_many", {
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("lcbench", active_session = TRUE)
  id = reticulate::py_id(b$session)
})