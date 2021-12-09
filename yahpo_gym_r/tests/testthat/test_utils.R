test_that("listing works", {
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  a = list_benchmarks()
  expect_class(a, "yahpo_gym.configuration.ConfigDict")
})

test_that("local config", {
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  tmp = tempfile()
  init_local_config("foo/bar", tmp)
  t = readLines(tmp)
  expect_true(t[1] == "data_path: foo/bar")
  unlink(tmp)
})