test_that("listing works", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  a = list_benchmarks()
  expect_class(a, "yahpo_gym.configuration.ConfigDict")
})

test_that("local config", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  tmp = tempfile()
  init_local_config("foo/bar", tmp)
  t = readLines(tmp)
  expect_true(t[1] == "data_path: foo/bar")
  unlink(tmp)
})

test_that("preproc_xs", {
  l = list(a = 1, b = NA)
  expect_identical(preproc_xs(l), list(a = 1))
  expect_identical(preproc_xs(l, c = 3), list(a = 1, c = 3))
  expect_identical(preproc_xs(l, c = 3, d = NA), list(a = 1, c = 3))
})