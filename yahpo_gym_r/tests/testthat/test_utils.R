test_that("listing works", {
  skip_on_ci()
  reticulate::use_condaenv("yahpo_gym", required = TRUE)
  a = list_benchmarks()
  expect_class(a, "yahpo_gym.configuration.ConfigDict")
})

test_that("local config", {
  skip_on_ci()
  reticulate::use_condaenv("yahpo_gym", required = TRUE)
  tmp = tempfile()
  init_local_config("foo/bar", tmp)
  t = readLines(tmp)
  expect_true(t[1] == "data_path: foo/bar")
  unlink(tmp)
})

test_that("preproc_xs", {
  skip_on_ci()
  l = list(list(a = 1, b = NA))
  expect_identical(preproc_xs(l), list(list(a = 1)))
  expect_identical(preproc_xs(l, c = 3), list(list(a = 1, c = 3)))
  expect_identical(preproc_xs(l, c = 3, d = NA), list(list(a = 1, c = 3)))
})
