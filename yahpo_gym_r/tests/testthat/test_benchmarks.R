test_that("lcbench", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("lcbench", active_session = TRUE)
  obj = b$get_objective("34539")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("nb301", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("nb301", active_session = TRUE)
  obj = b$get_objective("CIFAR10")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("rbv2_super", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("rbv2_super", active_session = TRUE)
  obj = b$get_objective("15")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("rbv2_svm", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("rbv2_svm", active_session = TRUE)
  obj = b$get_objective("15")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("rbv2_glmnet", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("rbv2_glmnet", active_session = TRUE)
  obj = b$get_objective("15")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("rbv2_ranger", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("rbv2_ranger", active_session = TRUE)
  obj = b$get_objective("15")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("rbv2_aknn", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("rbv2_aknn", active_session = TRUE)
  obj = b$get_objective("15")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("rbv2_rpart", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("rbv2_rpart", active_session = TRUE)
  obj = b$get_objective("15")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("rbv2_xgboost", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("rbv2_xgboost", active_session = TRUE)
  obj = b$get_objective("15")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("iaml_glmnet", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("iaml_glmnet", active_session = TRUE)
  obj = b$get_objective("40981")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("iaml_rpart", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("iaml_rpart", active_session = TRUE)
  obj = b$get_objective("40981")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("iaml_ranger", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("iaml_ranger", active_session = TRUE)
  obj = b$get_objective("40981")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("iaml_xgboost", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("iaml_xgboost", active_session = TRUE)
  obj = b$get_objective("40981")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("iaml_super", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("iaml_super", active_session = TRUE)
  obj = b$get_objective("40981")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("fair_fgrrm", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("fair_fgrrm", active_session = TRUE)
  obj = b$get_objective("31")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("fair_rpart", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("fair_rpart", active_session = TRUE)
  obj = b$get_objective("31")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("fair_ranger", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("fair_ranger", active_session = TRUE)
  obj = b$get_objective("31")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("fair_xgboost", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("fair_xgboost", active_session = TRUE, check=FALSE)
  obj = b$get_objective("31")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})

test_that("fair_super", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("fair_super", active_session = TRUE, check=FALSE)
  obj = b$get_objective("31")
  des = paradox::generate_design_random(obj$domain, 10)$transpose()
  res = obj$eval_many(des)
  expect_data_table(res)
})
