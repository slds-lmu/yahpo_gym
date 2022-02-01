test_that("benchmarkset can be instantiated", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("lcbench", active_session = TRUE)
  id = reticulate::py_id(b$session)
  expect_r6(b, "BenchmarkSet")
  expect_character(b$instances, len = 35)
  expect_class(b$py_instance, "yahpo_gym.benchmark_set.BenchmarkSet")
  expect_class(b$session, "onnxruntime.capi.onnxruntime_inference_collection.InferenceSession")
  expect_true(b$id == "lcbench")
  expect_class(b$get_opt_space_py("3945", FALSE), "ConfigSpace.configuration_space.ConfigurationSpace")
  # Can build the objective
  obj = b$get_objective("189909")
  expect_r6(obj, "ObjectiveYAHPO")
  expect_r6(obj$domain, "ParamSet")
  expect_r6(obj$codomain, "ParamSet")
  # Can be optimized
  p = opt("random_search")
  ois = OptimInstanceMultiCrit$new(obj, terminator = trm("evals", n_evals = 10), check_values = FALSE)
  p$optimize(ois)
  expect_true(nrow(ois$archive$data) == 10L)
  # We stay in the same session
  expect_true(id == reticulate::py_id(b$session))
  # Errors
  expect_error(b$get_objective("123"))

  b2 = BenchmarkSet$new("lcbench", onnx_session = b$session)
  expect_true(id == reticulate::py_id(b2$session))
  expect_r6(b2, "BenchmarkSet")
  expect_character(b2$instances, len = 35)
  expect_class(b2$py_instance, "yahpo_gym.benchmark_set.BenchmarkSet")
  expect_class(b2$session, "onnxruntime.capi.onnxruntime_inference_collection.InferenceSession")
  expect_true(b2$id == "lcbench")
  expect_class(b2$get_opt_space_py("3945", FALSE), "ConfigSpace.configuration_space.ConfigurationSpace")
  # Can build the objective
  obj2 = b2$get_objective("189909")
  expect_r6(obj2, "ObjectiveYAHPO")
  expect_r6(obj2$domain, "ParamSet")
  expect_r6(obj2$codomain, "ParamSet")
  # Can be optimized
  p = opt("random_search")
  ois2 = OptimInstanceMultiCrit$new(obj2, terminator = trm("evals", n_evals = 10), check_values = FALSE)
  p$optimize(ois2)
  expect_true(nrow(ois2$archive$data) == 10L)
  expect_true(id == reticulate::py_id(b2$session))
})

test_that("subsetting works", {
  skip("Tested locally")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("lcbench", active_session = TRUE)
  b$subset_codomain("val_accuracy")
  expect_true(names(b$codomain$params) == "val_accuracy")
})

test_that("Parallel", {
  skip("Tested locally")
  options(future.globals.onReference = "string")
  reticulate::use_condaenv("yahpo_gym", required=TRUE)
  b = BenchmarkSet$new("lcbench")
  objective = b$get_objective("3945", timed = FALSE, check_values = FALSE)

  xdt = generate_design_random(b$get_search_space(), 1)$data
  xss_trafoed = transform_xdt_to_xss(xdt, b$get_search_space())
  objective$eval_many(xss_trafoed)
  
  future::plan("multisession")
  pss = replicate(2, {
    xdt = generate_design_random(b$get_search_space(), 1)$data
    xss_trafoed = transform_xdt_to_xss(xdt, b$get_search_space())
    promise = future::future(objective$eval_many(xss_trafoed), packages = "yahpogym", seed = NULL, lazy = TRUE)
  })
  map(pss, future::value)

  promise = future::future(objective$eval_many(xss_trafoed), packages = "yahpogym", seed = NULL)
  future::value(promise)
})

