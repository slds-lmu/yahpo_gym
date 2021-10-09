test_that("benchmarkset can be instantiated", {
  b = BenchmarkSet$new("lcbench")
  obj = b$get_objective()
})