test_that("migration vignettes use installed core operations and synthetic data", {
  root <- file.path(testthat::test_path(), "..", "..", "vignettes")
  skip_if_not(dir.exists(root), "source vignettes unavailable in installed package")
  for (path in list.files(root, pattern = "Rmd$", full.names = TRUE)) {
    contents <- paste(readLines(path), collapse = "\n")
    expect_match(contents, "RETICULATE_PYTHON")
    expect_match(contents, "xyFromCell")
    expect_no_match(contents, "torch_module|ds_torch_setup|path/to/|pip install torch")
  }
})
