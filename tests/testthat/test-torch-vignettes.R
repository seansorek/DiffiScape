test_that("torch vignettes are self-contained fast examples", {
  vignette_dir <- file.path(testthat::test_path(), "..", "..", "vignettes")
  vignette_files <- file.path(vignette_dir, c("torch-mlp.Rmd", "spline-gam.Rmd"))
  skip_if_not(all(file.exists(vignette_files)),
              "source vignettes are unavailable in the installed package")

  for (vignette_file in vignette_files) {
    contents <- paste(readLines(vignette_file, warn = FALSE), collapse = "\n")

    expect_match(contents, "opts_chunk\\$set\\([^\\n]*eval = FALSE")
    expect_no_match(contents, "path/to/")
    expect_match(contents, "nrows = 100L, ncols = 100L")
    expect_match(contents, "canopy =")
    expect_match(contents, "impervious =")
    expect_match(contents, "1000L")
    expect_match(contents, "-1 \\* canopy \\+ 1 \\* impervious")
    expect_match(contents, "n_epochs.*<- (?:[1-9]|[1-2][0-9])L")
    expect_match(contents, "plot\\(results\\$loss_history")
  }
})
