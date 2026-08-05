test_that("torch vignettes are self-contained fast examples", {
  vignette_dir <- file.path(testthat::test_path(), "..", "..", "vignettes")
  skip_if_not(dir.exists(vignette_dir),
              "source vignettes are unavailable in the installed package")
  vignette_files <- file.path(
    vignette_dir,
    c("torch-mlp.Rmd", "spline-gam.Rmd", "irl.Rmd")
  )
  if (!all(file.exists(vignette_files))) {
    expect_true(all(file.exists(vignette_files)))
    return(invisible())
  }

  for (vignette_file in vignette_files) {
    contents <- paste(readLines(vignette_file, warn = FALSE), collapse = "\n")

    expected_eval <- if (basename(vignette_file) == "irl.Rmd") "TRUE" else "FALSE"
    expect_match(
      contents,
      paste0("opts_chunk\\$set\\([^\\n]*eval = ", expected_eval)
    )
    expect_no_match(contents, "path/to/")
    expect_match(contents, "nrows = 100L, ncols = 100L")
    expect_match(contents, "canopy =")
    expect_match(contents, "impervious =")
    expect_match(contents, "1000L")
    expect_match(contents, "-1 \\* canopy \\+ 1 \\* impervious")
    expect_match(contents, "n_epochs.*<- (?:[1-9]|[1-2][0-9])L")
    expect_match(contents, "plot\\(results\\$loss_history")
  }

  irl_contents <- paste(
    readLines(file.path(vignette_dir, "irl.Rmd"), warn = FALSE),
    collapse = "\n"
  )
  expect_match(irl_contents, "verify_irl_gradient\\(")
  expect_match(irl_contents, "cfg\\$model_type <- \"irl\"")
  expect_match(irl_contents, "cfg\\$beta <- 1\\.0")
  expect_match(irl_contents, "cfg\\$gamma_d <- 0\\.9")
  expect_match(irl_contents, "cfg\\$n_value_iter <- [1-9][0-9]*L")
  expect_match(irl_contents, "irl_resistance_model\\(")
})
