# Tests for R/resistance_link.R

# ---- constructor -----------------------------------------------------------

test_that("resistance_link constructor creates correct S3 object", {
  lnk <- resistance_link(
    name       = "test",
    forward_fn = function(eta, R_min = 1, R_max = 5000) eta,
    inverse_fn = function(R, R_min = 1, R_max = 5000) R,
    deriv_fn   = function(eta, R_min = 1, R_max = 5000) 1
  )
  expect_s3_class(lnk, "resistance_link")
  expect_named(lnk, c("name", "forward_fn", "inverse_fn", "deriv_fn",
                       "needs_clamp", "eta_fn"))
  expect_true(lnk$needs_clamp)
  expect_null(lnk$eta_fn)
})

test_that("resistance_link rejects non-function forward_fn", {
  expect_error(
    resistance_link("bad",
                    forward_fn = 42L,
                    inverse_fn = function(R, R_min=1, R_max=5000) R,
                    deriv_fn   = function(eta, R_min=1, R_max=5000) 1)
  )
})

test_that("resistance_link rejects non-function inverse_fn", {
  expect_error(
    resistance_link("bad",
                    forward_fn = function(eta, R_min=1, R_max=5000) eta,
                    inverse_fn = "wrong",
                    deriv_fn   = function(eta, R_min=1, R_max=5000) 1)
  )
})

test_that("resistance_link accepts custom eta_fn", {
  custom_eta <- function(theta, basis) theta[1] * rowSums(basis)
  lnk <- resistance_link(
    name       = "custom",
    forward_fn = function(eta, R_min = 1, R_max = 5000) exp(eta),
    inverse_fn = function(R, R_min = 1, R_max = 5000) log(R),
    deriv_fn   = function(eta, R_min = 1, R_max = 5000) exp(eta),
    eta_fn     = custom_eta
  )
  expect_false(is.null(lnk$eta_fn))
  expect_identical(lnk$eta_fn, custom_eta)
})


# ---- link_exp --------------------------------------------------------------

test_that("link_exp forward_fn computes exp(eta)", {
  lnk <- link_exp()
  expect_equal(lnk$forward_fn(log(5)), 5, tolerance = 1e-10)
  expect_equal(lnk$forward_fn(0), 1, tolerance = 1e-10)
})

test_that("link_exp forward_fn clamps to [R_min, R_max]", {
  lnk <- link_exp()
  expect_equal(lnk$forward_fn(-100), 1)      # exp(-100) < R_min=1
  expect_equal(lnk$forward_fn(100), 5000)    # exp(100) > R_max=5000
})

test_that("link_exp inverse_fn is log of clamped R", {
  lnk <- link_exp()
  expect_equal(lnk$inverse_fn(5), log(5), tolerance = 1e-10)
  expect_equal(lnk$inverse_fn(0.001), log(1))   # below R_min clamps to log(1)=0
  expect_equal(lnk$inverse_fn(9999), log(5000)) # above R_max clamps to log(5000)
})

test_that("link_exp roundtrip: inverse(forward(eta)) ~= clamp(exp(eta))", {
  lnk <- link_exp()
  etas <- c(-5, -1, 0, 1, 3, log(5000))
  for (eta in etas) {
    R     <- lnk$forward_fn(eta)
    eta2  <- lnk$inverse_fn(R)
    expect_equal(exp(eta2), R, tolerance = 1e-8,
                 label = paste("roundtrip at eta =", eta))
  }
})

test_that("link_exp deriv_fn is 0 at R boundaries", {
  lnk <- link_exp()
  # At R_min = 1: eta = log(1) = 0; exp(0) == 1 exactly on boundary
  expect_equal(lnk$deriv_fn(log(1) - 1e-3), 0)  # below R_min
  # At R_max = 5000: eta = log(5000); exp(eta) == 5000 on upper boundary -> 0
  expect_equal(lnk$deriv_fn(log(5000) + 0.01), 0)  # above R_max
})

test_that("link_exp deriv_fn equals exp(eta) inside bounds", {
  lnk <- link_exp()
  eta <- 3  # exp(3) ~ 20, well inside [1, 5000]
  expect_equal(lnk$deriv_fn(eta), exp(eta), tolerance = 1e-10)
})

test_that("link_exp has needs_clamp = TRUE", {
  expect_true(link_exp()$needs_clamp)
})


# ---- link_softplus ---------------------------------------------------------

test_that("link_softplus forward_fn uses log1p(exp(eta)) for small eta", {
  lnk <- link_softplus()
  eta <- 1
  expected <- log1p(exp(1)) + 1   # R_min = 1
  expect_equal(lnk$forward_fn(eta), expected, tolerance = 1e-10)
})

test_that("link_softplus forward_fn is numerically stable for large eta", {
  lnk <- link_softplus()
  # eta = 30 >> 20: should use eta + R_min path, not log1p(exp(30)) which
  # would still work here but we confirm no Inf
  result <- lnk$forward_fn(30)
  expect_true(is.finite(result))
  # Stable path gives eta + R_min clamped at R_max = 5000
  expected_raw <- 30 + 1  # = 31
  expect_equal(result, min(expected_raw, 5000), tolerance = 1e-10)
})

test_that("link_softplus forward_fn returns finite values for extreme inputs", {
  lnk <- link_softplus()
  expect_true(is.finite(lnk$forward_fn(-100)))
  expect_true(is.finite(lnk$forward_fn(1000)))
})

test_that("link_softplus roundtrip: inverse(forward(eta)) ~= eta", {
  lnk  <- link_softplus()
  etas <- c(-3, 0, 1, 5, 15, 25)
  for (eta in etas) {
    R    <- lnk$forward_fn(eta)
    eta2 <- lnk$inverse_fn(R)
    expect_equal(eta2, eta, tolerance = 1e-6,
                 label = paste("softplus roundtrip at eta =", eta))
  }
})

test_that("link_softplus deriv_fn is sigmoid at eta = 0", {
  lnk <- link_softplus()
  expect_equal(lnk$deriv_fn(0), 0.5, tolerance = 1e-10)
})

test_that("link_softplus deriv_fn is 0 when R hits R_max", {
  lnk <- link_softplus()
  # When forward_fn(eta) == R_max, derivative should be 0
  # R_max = 5000; eta such that eta + 1 >= 5000 (stable path), i.e. eta >= 4999
  expect_equal(lnk$deriv_fn(5000), 0)
})

test_that("link_softplus has needs_clamp = FALSE", {
  expect_false(link_softplus()$needs_clamp)
})


# ---- link_power ------------------------------------------------------------

test_that("link_power forward_fn computes |eta|^p for p=2", {
  lnk <- link_power(p = 2)
  expect_equal(lnk$forward_fn(3), 9, tolerance = 1e-10)  # 3^2 = 9
  expect_equal(lnk$forward_fn(-3), 9, tolerance = 1e-10) # |-3|^2 = 9
})

test_that("link_power forward_fn clamps to [R_min, R_max]", {
  lnk <- link_power(p = 2)
  expect_equal(lnk$forward_fn(0), 1)       # 0^2 = 0 < R_min, clamp to 1
  expect_equal(lnk$forward_fn(200), 5000)  # 200^2 >> R_max, clamp to 5000
})

test_that("link_power roundtrip for p = 2", {
  lnk  <- link_power(p = 2)
  etas <- c(1, 2, 5, 10)
  for (eta in etas) {
    R    <- lnk$forward_fn(eta)
    eta2 <- lnk$inverse_fn(R)
    # inverse gives R^(1/2), only recovers positive eta
    expect_equal(eta2, eta, tolerance = 1e-8,
                 label = paste("power(2) roundtrip at eta =", eta))
  }
})

test_that("link_power works for non-standard p values", {
  lnk_half  <- link_power(p = 0.5)
  lnk_cubic <- link_power(p = 3)

  expect_equal(lnk_half$forward_fn(4), sqrt(4), tolerance = 1e-10)   # 4^0.5 = 2
  expect_equal(lnk_cubic$forward_fn(2), 8, tolerance = 1e-10)         # 2^3 = 8

  expect_true(is.finite(lnk_half$deriv_fn(4)))
  expect_true(is.finite(lnk_cubic$deriv_fn(2)))
})

test_that("link_power deriv_fn has correct sign for negative eta", {
  lnk <- link_power(p = 2)
  d_pos <- lnk$deriv_fn(3)   # 2 * 3^1 * 1 = 6
  d_neg <- lnk$deriv_fn(-3)  # 2 * 3^1 * (-1) = -6
  expect_equal(d_pos, 6, tolerance = 1e-8)
  expect_equal(d_neg, -6, tolerance = 1e-8)
})

test_that("link_power deriv_fn is 0 outside bounds", {
  lnk <- link_power(p = 2)
  expect_equal(lnk$deriv_fn(0.1), 0)   # 0.1^2 = 0.01 < R_min = 1
  expect_equal(lnk$deriv_fn(200), 0)   # 200^2 >> R_max = 5000
})

test_that("link_power rejects non-positive p", {
  expect_error(link_power(p = 0))
  expect_error(link_power(p = -1))
})

test_that("link_power has needs_clamp = TRUE", {
  expect_true(link_power()$needs_clamp)
})


# ---- link_identity ---------------------------------------------------------

test_that("link_identity forward_fn clamps eta to [R_min, R_max]", {
  lnk <- link_identity()
  expect_equal(lnk$forward_fn(50), 50)
  expect_equal(lnk$forward_fn(-5), 1)
  expect_equal(lnk$forward_fn(9999), 5000)
})

test_that("link_identity deriv_fn is 1 inside bounds, 0 outside", {
  lnk <- link_identity()
  expect_equal(lnk$deriv_fn(100), 1)      # inside [1, 5000]
  expect_equal(lnk$deriv_fn(-1), 0)       # below R_min
  expect_equal(lnk$deriv_fn(10000), 0)    # above R_max
})

test_that("link_identity roundtrip is identity inside bounds", {
  lnk <- link_identity()
  for (eta in c(1, 10, 100, 1000, 5000)) {
    R <- lnk$forward_fn(eta)
    expect_equal(lnk$inverse_fn(R), R, tolerance = 1e-10)
  }
})

test_that("link_identity has needs_clamp = TRUE", {
  expect_true(link_identity()$needs_clamp)
})


# ---- compute_eta -----------------------------------------------------------

test_that("compute_eta with matrix basis uses standard additive formula", {
  lnk         <- link_exp()
  theta       <- c(0.5, 1.0, -0.5)
  basis_mat   <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2)
  expected    <- theta[1] + basis_mat %*% theta[-1]
  result      <- DiffiScape:::compute_eta(lnk, theta, basis_mat)
  expect_equal(as.numeric(result), as.numeric(expected), tolerance = 1e-10)
})

test_that("compute_eta with custom eta_fn calls the override", {
  called <- FALSE
  custom_eta <- function(theta, basis) {
    called <<- TRUE
    theta[1] + rowSums(basis) * theta[2]
  }
  lnk <- resistance_link(
    name       = "custom",
    forward_fn = function(eta, R_min = 1, R_max = 5000) exp(eta),
    inverse_fn = function(R, R_min = 1, R_max = 5000) log(R),
    deriv_fn   = function(eta, R_min = 1, R_max = 5000) exp(eta),
    eta_fn     = custom_eta
  )
  # matrix(1:6, nrow=3) fills col-wise: col1=[1,2,3], col2=[4,5,6]
  # rowSums = [5, 7, 9]; theta[1]+rowSums*theta[2] = [1+10, 1+14, 1+18]
  basis_mat <- matrix(1:6, nrow = 3)
  result    <- DiffiScape:::compute_eta(lnk, c(1, 2), basis_mat)
  expect_true(called)
  expect_equal(result, c(1 + 5 * 2, 1 + 7 * 2, 1 + 9 * 2))
})

test_that("compute_eta with SpatRaster basis adds layers correctly", {
  skip_if_not_installed("terra")
  lnk <- link_exp()

  r1 <- terra::rast(nrows = 2, ncols = 2, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  r2 <- terra::rast(nrows = 2, ncols = 2, xmin = 0, xmax = 1,
                    ymin = 0, ymax = 1)
  terra::values(r1) <- c(1, 2, 3, 4)
  terra::values(r2) <- c(0.1, 0.2, 0.3, 0.4)
  basis_rast <- c(r1, r2)

  theta  <- c(0.5, 1.0, -0.5)
  result <- DiffiScape:::compute_eta(lnk, theta, basis_rast)

  expect_s4_class(result, "SpatRaster")
  vals <- terra::values(result)[, 1]
  expected <- 0.5 + 1.0 * c(1, 2, 3, 4) + (-0.5) * c(0.1, 0.2, 0.3, 0.4)
  expect_equal(vals, expected, tolerance = 1e-8)
})
