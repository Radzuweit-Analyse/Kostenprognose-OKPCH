#' Fit multivariate structural time-series model
#'
#' @param Y Matrix of observations (T × N cantons)
#'
#' @return Fitted KFAS model
#'
#' @export
fit_multivariate_kalman <- function(Y) {
  na_ratio <- colMeans(is.na(Y))
  Y <- Y[, na_ratio < 0.3]
  
  n_cantons <- ncol(Y)
  if (n_cantons == 0) stop("No canton has sufficient data (less than 30% missing).")
  
  model <- SSModel(Y ~ SSMtrend(1, Q = matrix(NA)) +
                     SSMseasonal(period = 4, sea.type = "dummy", Q = matrix(NA)),
                   H = diag(NA, n_cantons))
  
  y_sd <- apply(Y, 2, sd, na.rm = TRUE)
  init_vals <- log(c(median(y_sd, na.rm = TRUE)^2,     # Q (level)
                     median(y_sd, na.rm = TRUE)^2,     # Q (seasonal)
                     y_sd^2))                          # H diagonals
  
  fit <- fitSSM(model, inits = init_vals, method = "BFGS")
  
  stopifnot(is.SSModel(fit$model))
  return(fit$model)
}
#' Forecast multivariate Kalman model
#'
#' @param model Fitted SSModel
#' @param h Forecast horizon (quarters)
#'
#' @return Tibble with Date, Canton, Value, Model, Origin
#'
#' @export
forecast_multivariate_kalman <- function(model, h, dates, cantons) {
  if (any(is.na(dates))) stop("dates contain NA values. Cannot compute forecast horizon.")
  origin <- tail(dates, 1)
  if (is.na(origin)) stop("Last date is NA. Check input.")
  
  fc <- predict(model, n.ahead = h, interval = "none")
  fut_dates <- seq(from = origin + months(3), by = "quarter", length.out = h)
  
  purrr::map_dfr(seq_along(cantons), function(i) {
    tibble(Date = fut_dates,
           Canton = cantons[i],
           Value = fc[, i],
           Model = "MultivariateKalman",
           Origin = origin)
  })
}
