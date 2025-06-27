#' Fit benchmark models and produce h‑step forecasts
#'
#' @param train_y `ts` – training window (≥ 8 observations).
#' @param h       `integer(1)` – forecast horizon in quarters.
#' @param train_x Optional numeric vector of the same length as `train_y` to
#'   be used as a regression term (e.g., COVID dummy).
#'
#' @return A **named list** of numeric vectors (`Kalman`, `ARMA`, `RW`) each of
#'   length `h`.  Elements are filled with `NA_real_` if model estimation fails.
#'
#' @examples
#' y <- AirPassengers ; h <- 4
#' fit_and_forecast(y, h)
#' @keywords internal
fit_and_forecast <- function(train_y, h, train_x = NULL) {
  # --- internal helper: wrap estimator in try‑catch -------------------------
  safe_fc <- function(expr) tryCatch(expr, error = function(e) rep(NA_real_, h))
  
  # Seasonal ARIMA -----------------------------------------------------------
  arma_fc <- safe_fc({
    fit <- Arima(train_y,
                 order    = c(1, 1, 1),
                 seasonal = list(order = c(1, 0, 1), period = 4),
                 include.drift = TRUE)
    forecast(fit, h = h)$mean
  })
  
  # Random Walk with drift ---------------------------------------------------
  rw_fc <- safe_fc({
    fit <- Arima(train_y, order = c(0, 1, 0), include.drift = TRUE)
    forecast(fit, h = h)$mean
  })
  
  # Structural‑time‑series (local‑level + seasonal) --------------------------
  kalman_fc <- safe_fc({
    if (!is.null(train_x)) {
      model <- SSModel(train_y ~ train_x +
                         SSMtrend(2, Q = list(NA, NA)) +
                         SSMseasonal(period = 4, sea.type = "dummy", Q = NA),
                       H = NA)
      fit <- fitSSM(model, inits = rep(log(var(train_y) * 10), 4))
      future_x <- rep(0, h)
      as.numeric(predict(fit$model, n.ahead = h,
                         newdata = data.frame(train_x = future_x)))
    } else {
      model <- SSModel(train_y ~ SSMtrend(2, Q = list(NA, NA)) +
                         SSMseasonal(period = 4, sea.type = "dummy", Q = NA),
                       H = NA)
      fit <- fitSSM(model, inits = rep(log(var(train_y)), 4))
      as.numeric(predict(fit$model, n.ahead = h))
    }
  })
  
  list(Kalman = kalman_fc, ARMA = arma_fc, RW = rw_fc)
}

#' Rolling‑origin forecasts over historical span
#'
#' @param y     `ts` – full observed series.
#' @param dates `Date` – companion date vector.
#' @param h     `integer(1)` – forecast horizon (quarters).
#' @param x     Optional numeric vector aligned with `y` used as regression
#'   input.
#'   
#' @return A tidy `tibble` with columns *Date*, *Model*, *Value*, *Origin*.
#' @export
rolling_origin_forecasts <- function(y, dates, h, x = NULL) {
  purrr::map_dfr(seq_len(length(y) - h), function(i) {
    train_y <- window(y, end = time(y)[i])
    train_x <- if (!is.null(x)) window(x, end = time(y)[i]) else NULL
    if (length(train_y) < 8) return(NULL)  # skip initial burn‑in
    
    origin  <- dates[i]
    fc      <- fit_and_forecast(train_y, h, train_x)
    fut_dt  <- dates[(i + 1):(i + h)]
    last_obs <- tail(train_y, 1)
    
    bind_rows(
      tibble(Date = c(origin, fut_dt), Model = "Kalman", Value = c(last_obs, fc$Kalman), Origin = origin),
      tibble(Date = c(origin, fut_dt), Model = "ARMA",   Value = c(last_obs, fc$ARMA),   Origin = origin),
      tibble(Date = c(origin, fut_dt), Model = "RW",     Value = c(last_obs, fc$RW),     Origin = origin)
    )
  })
}

#' Forecast from the most recent *h* origins (produces future fan chart)
#'
#' @inheritParams rolling_origin_forecasts
#' @return Same format as `rolling_origin_forecasts()` but limited to the last
#'   `h` origins.
#' @export
final_forecasts <- function(y, dates, h, x = NULL) {
  tail_idx <- (length(y) - h + 1):length(y)
  purrr::map_dfr(tail_idx, function(i) {
    train_y <- window(y, end = time(y)[i])
    train_x <- if (!is.null(x)) window(x, end = time(y)[i]) else NULL
    origin  <- dates[i]
    fc      <- fit_and_forecast(train_y, h, train_x)
    fut_dt  <- seq(from = origin + months(3), by = "quarter", length.out = h)
    last_obs<- tail(train_y, 1)
    
    bind_rows(
      tibble(Date = c(origin, fut_dt), Model = "Kalman", Value = c(last_obs, fc$Kalman), Origin = origin),
      tibble(Date = c(origin, fut_dt), Model = "ARMA",   Value = c(last_obs, fc$ARMA),   Origin = origin),
      tibble(Date = c(origin, fut_dt), Model = "RW",     Value = c(last_obs, fc$RW),     Origin = origin)
    )
  })
}
