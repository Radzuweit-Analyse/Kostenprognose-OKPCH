#' -----------------------------------------------------------------------------
#' OKP Healthcare Cost Forecasting – Modular Evaluation Script
#' -----------------------------------------------------------------------------
#' @file        okp_forecast_evaluation.R
#' @author      Raphaël Radzuweit
#' @date        2025‑05‑18
#' @license     Proprietary License, file LICENSE
#'
#' @title Out‑of‑Sample Forecast Evaluation for Swiss Healthcare Costs
#'
#' @description
#' This script performs a rigorous, rolling‑origin evaluation of quarterly
#' healthcare costs per insured in Switzerland (OKP system). Three benchmark
#' models are compared across an eight‑quarter horizon:
#'
#' * **Seasonal ARIMA** – (1,1,1)(1,0,1)[4] with drift
#' * **Random Walk with Drift** – ARIMA(0,1,0)+drift
#' * **Structural Time Series** – local‑level + deterministic quarterly seasonality
#'   estimated via **KFAS** (Kalman filter / smoother)
#'
#' The codebase is fully modular – each logical step (data ingestion, model
#' fitting, forecasting, evaluation, plotting) lives in a self‑contained,
#' unit‑testable function and is documented using **roxygen2** tags.  Running
#' `main()` executes the entire pipeline and produces: (i) a tidy RMSE table
#' and (ii) two diagnostic plots (RMSE‑by‑horizon line chart, fan chart of all
#' forecast trajectories).
#'
#' ## Usage
#' ```r
#' source("okp_forecast_evaluation.R")  # brings all functions into scope
#' main()                                # run end‑to‑end study
#' ```
#'
#' ## Session Info (for reproducibility)
#' * R version 4.4.0 (nickname: "Puppy Cup")
#' * tidyverse 2.0.0
#' * forecast  8.23
#' * KFAS      1.5‑3
#' * zoo       1.8‑12
#' -----------------------------------------------------------------------------

# ---------------------------- Libraries --------------------------------------
# Use explicit calls so `renv::snapshot()` captures them.
library(tidyverse)    # dplyr, purrr, ggplot2, tidyr
library(lubridate)    # date manipulation helpers
library(zoo)          # yearqtr ⇄ Date conversion
library(readxl)       # XLSX import
library(forecast)     # ARIMA / naïve benchmarks
library(KFAS)         # structural time‑series & Kalman smoother
source("rmse_utils.R")
source("out_of_sample_plot_utils.R")
source("data_utils.R")
source("multivariate_kalman_utils.R")

# ---------------------------- Constants --------------------------------------
FILE_PATH        <- "02_Monitoring-des-couts_Serie-temporelle-trimestre.xlsx"  # raw data
FORECAST_HORIZON <- 8      # eight quarters ≙ two years
CANTON           <- "Suisse"
COST_GROUP       <- "Total"  # default cost basket

# Corporate visual identity (neutral grey scale, printer‑friendly)
GREY_PALETTE <- c(Kalman = "#444444", ARMA = "#888888", RW = "#BBBBBB")
LINETYPES    <- c(Kalman = "solid", ARMA = "dashed", RW = "dotted", Actual = "solid")

#' Fit benchmark models and produce h‑step forecasts
#'
#' @param train_y `ts` – training window (≥ 8 observations).
#' @param h       `integer(1)` – forecast horizon in quarters.
#'
#' @return A **named list** of numeric vectors (`Kalman`, `ARMA`, `RW`) each of
#'   length `h`.  Elements are filled with `NA_real_` if model estimation fails.
#'
#' @examples
#' y <- AirPassengers ; h <- 4
#' fit_and_forecast(y, h)
#' @keywords internal
fit_and_forecast <- function(train_y, h) {
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
    model <- SSModel(train_y ~ SSMtrend(2, Q = list(NA, NA)) +
                       SSMseasonal(period = 4, sea.type = "dummy", Q = NA),
                     H = NA)
    fit <- fitSSM(model, inits = rep(log(var(train_y)), 4))
    as.numeric(predict(fit$model, n.ahead = h))
  })
  
  list(Kalman = kalman_fc, ARMA = arma_fc, RW = rw_fc)
}

#' Rolling‑origin forecasts over historical span
#'
#' @param y     `ts` – full observed series.
#' @param dates `Date` – companion date vector.
#' @param h     `integer(1)` – forecast horizon (quarters).
#'
#' @return A tidy `tibble` with columns *Date*, *Model*, *Value*, *Origin*.
#' @export
rolling_origin_forecasts <- function(y, dates, h) {
  purrr::map_dfr(seq_len(length(y) - h), function(i) {
    train_y <- window(y, end = time(y)[i])
    if (length(train_y) < 8) return(NULL)  # skip initial burn‑in
    
    origin  <- dates[i]
    fc      <- fit_and_forecast(train_y, h)
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
final_forecasts <- function(y, dates, h) {
  tail_idx <- (length(y) - h + 1):length(y)
  purrr::map_dfr(tail_idx, function(i) {
    train_y <- window(y, end = time(y)[i])
    origin  <- dates[i]
    fc      <- fit_and_forecast(train_y, h)
    fut_dt  <- seq(from = origin + months(3), by = "quarter", length.out = h)
    last_obs<- tail(train_y, 1)
    
    bind_rows(
      tibble(Date = c(origin, fut_dt), Model = "Kalman", Value = c(last_obs, fc$Kalman), Origin = origin),
      tibble(Date = c(origin, fut_dt), Model = "ARMA",   Value = c(last_obs, fc$ARMA),   Origin = origin),
      tibble(Date = c(origin, fut_dt), Model = "RW",     Value = c(last_obs, fc$RW),     Origin = origin)
    )
  })
}

#' ------------------------- Main Orchestration ------------------------------
#' @title End‑to‑End Pipeline
#' @description Execute data loading, model estimation, evaluation, and
#'   diagnostic plotting in a single call.
#' @return Invisibly returns a list with components `rmse_table`, `rmse_plot`,
#'   and `fan_chart` for further manipulation.
#' @export
main <- function(file_path = FILE_PATH,
                 horizon    = FORECAST_HORIZON,
                 verbose    = TRUE,
                 plots      = TRUE) {
  # ---------------------------------------------------------------------------
  # 1 ▸ Data ingest ------------------------------------------------------------
  # ---------------------------------------------------------------------------
  raw_data <- load_okp_dataset(file_path)
  canton_df<- extract_series(raw_data)
  series   <- build_quarterly_ts(canton_df)
  y      <- series$ts
  dates  <- series$dates
  
  # ---------------------------------------------------------------------------
  # ▸ Kalman smoother to estimate actual series (2016–2024) -------------------
  # ---------------------------------------------------------------------------
  model_kalman <- SSModel(y ~ SSMtrend(2, Q = list(NA, NA)) +
                            SSMseasonal(period = 4, sea.type = "dummy", Q = NA),
                          H = NA)
  fit_kalman <- fitSSM(model_kalman, inits = rep(log(var(y)), 4))
  smooth_kalman <- KFS(fit_kalman$model, smoothing = "state")
  
  # Extract smoothed level + seasonality
  smoothed_vals <- rowSums(smooth_kalman$alphahat)
  kalman_actual_estimates <- tibble(
    Date = dates,
    Value = as.numeric(smoothed_vals),
    Model = "Kalman_Smoothed"
  )
  
  # Save to global environment
  assign("kalman_smoothed", kalman_actual_estimates, envir = .GlobalEnv)
  
  # ---------------------------------------------------------------------------
  # 2 ▸ Forecast generation (historical roll + future fan) --------------------
  # ---------------------------------------------------------------------------
  fc_hist <- rolling_origin_forecasts(y, dates, horizon)
  fc_fut  <- final_forecasts(y, dates, horizon)
  forecast_df <- dplyr::bind_rows(fc_hist, fc_fut)
  assign("kalman_forecast", forecast_df, envir = .GlobalEnv)
  
  # ---------------------------------------------------------------------------
  # 3 ▸ Actual observations table --------------------------------------------
  # ---------------------------------------------------------------------------
  actual_df <- tibble(Date = dates,
                      Value = as.numeric(y),
                      Model = "Actual")
  
  # test in growht term
  # Convert both forecast and actual series to growth terms
  # Apply growth transformation to actuals
  actual_growth <- actual_df %>%
    arrange(Date) %>%
    mutate(Value = Value - lag(Value)) %>%
    filter(!is.na(Value))
  
  # Apply growth transformation to forecasts
  forecast_growth <- forecast_df %>%
    arrange(Model, Origin, Date) %>%
    group_by(Model, Origin) %>%
    mutate(Value = Value - lag(Value)) %>%
    filter(!is.na(Value)) %>%
    ungroup()
  
  fan_chart <- plot_forecast_paths(forecast_growth, actual_growth)
  
  # ---------------------------------------------------------------------------
  # 4 ▸ RMSE evaluation -------------------------------------------------------
  # ---------------------------------------------------------------------------
  rmse_tbl <- compute_rmse_table(forecast_df, actual_df, horizon)
  
  # ---------------------------------------------------------------------------
  # 5 ▸ Prepare scored long table (for plotting) ------------------------------
  # ---------------------------------------------------------------------------
  scored_long <- forecast_df %>%
    dplyr::left_join(actual_df %>% rename(True = Value) %>% dplyr::select(Date, True), by = "Date") %>%
    dplyr::mutate(Horizon = as.integer((Date - Origin) / lubridate::dmonths(3))) %>%
    dplyr::filter(dplyr::between(Horizon, 1, horizon))
  
  # ---------------------------------------------------------------------------
  # 6 ▸ Diagnostics plots -----------------------------------------------------
  # ---------------------------------------------------------------------------
  rmse_plot <- plot_rmse(scored_long)
  fan_chart <- plot_forecast_paths(forecast_df, actual_df)
  
  # ---------------------------------------------------------------------------
  # 7 ▸ Output handling -------------------------------------------------------
  # ---------------------------------------------------------------------------
  if (verbose) print(rmse_tbl)
  if (plots) {
    print(rmse_plot)
    print(fan_chart)
  }
  
  invisible(list(
    rmse_table = rmse_tbl,
    rmse_plot  = rmse_plot,
    fan_chart  = fan_chart,
    forecasts  = forecast_df,
    actual     = actual_df
  ))
}
main()
