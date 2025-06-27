# -----------------------------------------------------------------------------
# OKP Healthcare Cost Forecasting – Nonlinear Kalman Filter Version
# -----------------------------------------------------------------------------

library(tidyverse)
library(lubridate)
library(zoo)
library(readxl)
library(bssm)

source("R/data_utils.R")

FILE_PATH        <- "02_Monitoring-des-couts_Serie-temporelle-trimestre.xlsx"
FORECAST_HORIZON <- 8
CANTON           <- "Suisse"
COST_GROUP       <- "Total"

raw_data  <- load_okp_dataset(FILE_PATH)
canton_df <- extract_series(raw_data)
series    <- build_quarterly_ts(canton_df)
y         <- series$ts
dates     <- series$dates
y_raw     <- as.numeric(y)
stopifnot(is.numeric(y_raw), is.vector(y_raw), !anyNA(y_raw))
n         <- length(y_raw)

state_dim   <- 2
init_theta  <- rep(log(sd(y_raw)), 2)

obs_fn <- function(t, state, theta, ...) {
  exp(state[1])
}

obs_grad <- function(t, state, theta, ...) {
  matrix(c(exp(state[1]), 0), nrow = 1)
}

T_grad <- function(t, state, theta, ...) {
  diag(2)
}

log_prior_pdf <- function(theta) {
  0
}

model <- ssm_nlg(
  y = y_raw,
  Z = obs_fn,
  Z_gn = obs_grad,
  T = matrix(c(1, 1, 0, 1), 2, 2),
  T_gn = T_grad,
  R = function(theta) matrix(c(0, exp(theta[2])), 2, 1),
  a1 = c(0, 0),
  P1 = diag(1e4, 2),
  H = function(theta) matrix(exp(theta[1])),
  theta = init_theta,
  log_prior_pdf = log_prior_pdf,
  state_names = c("level", "slope"),
  n_states = 2
)


set.seed(123)
fit <- run_mcmc(model, iter = 2000, particles = 1000, burnin = 500)

summary_fit    <- summary(fit, return_se = TRUE)
smoothed_vals  <- apply(fitted(fit), 2, mean)
smoothed_df    <- tibble(Date = dates, Value = smoothed_vals, Model = "Nonlinear_Kalman")
assign("nonlinear_kalman_smoothed", smoothed_df, envir = .GlobalEnv)

nsim            <- 1000
forecast_samples <- predict(fit, particles = nsim, future = TRUE, ahead = FORECAST_HORIZON)
forecast_means   <- apply(forecast_samples, 2, mean)
forecast_dates   <- seq(from = tail(dates, 1) + months(3), by = "quarter", length.out = FORECAST_HORIZON)
forecast_df      <- tibble(Date = forecast_dates, Value = forecast_means, Model = "Nonlinear_Kalman_Forecast")
assign("nonlinear_kalman_forecast", forecast_df, envir = .GlobalEnv)

message("Nonlinear Kalman Filter (ssm_nlg) smoothing and forecasting complete.\nSmoothed data saved to `nonlinear_kalman_smoothed`, forecast to `nonlinear_kalman_forecast`.")
