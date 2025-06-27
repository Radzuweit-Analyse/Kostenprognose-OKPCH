# -----------------------------------------------------------------------------
# Multivariate Kalman Forecasting for OKP Health Costs
# -----------------------------------------------------------------------------

library(tidyverse)
library(lubridate)
library(KFAS)
library(zoo)
source("data_utils.R")
source("rmse_utils.R")
source("out_of_sample_plot_utils.R")
source("multivariate_kalman_utils.R")

# ---------------------------- Constants --------------------------------------
FILE_PATH        <- "02_Monitoring-des-couts_Serie-temporelle-trimestre.xlsx"  # raw data

# ------------------------------------------------------------------------------
# Main Routine for Multivariate Evaluation
# ------------------------------------------------------------------------------

#' Run multivariate Kalman pipeline
#'
#' @param file_path Path to raw XLSX dataset
#' @param horizon Forecast horizon (quarters)
#'
#' @return List of results
#' @export
main_multivariate <- function(file_path = FILE_PATH, cost_grp = "Total", h = 8) {
  message("[1/6] Loading and transforming data...")
  raw_data <- load_okp_dataset(file_path)
  mv_data  <- build_multivariate_ts(raw_data, cost_grp)
  Y        <- mv_data$Y
  dates    <- mv_data$dates
  cantons  <- colnames(Y)
  
  message("[2/6] Fitting multivariate Kalman filter model...")
  model <- fit_multivariate_kalman(Y)
  
  message("[3/6] Forecasting ", h, " quarters ahead...")
  fc_df <- forecast_multivariate_kalman(model, h, dates, cantons)
  
  message("[4/6] Preparing actual observations table...")
  actual_df <- as_tibble(Y) %>%
    mutate(Date = dates) %>%
    pivot_longer(-Date, names_to = "Canton", values_to = "Value") %>%
    mutate(Model = "Actual")
  
  message("[5/6] Computing RMSE by canton and horizon...")
  rmse_tbl <- compute_rmse_table(fc_df, actual_df, h, include_canton_column = TRUE)
  
  message("[6/6] Plotting results...")
  # Join forecasts and actuals for plotting
  all_df <- bind_rows(fc_df, actual_df)
  
  fan_plot  <- plot_forecast_paths(all_df, actual_df, facet_by_canton = TRUE)
  rmse_plot <- plot_rmse(all_df, facet_by_canton = TRUE)
  
  print(rmse_plot)
  print(fan_plot)
  
  invisible(list(
    forecasts   = fc_df,
    actual      = actual_df,
    rmse_table  = rmse_tbl,
    rmse_plot   = rmse_plot,
    fan_chart   = fan_plot
  ))
}

# Run the multivariate analysis
main_multivariate()
