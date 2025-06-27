#' -----------------------------------------------------------------------------
#' RMSE Diagnostics Utilities for OKP Forecast Evaluation
#' -----------------------------------------------------------------------------

library(tidyverse)
library(lubridate)

# Custom GREY palette and linetypes for visual identity
GREY_PALETTE <- c(Kalman = "#444444", ARMA = "#888888", RW = "#BBBBBB")
LINETYPES    <- c(Kalman = "solid", ARMA = "dashed", RW = "dotted", Actual = "solid")

# ------------------------------------------------------------------------------
# Compute RMSE Table with Optional Canton Column
# ------------------------------------------------------------------------------

#' Compute RMSE table, optionally grouped by canton
#'
#' @param forecasts `tibble` – Forecast data with `Date`, `Model`, `Value`, `Origin`, and optionally `Canton`.
#' @param actual    `tibble` – Ground truth with `Date`, `Value`, and optionally `Canton`.
#' @param h         `integer(1)` – Forecast horizon (number of quarters).
#' @param include_canton_column `logical(1)` – If TRUE, include `Canton` in output.
#'
#' @return A wide-format tibble with RMSE by horizon, optionally grouped by canton.
#' @export
compute_rmse_table <- function(forecasts, actual, h, include_canton_column = FALSE) {
  actual <- actual %>%
    rename(True = Value)
  
  if (include_canton_column && "Canton" %in% names(actual)) {
    actual <- actual %>% dplyr::select(Date, True, Canton)
  } else {
    actual <- actual %>% dplyr::select(Date, True)
  }
  
  scored <- forecasts %>%
    left_join(actual, by = "Date") %>%
    mutate(Horizon = as.integer((Date - Origin) / dmonths(3))) %>%
    filter(between(Horizon, 1, h), !is.na(True))
  
  grouping_vars <- if (include_canton_column) c("Canton", "Model", "Horizon") else c("Model", "Horizon")
  
  scored %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(RMSE = sqrt(mean((True - Value)^2, na.rm = TRUE)), .groups = "drop") %>%
    pivot_wider(names_from = Horizon, names_prefix = "horizon_", values_from = RMSE) %>%
    mutate(Avg_RMSE = rowMeans(dplyr::select(., starts_with("horizon_")), na.rm = TRUE)) %>%
    arrange(Avg_RMSE)
}

# ------------------------------------------------------------------------------
# Faceted RMSE Line Plot (One Panel per Canton)
# ------------------------------------------------------------------------------

#' Line plot of RMSE by forecast horizon, optionally faceted by canton
#'
#' @param scored `tibble` – Scored forecasts with columns: `Model`, `Horizon`, `True`, `Value`, optionally `Canton`.
#' @param facet_by_canton `logical(1)` – If TRUE, plot one graph per canton.
#'
#' @return A ggplot object.
#' @export
plot_rmse <- function(scored, facet_by_canton = FALSE) {
  rmse_summary <- scored %>%
    mutate(Horizon = as.integer((Date - Origin) / dmonths(3))) %>%
    filter(between(Horizon, 1, 8), !is.na(True)) %>%
    group_by(across(c(if (facet_by_canton) "Canton", "Model", "Horizon"))) %>%
    summarise(RMSE = sqrt(mean((True - Value)^2, na.rm = TRUE)), .groups = "drop")
  
  p <- ggplot(rmse_summary, aes(Horizon, RMSE, colour = Model, linetype = Model)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2) +
    labs(title = "RMSE by Forecast Horizon",
         x = "Horizon (quarters ahead)", y = "RMSE") +
    theme_minimal() +
    scale_colour_manual(values = GREY_PALETTE) +
    scale_linetype_manual(values = LINETYPES)
  
  if (facet_by_canton) {
    p <- p + facet_wrap(~Canton, scales = "free_y", ncol = 5)
  }
  
  return(p)
}
