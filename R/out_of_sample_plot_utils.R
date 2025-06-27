#' -----------------------------------------------------------------------------
#' Forecast Path Plot Utilities for OKP Forecasting
#' -----------------------------------------------------------------------------

library(tidyverse)
library(lubridate)

# Corporate style
GREY_PALETTE <- c(Kalman = "#444444", ARMA = "#888888", RW = "#BBBBBB")
LINETYPES    <- c(Kalman = "solid", ARMA = "dashed", RW = "dotted", Actual = "solid")

# ------------------------------------------------------------------------------
# Faceted Forecast Paths Plot
# ------------------------------------------------------------------------------

#' Plot forecast trajectories overlaid on actuals, with optional faceting
#'
#' @param forecasts `tibble` – forecast data with columns: `Date`, `Value`, `Model`, `Origin`, optionally `Canton`.
#' @param actual    `tibble` – actual data with columns: `Date`, `Value`, optionally `Canton`.
#' @param facet_by_canton `logical(1)` – If TRUE, facet the plot by canton.
#'
#' @return A ggplot object.
#' @export
plot_forecast_paths <- function(forecasts, actual, facet_by_canton = FALSE) {
  base_plot <- ggplot() +
    geom_line(data = forecasts,
              aes(Date, Value,
                  group = interaction(Model, Origin),
                  colour = Model, linetype = Model),
              linewidth = 0.8, alpha = 0.45) +
    geom_line(data = actual,
              aes(Date, Value),
              colour = "black", linewidth = 1.1) +
    labs(title = "Forecast Paths vs. Actuals",
         x = "Quarter", y = "Cost per Insured (CHF)") +
    theme_minimal() +
    scale_colour_manual(values = GREY_PALETTE) +
    scale_linetype_manual(values = LINETYPES)
  
  if (facet_by_canton) {
    base_plot <- base_plot + facet_wrap(~Canton, scales = "free_y", ncol = 5)
  }
  
  return(base_plot)
}
