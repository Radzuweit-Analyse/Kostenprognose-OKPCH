#' -----------------------------------------------------------------------------
#' OKP Data Utilities – Load, Filter, and Transform Health Cost Data
#' -----------------------------------------------------------------------------

library(tidyverse)
library(lubridate)
library(zoo)
library(readxl)

# ------------------------------------------------------------------------------
# Load the raw OKP quarterly monitoring dataset (all cantons)
# ------------------------------------------------------------------------------

#' Load raw quarterly OKP data from Excel
#'
#' @param file_path `character(1)` – Path to the XLSX file.
#'
#' @return A tibble with columns:
#'   * `Canton` – French name of canton (e.g. "Vaud", "Zurich") or "Suisse"
#'   * `Groupe_de_couts` – Cost basket
#'   * `Date` – Quarter end date (class `Date`)
#'   * `Prestations_brutes_par_assure` – CHF per insured
#'
#' @export
load_okp_dataset <- function(file_path) {
  readxl::read_excel(file_path, sheet = "Data") %>%
    mutate(Date = as.Date(Date)) %>%
    arrange(Canton, Date)
}

# ------------------------------------------------------------------------------
# Extract a canton/cost-group specific cost series
# ------------------------------------------------------------------------------

#' Extract data for a specific canton and cost group
#'
#' @param data      Tibble from `load_okp_dataset()`.
#' @param canton    `character(1)` – French name of canton or "Suisse".
#' @param cost_grp  `character(1)` – Cost group label (default = "Total").
#'
#' @return Filtered tibble for the canton/cost group, sorted by date.
#'
#' @export
extract_series <- function(data, canton = "Suisse", cost_grp = "Total") {
  data %>%
    filter(Canton == canton, Groupe_de_couts == cost_grp) %>%
    arrange(Date)
}

# ------------------------------------------------------------------------------
# Convert to quarterly time series object
# ------------------------------------------------------------------------------

#' Convert tibble to quarterly time series
#'
#' @param df Tibble with columns `Date` and `Prestations_brutes_par_assure`.
#'
#' @return A list:
#'   * `ts` – `ts` object (frequency = 4)
#'   * `dates` – `Date` vector matching time index
#'
#' @export
build_quarterly_ts <- function(df) {
  stopifnot(nrow(df) > 0,
            all(c("Date", "Prestations_brutes_par_assure") %in% names(df)))
  
  ts_y <- ts(df$Prestations_brutes_par_assure,
             start     = c(year(min(df$Date)), quarter(min(df$Date))),
             frequency = 4)
  
  list(ts = ts_y, dates = as.Date(as.yearqtr(time(ts_y))))
}

#' Prepare wide-format multivariate time series (Canton × Time)
#'
#' @param data Tibble from `load_okp_dataset()`
#' @param cost_grp Cost group to filter on (e.g., "Total")
#' @return List with:
#'   * `Y` – matrix (rows = time, columns = cantons)
#'   * `dates` – Date vector
#'
#' @export
build_multivariate_ts <- function(data, cost_grp = "Total") {
  df <- data %>%
    filter(Groupe_de_couts == cost_grp) %>%
    dplyr::select(Date, Canton, Prestations_brutes_par_assure) %>%
    pivot_wider(names_from = Canton, values_from = Prestations_brutes_par_assure) %>%
    arrange(Date)
  
  Y <- df %>% dplyr::select(-Date) %>% as.matrix()
  dates <- df$Date
  
  list(Y = Y, dates = dates)
}
