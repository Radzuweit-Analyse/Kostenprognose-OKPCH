#' Compute year-over-year growth from quarterly data
#'
#' Aggregates quarterly values to yearly sums for each combination of grouping
#' variables and then calculates the year-over-year (YoY) percentage growth.
#'
#' @param df A tibble containing quarterly data with columns `Date`, `Value`,
#'   `Model` and optionally `Origin` and `Canton`.
#' @param include_canton_column Logical. If `TRUE` and if `Canton` is present in
#'   `df`, the canton column will be kept in the output.
#'
#' @return A tibble with yearly YoY percentage growth. Columns include `Date`,
#'   `Value`, `Model` and optionally `Origin` and `Canton` if supplied.
#' @export
compute_yoy_growth <- function(df, include_canton_column = FALSE) {
  stopifnot(all(c("Date", "Value", "Model") %in% names(df)))

  df <- df %>% mutate(Year = lubridate::year(Date))

  grouping_vars <- c(if (include_canton_column && "Canton" %in% names(df)) "Canton",
                     "Model",
                     if ("Origin" %in% names(df)) "Origin",
                     "Year")

  yearly <- df %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(total = sum(Value), .groups = "drop") %>%
    arrange(across(all_of(grouping_vars)))

  diff_vars <- c(if (include_canton_column && "Canton" %in% names(df)) "Canton",
                 "Model",
                 if ("Origin" %in% names(df)) "Origin")

  yoy <- yearly %>%
    group_by(across(all_of(diff_vars))) %>%
    mutate(Value = (total / lag(total) - 1) * 100) %>%
    ungroup() %>%
    filter(!is.na(Value)) %>%
    mutate(Date = lubridate::make_date(Year, 1, 1))

  yoy %>% select(all_of(c(diff_vars)), Date, Value)
}
