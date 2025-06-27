#' Compute year-over-year growth from quarterly data
#'
#' Aggregates quarterly values to yearly sums for each combination of grouping
#' variables and then calculates the year-over-year (YoY) percentage growth.
#'
#' @param df A tibble containing quarterly data with columns `Date`, `Value`,
#'   `Model` and optionally `Origin` and `Canton`.
#' @param include_canton_column Logical. If `TRUE` and if `Canton` is present in
#'   `df`, the canton column will be kept in the output.
#' @param actuals Optional tibble of historical actual observations with columns
#'   `Date` and `Value`.  When supplied together with `Origin` in `df`, these
#'   observations are used to complete partial years so that year\-over\-year
#'   growth can be computed for every origin date.
#' 
#' Years with fewer than four quarters of data are discarded.
#' 
#' @return A tibble with yearly YoY percentage growth and yearly totals. Columns
#'   include `Date`, `Value` (YoY % growth), `Yearly_Cost`, `Model` and
#'   optionally `Origin` and `Canton` if supplied.
#' @export
compute_yoy_growth <- function(df, include_canton_column = FALSE, actuals = NULL) {
  stopifnot(all(c("Date", "Value", "Model") %in% names(df)))

  # ---------------------------------------------------------------------------
  # Optionally augment forecast paths with historical actuals -----------------
  # ---------------------------------------------------------------------------
  if (!is.null(actuals) && "Origin" %in% names(df)) {
    stopifnot(all(c("Date", "Value") %in% names(actuals)))
    
    hist_cols <- c("Date", "Value",
                   if (include_canton_column && "Canton" %in% names(actuals))
                     "Canton")
    
    actuals <- actuals %>% dplyr::select(all_of(hist_cols))
    
    origins <- unique(df$Origin)
    models  <- unique(df$Model)
    
    hist_long <- purrr::map_dfr(origins, function(orig) {
      add_vals <- actuals %>% dplyr::filter(Date < orig)
      tidyr::crossing(Origin = orig, Model = models) %>%
        mutate(data = list(add_vals)) %>% tidyr::unnest(data)
    })
    
    df <- dplyr::bind_rows(df, hist_long) %>% dplyr::distinct()
  }
  
  df <- df %>% mutate(Year = lubridate::year(Date))

  grouping_vars <- c(if (include_canton_column && "Canton" %in% names(df)) "Canton",
                     "Model",
                     if ("Origin" %in% names(df)) "Origin",
                     "Year")

  yearly <- df %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(
      Yearly_Cost = sum(Value),
      Quarter_Count = dplyr::n(),
      .groups = "drop"
    ) %>%
    arrange(across(all_of(grouping_vars))) %>%
    filter(Quarter_Count == 4)

  diff_vars <- c(if (include_canton_column && "Canton" %in% names(df)) "Canton",
                 "Model",
                 if ("Origin" %in% names(df)) "Origin")

  yoy <- yearly %>%
    group_by(across(all_of(diff_vars))) %>%
    mutate(Value = (Yearly_Cost / lag(Yearly_Cost) - 1) * 100) %>%
    ungroup() %>%
    filter(!is.na(Value)) %>%
    mutate(Date = lubridate::make_date(Year, 10, 1))

  yoy %>% select(all_of(c(diff_vars)), Date, Yearly_Cost, Value)
}
