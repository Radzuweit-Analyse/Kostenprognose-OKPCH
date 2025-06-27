# =======================================================================
# multivariate_KF.R — Multivariate Kalman‑Filter Utilities
# =======================================================================

#' @title Multivariate Kalman‑Filter Utilities for OKP Cost Data
#' @description
#' Provides helper functions to:
#' \enumerate{
#'   \item Transform OKP health‑care cost data into a quarterly multivariate
#'     time‑series (cantons × quarters).
#'   \item Fit a single local‑level + seasonal state‑space model to all cantons
#'     simultaneously via the Kalman filter and generate forecasts.
#'   \item Perform a robust rolling‑origin back‑test of the forecasting
#'     procedure.
#' }
#'
#' The code is intentionally dependency‑light and returns base‑R objects
#' (`ts`, `matrix`, `tibble`), making it appropriate for both ad‑hoc analysis
#' and inclusion in package workflows.
#'
#' @section Required Packages:
#' \pkg{dplyr}, \pkg{tidyr}, \pkg{lubridate}, \pkg{zoo}, \pkg{purrr},
#' \pkg{tibble}, and \pkg{KFAS}.
#'
#' @section Usage:
#' ```r
#' source("main.R")             # brings COST_GROUP, FORECAST_HORIZON, etc. into scope
#' source("multivariate_KF.R")  # loads the helpers below
#'
#' df   <- load_okp_dataset()
#' cmat <- build_canton_matrix_ts(df)
#' fc   <- kalman_mv_forecast(cmat$ts, h = 8)  # 2‑year horizon (8 quarters)
#'
#' # Back‑testing
#' ro <- rolling_origin_kalman_mv(cmat$ts, cmat$dates, h = 8)
#' ```
#'
#' @author xxx
#' @keywords time‑series Kalman multivariate forecasting
#' @family multivariate‑KF
#' @noRd
NULL

# ---- 1. Build quarterly canton matrix -----------------------------------

#' Build a quarterly canton matrix time‑series
#'
#' Converts the raw OKP data set (long form) into a wide, quarterly
#' multivariate \code{ts} where each column corresponds to one canton and
#' each row to a quarter.
#'
#' @param data A \link[tibble]{tbl_df} as returned by \code{\link{load_okp_dataset}}.
#' @param cost_grp Single string. Cost basket to filter on; defaults to the
#'   constant \code{COST_GROUP}, which is usually set to \code{"Total"} in
#'   \code{main.R}.
#' @return A named list with elements
#' \describe{
#'   \item{ts}{Quarterly multivariate \code{ts}. Rows = quarters, columns = cantons.}
#'   \item{dates}{Companion \code{Date} vector giving the quarter‑end date for
#'     each row of \code{ts}.}
#'   \item{cantons}{Character vector of canton ISO‑2 codes (column order).}
#' }
#' @details
#' The national aggregate row (\dQuote{Suisse}) is dropped because it is a
#' linear combination of the cantonal series and would lead to a singular
#' state‑space covariance matrix.
#'
#' Missing observations are left as \code{NA}; the Kalman filter treats them
#' correctly via its internal missing‑data handling.
#'
#' @examples
#' df   <- load_okp_dataset()
#' cmts <- build_canton_matrix_ts(df)
#' str(cmts$ts)
#' @export
build_canton_matrix_ts <- function(data,
                                   cost_grp = COST_GROUP) {
  # --- Step 1: Filter and reshape data ------------------------------------
  wide <- data |>
    dplyr::filter(.data$Groupe_de_couts == cost_grp, .data$Canton != "Suisse") |>
    dplyr::select(Canton, Date,
                  Value = Prestations_brutes_par_assure) |>
    tidyr::pivot_wider(names_from  = Canton,
                       values_from = Value) |>
    dplyr::arrange(.data$Date)
  
  # --- Step 2: Convert to matrix ------------------------------------------
  mat <- as.matrix(dplyr::select(wide, -Date))
  
  # --- Step 3: Build ts object --------------------------------------------
  ts_y <- stats::ts(mat,
                    start = c(lubridate::year(min(wide$Date)),
                              lubridate::quarter(min(wide$Date))),
                    frequency = 4)
  
  # --- Step 4: Return results ---------------------------------------------
  list(ts      = ts_y,
       dates   = as.Date(zoo::as.yearqtr(stats::time(ts_y))),
       cantons = colnames(mat))
}

# ---- 2. Fit multivariate state‑space model ------------------------------

#' Multivariate local‑level + seasonal Kalman‑filter forecast
#'
#' Fits a single local‑level plus quarterly seasonal dummy state‑space model
#' to the entire canton panel and generates \code{h}-step‑ahead forecasts
#' for every canton simultaneously.
#'
#' @param train_y Multivariate time‑series (\code{ts}) with one column per
#'   canton. Missing values are allowed and handled by the Kalman filter.
#' @param h Integer \(\ge 1\). Forecast horizon measured in quarters.
#'   Defaults to the global constant \code{FORECAST_HORIZON}.
#' @return A numeric matrix of dimension \eqn{h \times k}, where \eqn{k} is the
#'   number of cantons. The matrix inherits the column names of
#'   \code{train_y}.
#' @section Model specification:
#' \eqn{y_{t} = \mu_{t} + \gamma_{t} + \varepsilon_{t}} with
#'   \eqn{\mu_{t} = \mu_{t-1} + \eta_{t}}, \\
#'   \eqn{\gamma_{t}} = quarterly seasonal dummy (sum‑to‑zero constraint), \\
#'   \eqn{\varepsilon_{t} \sim N(0, H)}, \\
#'   \eqn{\eta_{t} \sim N(0, Q_\text{level})}.
#'
#' All variances \eqn{H}, \eqn{Q_\text{level}}, and \eqn{Q_\text{seasonal}}
#' are estimated by maximum likelihood via \code{\link{fitSSM}} (BFGS).
#'
#' @note If the optimiser fails to converge, the function returns a matrix of
#' \code{NA_real_} with the correct dimensions so that calling code can proceed
#' gracefully.
#'
#' @examples
#' df   <- load_okp_dataset()
#' cmts <- build_canton_matrix_ts(df)
#' kalman_mv_forecast(cmts$ts, h = 4)
#' @importFrom KFAS SSModel SSMcustom SSMseasonal fitSSM
#' @export
kalman_mv_forecast <- function(train_y, h = FORECAST_HORIZON) {
  
  stopifnot(is.matrix(train_y), is.ts(train_y), h >= 1)
  k <- ncol(train_y)
  
  # --- Step 1: Derive robust initial variance guesses ---------------------
  var_vec <- apply(train_y, 2, stats::var, na.rm = TRUE)
  var_vec[!is.finite(var_vec) | var_vec <= 0] <- 1         # fallback
  var_vec <- pmin(var_vec, 1e6)                            # avoid overflow
  init_val <- rep(log(var_vec), 3)                         # level, seasonal, obs
  
  # --- Step 2: Build state‑space model ------------------------------------
  Z  <- matrix(diag(1, k, k),  nrow = k, ncol = k)
  Tt <- matrix(diag(1, k, k),  nrow = k, ncol = k)
  R  <- matrix(diag(1, k, k),  nrow = k, ncol = k)
  Q  <- matrix(NA_real_, k, k)
  H  <- matrix(NA_real_, k, k)
  P1inf <- matrix(1, k, k)                 # diffuse prior for level
  
  model <- SSModel(train_y ~ -1 +
                     SSMcustom(Z = Z,  T = Tt, R = R, Q = Q, P1inf = P1inf) +
                     SSMseasonal(period = 4, sea.type = "dummy",
                                 Q = matrix(NA_real_, k, k)),
                   H = H)
  
  # --- Step 3: Fit model and forecast -------------------------------------
  fit <- tryCatch(
    KFAS::fitSSM(model, inits = init_val, method = "BFGS"),
    error = function(e) NULL
  )
  
  if (is.null(fit)) {
    return(matrix(NA_real_, nrow = h, ncol = k,
                  dimnames = list(NULL, colnames(train_y))))
  }
  
  stats::predict(fit$model, n.ahead = h)
}

# ---- 3. Rolling‑origin evaluation ---------------------------------------

#' Rolling‑origin evaluation of the multivariate Kalman‑filter
#'
#' Generates a tidy data set of out‑of‑sample forecasts for every possible
#' origin in the training window, thereby enabling time‑series cross‑validation
#' or back‑testing with custom scoring rules.
#'
#' @param ymt   Multivariate \code{ts} produced by
#'   \code{\link{build_canton_matrix_ts}}.
#' @param dates Companion vector of \code{Date}s with the same length as
#'   \code{nrow(ymt)}.
#' @param h     Forecast horizon in quarters (passed to
#'   \code{\link{kalman_mv_forecast}}).
#' @return A tidy \link[tibble]{tibble} with columns
#'   \describe{
#'     \item{Date}{The target quarter.}
#'     \item{Canton}{Canton code.}
#'     \item{Value}{Either the observed cost (if \code{Date == Origin}) or the
#'       point forecast.}
#'     \item{Origin}{The quarter in which the forecast was issued.}
#'   }
#'
#' @examples
#' df   <- load_okp_dataset()
#' cmts <- build_canton_matrix_ts(df)
#' ro   <- rolling_origin_kalman_mv(cmts$ts, cmts$dates, h = 4)
#' dplyr::glimpse(ro)
#' @export
rolling_origin_kalman_mv <- function(ymt, dates, h = FORECAST_HORIZON) {
  
  k  <- ncol(ymt)
  cn <- colnames(ymt)
  
  purrr::map_dfr(seq_len(nrow(ymt) - h), function(i) {
    
    train_y <- window(ymt, end = time(ymt)[i])
    if (nrow(train_y) < 8) return(NULL)                 # ensure burn-in period
    
    origin  <- dates[i]
    
    # --- Step 1: Produce h-step forecasts ------------------------------
    fc_raw <- kalman_mv_forecast(train_y, h)            # any shape allowed
    
    # --- Step 2: Coerce forecast into numeric matrix -------------------
    fc_mat <- tryCatch({
      if (is.null(dim(fc_raw))) {
        matrix(as.numeric(fc_raw), nrow = 1, ncol = k,
               dimnames = list(NULL, cn))
      } else if (length(dim(fc_raw)) == 3) {
        matrix(as.numeric(fc_raw[, , 1]),
               nrow = dim(fc_raw)[1], ncol = k,
               dimnames = list(NULL, cn))
      } else {
        matrix(as.numeric(fc_raw), ncol = k,
               dimnames = list(NULL, cn))
      }
    }, error = function(e) matrix(NA_real_, nrow = 1, ncol = k,
                                  dimnames = list(NULL, cn)))
    
    h_eff <- nrow(fc_mat)                               # actual horizon
    fut_dt <- dates[(i + 1):(i + h_eff)]
    
    # --- Step 3: Build tidy rows ---------------------------------------
    obs_row <- tibble::tibble(
      Date   = rep(origin, k),
      Canton = cn,
      Value  = as.numeric(train_y[nrow(train_y), ]),
      Origin = rep(origin, k)
    )
    
    fc_rows <- tibble::tibble(
      Date   = rep(fut_dt, each = k),
      Canton = rep(cn,     times = h_eff),
      Value  = as.numeric(fc_mat),                      # *** pure numeric ***
      Origin = rep(origin, k * h_eff)
    )
    
    dplyr::bind_rows(obs_row, fc_rows)
  })
}
