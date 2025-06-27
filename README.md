# OKP Cost Forecast

This repository provides R scripts to forecast Swiss health‑care costs under the compulsory insurance scheme (OKP). Several Kalman filter variants and standard benchmarks are implemented. Helper functions for data loading, error metrics and plotting reside in the `R/` directory.

## Required R packages
Install the following CRAN libraries before running the code:

* **tidyverse** (dplyr, ggplot2, purrr, tidyr)
* **lubridate**
* **zoo**
* **readxl**
* **forecast**
* **KFAS**
* **bssm** (used only in the non‑linear example)

## How to run the scripts

### main.R – univariate benchmark comparison

This script compares ARIMA, random walk and structural time‑series models. From an R session run:

```r
source("main.R")            # loads functions
main()                      # run the analysis
```

Optional arguments allow overriding the data file path or forecast horizon.

### multivariate_main.R – canton level analysis

The multivariate workflow fits a structural time‑series model across all cantons:

```r
source("multivariate_main.R")
main_multivariate()   # returns forecasts, RMSE table and plots
```

### non_linear_main.R – non‑linear state space model

This example demonstrates a non‑linear Kalman filter using **bssm**. Sourcing the file performs the analysis and creates two data frames in the global environment: `nonlinear_kalman_smoothed` and `nonlinear_kalman_forecast`.

```r
source("non_linear_main.R")
```

## Dataset

All scripts expect the quarterly monitoring file `02_Monitoring-des-couts_Serie-temporelle-trimestre.xlsx` in the working directory. The data originates from the Swiss Federal Office of Public Health and is **not distributed** with this repository for licensing reasons. Please obtain the file separately before running the code.

## License

This project is licensed under a proprietary license. See the [LICENSE](LICENSE) file for details.
