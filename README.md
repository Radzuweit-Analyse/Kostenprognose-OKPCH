# OKP Cost Forecast

This repository contains R scripts for forecasting Swiss healthcare costs using various Kalman filter approaches. Utility functions live under the `R/` directory.

## Running the main pipeline

Either source `run_main.R` or call `main()` manually:

```r
source("run_main.R")        # sources main.R and executes main()
# OR
source("main.R")            # loads functions
main()                      # run the analysis
```

The multivariate workflow can be executed similarly by sourcing `multivariate_main.R` and then calling `main_multivariate()` from an R session.

This project is licensed under a proprietary license. See the [LICENSE](LICENSE) file for details.
