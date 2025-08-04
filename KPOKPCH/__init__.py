__version__ = "0.0.1"

from importlib import import_module

__DMFM__ = [
    "DMFMModel",
    "DMFMDynamics",
    "KalmanFilterDMFM",
    "EMEstimatorDMFM",
]

__forecast__ = [
    "seasonal_difference",
    "integrate_seasonal_diff",
    "forecast_dmfm",
    "out_of_sample_rmse",
]

__all__ = __DMFM__ + __forecast__


def __getattr__(name):
    if name in __DMFM__:
        module = import_module(".DMFM", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    elif name in __forecast__:
        module = import_module(".forecast", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
