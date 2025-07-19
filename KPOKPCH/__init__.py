__version__ = "0.0.1"

from importlib import import_module

__all__ = [
    "initialize_dmfm",
    "_construct_state_matrices",
    "kalman_smoother_dmfm",
    "qml_loglik_dmfm",
    "qml_objective_dmfm",
    "pack_dmfm_parameters",
    "unpack_dmfm_parameters",
    "em_step_dmfm",
    "fit_dmfm_em",
    "optimize_qml_dmfm",
    "compute_standard_errors_dmfm",
    "compute_standard_errors_dynamics",
    "identify_dmfm_trends",
    "test_unit_root_factors",
    "select_dmfm_rank",
    "select_dmfm_qml",
    "forecast_dmfm",
    "conditional_forecast_dmfm",
    "subsample_panel",
    "fit_dmfm_local_qml",
    "aggregate_dmfm_estimates",
    "fit_dmfm_distributed",
]

def __getattr__(name):
    if name in __all__:
        module = import_module(".dmfm", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
