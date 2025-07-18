__version__ = "0.0.1"

from importlib import import_module

__all__ = [
    "initialize_dmfm",
    "_construct_state_matrices",
    "kalman_smoother_dmfm",
    "qml_loglik_dmfm",
    "em_step_dmfm",
    "fit_dmfm_em",
    "compute_standard_errors_dmfm",
    "select_dmfm_rank",
    "select_dmfm_qml",
]

def __getattr__(name):
  if name in __all__:
    module = import_module(".dmfm", __name__)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
