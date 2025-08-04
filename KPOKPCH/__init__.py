__version__ = "0.0.1"

from importlib import import_module

__all__ = [
    "DMFM",
    "seasonal_difference",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(".dmfm", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
