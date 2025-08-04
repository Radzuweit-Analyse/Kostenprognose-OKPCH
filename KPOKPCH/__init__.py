__version__ = "0.0.1"

from importlib import import_module

__dmfm__ = [
    "DMFM",
]

__utils__ = [
    "seasonal_difference",
]

__all__ = __dmfm__ + __utils__


def __getattr__(name):
    if name in __dmfm__:
        module = import_module(".dmfm", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    elif name in __utils__:
        module = import_module(".utils", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
