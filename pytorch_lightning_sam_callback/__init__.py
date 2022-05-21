try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore


try:
    __version__: str = version(__name__)
except PackageNotFoundError:
    __version__: str = "unknown"

from .sam import SAM

__all__ = ["SAM"]
