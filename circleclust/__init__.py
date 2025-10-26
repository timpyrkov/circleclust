"""
CircleClust - Clustering on periodic circular coordinates with automatic detection of centroids and boundary handling.
"""

# Public API
from .circleclust import CircleClust

# Package version (prefer stdlib importlib.metadata; fall back to backport if needed)
try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    try:
        from importlib_metadata import PackageNotFoundError, version  # type: ignore
    except Exception:  # pragma: no cover
        PackageNotFoundError = Exception  # type: ignore
        def version(_name: str) -> str:  # type: ignore
            return "0.0.0"

try:
    __version__ = version("circleclust")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["CircleClust", "__version__"]
