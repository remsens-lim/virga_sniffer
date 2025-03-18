from .virga_detection import virga_mask
from .layer_utils import process_cbh

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("virga_sniffer")
except PackageNotFoundError:
    # package is not installed
    pass


__all__ = (
    # Top-level functions
    "virga_mask",
    "process_cbh",
    "DEFAULT_CONFIG"
)
