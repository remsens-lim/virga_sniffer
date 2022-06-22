"""

cloud_base_height.py
====================================
Core module for cloud-base height preprocessing.

"""

import xarray as xr

from .virga_detection import DEFAULT_CONFIG

def process_cbh(input_data: xr.DataArray, config:dict = None) -> xr.DataArray:

    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        config = {**DEFAULT_CONFIG, **config}


    process_data = input_data.copy()
    pass
