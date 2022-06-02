"""
virga_detection.py
====================================
Core module for virga detection and masking.

"""

from typing import Dict
import xarray as xr
import numpy as np

#: The recommended default configuration of virga-detection
DEFAULT_CONFIG = dict(
    smooth_window_cbh=60,  # [s] smoothing of CBH
    smooth_window_lcl=300,  # [s] smoothing of LCL if provided
    require_cbh=True,  # need a cloud base to be considered as virga?
    mask_below_cbh=True,  # vira if below cbh?
    mask_rain=True,  # apply rain mask from ancillary data?
    mask_zet=True,  # apply rain mask from radar signal?
    ze_thres=0,  # [dBz] minimum Radar signal at lowest range-gate which is considered rain
    mask_connect=True,  # apply virga mask cleaning regarding cbh layers
    mask_minrg=2,  # minimum number of range-gates in column to be considered virga
    mask_vel=True,  # apply velocity mask ?
    vel_thres=0,  # [ms-1] velocity threshold
    mask_clutter=True,  # apply clutter threshold line ?
    clutter_c=-8,  # [ms-1] intercept of clutter threshold line
    clutter_m=4,  # [ms-1 dBz-1] slope of clutter threshold line
    layer_threshold=500,  # [m] cbh layer separation
    virga_max_gap=150,  # [m] maximum gap between virga signal to count as connected virga and for clouds to cloud base
    clean_threshold=0.05,  # [0-1] remove cbh layer if below (clean_treshold*100)% of total data
    cbh_layer_fill=True,  # fill gaps of cbh layer?
    cbh_fill_method='slinear',  # fill method of cbh layer gaps
    layer_fill_limit=1,  # [min] fill gaps of cbh layer with this gap limit
    cbh_ident_function=[1, 0, 2, 0, 3, 1, 0, 2, 0, 3, 4])# order of operations applied to cbh: 0-clean, 1-split, 2-merge, 3-add-LCL, 4-smooth


def check_input_config(input: xr.Dataset, config: dict) -> None:
    input_vars = [v for v in input.keys()]
    if config['mask_rain'] and not "flag_surface_rain" in input_vars:
        raise Exception("config['mask_rain']==True while input['flag_surface_rain'] is missing.")

def virga_mask(input: xr.Dataset, config: dict = {}) -> xr.Dataset:
    """
    This function identifies virga from input data of radar reflectivity, ceilometer cloud-base height and
    optionally doppler velocity, lifting condensation level and surface rain-sensor--rain-flag.

    Parameters
    ----------
    input: xarray.Dataset
        The input data
    config: dict
        The configuration flags and thresholds. Will be merged with the default configuration, see :ref:`02_setup.md#configuration`.

    Returns
    -------
    vmask: xarray.Dataset
        The output data
    --------
    """
    def _expand_mask(data: np.ndarray, size: int) -> np.ndarray:
        """ Expands data of 1D shape, by adding a dimension and repeating <size> times.
        """
        return np.repeat(data[:, np.newaxis], size, axis=1)

    # merge user config with default config
    config = {**DEFAULT_CONFIG, **config}

    # check input
    check_input_config(input, config)


    return input
