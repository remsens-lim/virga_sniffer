"""
layer_utils.py
====================================
Functions to smooth, split, merge and sort layered data.
Layered data are time periods of one or more layers of height data (e.g. cloud-base-height).
Layer DataArrays are 2-D arrays of usually of dims (time,layer).
The time dimension coordinate data is often required in numpy.datetime64 format,
whereas layer dimension data is not required. The naming of the dimensions is irrelevant.
"""
from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np
import xarray as xr

from . import utils


def clean(input_data: xr.DataArray, clean_threshold: float) -> xr.DataArray:
    """
    Clean input layer data by dropping layer with low number of data points according to `clean_threshold`.

    Parameters
    ----------
    input_data: xarray.DataArray
        Input 2-D DataArray, assuming 2nd dimension refers to layer number.
    clean_threshold: float
        Layer with number of datapoints < `clean_threshold`*`input_data`.shape[0] will be deleted

    Returns
    -------
    xarray.DataArray
        `input_data` but layer with no or low number of  datapoints dropped.
        The 2nd dimension will be re-indexed.
    """
    dims = input_data.dims
    data_tmp = input_data.values
    layer_ndatapoints = np.count_nonzero(~np.isnan(data_tmp), axis=0)
    layer_nthreshold = clean_threshold * data_tmp.shape[0]
    data_tmp = data_tmp[:, layer_ndatapoints > layer_nthreshold]
    output_data = xr.DataArray(data_tmp,
                               coords={dims[0]: input_data[dims[0]].data,
                                       dims[1]: np.arange(data_tmp.shape[1])})
    return output_data


def fill_nan(input_data: xr.DataArray,
             limit: Union[float, None] = None,
             method: str = 'ffill',
             return_mask: bool = True) -> (xr.DataArray, Optional[NDArray[bool]]):
    """
    Fill input_data nan values using chosen xarray method. The gap limit is specified in seconds.

    Parameters
    ----------
    input_data: xarray.DataArray
        The input layer data (N,M), where N coordinate data is assumed to be datetime64 type.
    limit: float or None, optional
        The maximum gap of successive nan-values in seconds. If None fill all gaps. The default is None.
    method: {'ffill', 'bfill', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial'}, optional
        The method to apply for filling.

        * ffill: propagate last valid value forward (:py:func:`xarray.DataArray.ffill`)
        * bfill: propagate the next valid value backward (:py:func:`xarray.DataArray.bfill`)
        * nearest, zero, slinear, quadratic, cubic, polynomial: interpolate gaps using given method
        (:py:func:`xarray.DataArray.interpolate_na`)

        The default is 'ffill'.
    return_mask: bool, optional
        If True return array of bool indicating which data got filled (True if filled). The default is True.

    Returns
    -------
    xarray.DataArray, (numpy.ndarray, optional)
        If return_mask is True, returns merged `input_data` and mask indicating filled data,
         else only filled `input_data` is returned.
    """
    time_dim = input_data.dims[0]
    # translate gap limit in seconds to number of datapoints
    if not isinstance(limit, type(None)):
        cbh_freq = 1. / np.diff(input_data[time_dim]).mean().astype("timedelta64[s]").astype(float)
        limit = int(limit * cbh_freq)

    # Use on of xarrays methods to fill the gaps < limit
    if method == 'ffill':
        output_data = input_data.ffill(dim=time_dim, limit=limit)
    elif method == 'bfill':
        output_data = input_data.bfill(dim=time_dim, limit=limit)
    elif method in ['nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial']:
        output_data = input_data.interpolate_na(dim=time_dim, method=method, limit=limit)
    else:
        raise ValueError(
            "method must be one of: 'ffill','bfill','nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial'")

    if return_mask:
        merge_mask = np.isnan(input_data.values) * ~np.isnan(output_data.values)
        return output_data, merge_mask
    else:
        return output_data


def merge(input_data: xr.DataArray, layer_threshold: float) -> xr.DataArray:
    """
    Merging layer height data by comparing gap-filled lower layers to all layers above them.
    If distance smaller given threshold, upper layer data will be added to lower layer, or merged by mean value.

    Parameters
    ----------
    input_data: xarray.DataArray
        The 2D input data of height values, assuming the 2nd dimension to be the layer number dimension
    layer_threshold: float
        Distance threshold in [m]. Merge will be performed if distance of two layers is smaller than this threshold.

    Returns
    -------
    xarray.DataArray
        Copy of Input data, but layer data is merged. Note: intermediate layers might be all nan values.
    """
    ishape = input_data.shape
    idims = input_data.dims
    output_data = input_data.copy()
    for lower_layer in range(ishape[1] - 1):
        # fill nan-values in gaps to compare them to other layers
        layer_filled = output_data[:, lower_layer].interpolate_na(dim=idims[0], method='linear')
        # fill also nan-values at start and end of the data
        layer_filled = layer_filled.bfill(idims[0]).ffill(idims[0])
        for upper_layer in range(lower_layer + 1, ishape[1]):
            cbhdiff = np.abs(layer_filled.values - output_data.values[:, upper_layer])
            # check if distance smaller than given threshold, these datapoints will be merged
            mergeidx = np.argwhere(cbhdiff < layer_threshold)
            # split 'to merge' index in data which will simply be added (original layer is nan)
            mergeidx_add = mergeidx[np.isnan(output_data.values[mergeidx, lower_layer])]
            # .. or merged by mean if both compared layers have data
            mergeidx_mean = mergeidx[~np.isnan(output_data.values[mergeidx, lower_layer])]

            # apply merging to lower layer, and subsequently remove data form the upper layer
            output_data.values[mergeidx_add, lower_layer] = output_data.values[mergeidx_add, upper_layer]
            output_data.values[mergeidx_add, upper_layer] = np.nan
            output_data.values[mergeidx_mean, lower_layer] = np.mean([output_data.values[mergeidx_mean, upper_layer],
                                                                      output_data.values[mergeidx_mean, lower_layer]],
                                                                     axis=0)
            output_data.values[mergeidx_mean, upper_layer] = np.nan
    return output_data


def replace_nan(input_data: xr.DataArray,
                input_layer: xr.DataArray,
                layer: int = 0,
                return_mask: bool = True) -> (xr.DataArray, Optional[NDArray[bool]]):
    """
    Replace nan values of `input_data` nan values of certain `layer` (index of 2nd axis) with `input_layer` data.

    Parameters
    ----------
    input_data: xarray.DataArray
        The 2D input data (M,N), assuming the 2nd dimension to be the layer number dimension.
    input_layer: xarray.DataArray
        1D layer data (M), will be used to replace nan values of `input_data`[:,`layer`].
    layer: int, optional
        Index of N of `input_data` to be merged with input_layer. The default is 0.
    return_mask: bool, optional
            If True return array of bool indicating which data got merged (True if merged). The default is True.

    Returns
    -------
    xarray.DataArray, (numpy.ndarray, optional)
        If return_mask is True, returns merged `input_data` and mask indicating merged data,
         else only merged `input_data` is returned.
    """
    data_tmp = input_data.copy()
    data_tmp.values[:, layer] = input_layer.values
    output_data = input_data.dropna(dim=input_data.dims[layer])
    output_data = output_data.combine_first(data_tmp)
    if return_mask:
        merge_mask = np.isnan(input_data.values[:, layer])
        merge_mask *= ~np.isnan(input_layer.values)
        return output_data, merge_mask
    else:
        return output_data


def smooth(input_data: xr.DataArray, window: int) -> xr.DataArray:
    """
    Smooth layered data using a rolling median filter of `window` size.

    Parameters
    ----------
    input_data: xarray.DataArray
        Input DataArray  either 1D or 2D, assuming first dimension of type np.datetime64.
    window: int
        Smoothing window size in seconds

    Returns
    -------
    xarray.DataArray
        Copy of `input_data` but smoothed.
    """
    dims = input_data.dims
    output_data = input_data.copy()

    # find mean frequency of time series
    freq = 1. / np.diff(input_data[dims[0]]).mean().astype("timedelta64[s]").astype(float)
    if len(dims) == 2:
        for layer in range(input_data[dims[1]].size):
            output_data.values[:, layer] = utils.medfilt(input_data.values[:, layer], freq, window)
    elif len(dims) == 1:
        output_data.values[:] = utils.medfilt(input_data.values[:], freq, window)
    else:
        raise Exception(f"layer_utils.smooth input_data requires 1 or 2 dimensions, not {len(dims)}.")
    return output_data


def sort(input_data: xr.DataArray) -> xr.DataArray:
    """
    Sort input layer data by layer mean value.

    Parameters
    ----------
    input_data: xarray.DataArray
        Input 2-D DataArray, assuming 2nd dimension refers to layer number.

    Returns
    -------
    xarray.DataArray
        Copy of `input_data` but sorted .

    """
    output_data = input_data.copy()
    data_tmp = input_data.values
    output_data.values = data_tmp[:, np.nanmean(data_tmp, axis=0).argsort()]
    return output_data


def split(input_data: xr.DataArray, layer_threshold: float) -> xr.DataArray:
    """
    Split input layer data by creating new layer for values up or below `layer_threshold` from layer mean.

    Parameters
    ----------
    input_data: xarray.DataArray
        Input 2-D DataArray (M,N), assuming 2nd dimension refers to layer number.
    layer_threshold: float
        layer mean +- `layer_threshold` will be used to identify layer values to push to a new layer.
         Same unit as `input_data.values`

    Returns
    -------
    xarray.DataArray
        Output 2-D DataArray (M,N+x), where x is number of newly created layer by splitting data using
        `layer_threshold` value. This DataArray is sorted using :py:func:`layer_sort`.

    """
    data_dims = input_data.dims
    data_values = input_data.values
    # the time dimension will not change, so fix it here
    tsize = data_values.shape[0]

    # new layers will continuously insert until all layer values converge below layer_threshold
    # keep track of new layers for each cycle until no new layers are generated
    newlayer_count = np.inf
    while newlayer_count > 0:
        newlayer_count = 0
        for layer in range(data_values.shape[1]):
            layer_data = data_values[:, layer]
            layer_mean = np.nanmean(layer_data)
            # create up to two new layers with data up or below layer threshold
            idx1 = layer_data > layer_mean + layer_threshold
            idx2 = layer_data < layer_mean - layer_threshold
            for idx in [idx1, idx2]:
                if np.count_nonzero(idx) == 0:
                    continue
                newlayer_count += 1
                newlayer_data = np.full((tsize, 1), np.nan)
                newlayer_data[idx, 0] = layer_data[idx]
                data_values[idx, layer] = np.nan
                data_values = np.concatenate((data_values, newlayer_data), axis=1)
    output_data = xr.DataArray(data_values,
                               coords={data_dims[0]: input_data[data_dims[0]].data,
                                       data_dims[1]: np.arange(data_values.shape[1])})
    output_data = sort(output_data)
    return output_data
