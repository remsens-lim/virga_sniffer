"""
virga_detection.py
====================================
Core module for virga detection and masking.

"""
import os
import xarray as xr
import numpy as np
import json

from . import vsplot
from . import layer_utils
from . import utils

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
    cbh_ident_function=[1, 0, 2, 0, 3, 1, 0, 2, 0, 3, 4])  # order of operations applied to cbh: 0-clean, 1-split, 2-merge, 3-add-LCL, 4-smooth


def check_input_config(input_data: xr.Dataset, config: dict) -> None:
    """
    Check input requirements of `virga_mask`

    Parameters
    ----------
    input_data: xarray.Dataset
    config: dict

    Returns
    -------
    None

    """
    input_vars = [v for v in input_data.variables]
    if "ze" not in input_vars:
        raise Exception("input missing 'ze' variable.")
    if "cbh" not in input_vars:
        raise Exception("input missing 'cbh' variable.")
    if config['mask_rain'] and ("flag_surface_rain" not in input_vars):
        raise Exception("config['mask_rain']==True while input['flag_surface_rain'] is missing.")
    if config['mask_vel'] and ("vel" not in input_vars):
        raise Exception("config['mask_vel']==True while input['vel'] is missing.")
    if config['mask_clutter'] and ("vel" not in input_vars):
        raise Exception("config['mask_clutter']==True while input['vel'] is missing.")


def virga_mask(input_data: xr.Dataset, config: dict = None) -> xr.Dataset:
    """
    This function identifies virga from input data of radar reflectivity, ceilometer cloud-base height and
    optionally doppler velocity, lifting condensation level and surface rain-sensor--rain-flag.

    Parameters
    ----------
    input_data: xarray.Dataset
        The input dataset with dimensions ('time','range','layer' (actual dimension-names doesn't matter)).

        Variables:

        * **Ze**: ('time','range') - radar reflectivity [dBz]
        * **cloud_base_height**: ('time','layer') - cloud base height [m]
        * **vel**: ('time', 'range') [optional] - radar doppler velocity [ms-1]
        * **lcl**: ('time') [optional] - lifting condensation level [m]
        * **flag_surface_rain**: ('time') [optional] - flags if rain at the surface [bool]

        Coords:

        * **time** ('time') - datetime [UTC]
        * **range** ('range') - radar range gate altitude (mid of rangegate) [m]
        * **layer** ('layer') - counting **cbh** layer (np.arange(cbh.shape[1])) [-]

    config: dict, optional
        The configuration flags and thresholds.
        Will be merged with the default configuration, see :ref:`02_setup.md#configuration`.

    Returns
    -------
    xarray.Dataset
        The output dataset
    --------
    """

    def _expand_mask(data: np.ndarray, size: int) -> np.ndarray:
        """ Expands data of 1D shape, by adding a dimension and repeating <size> times.
        """
        return np.repeat(data[:, np.newaxis], size, axis=1)

    # merge user config with default config
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        config = {**DEFAULT_CONFIG, **config}

    # unify coordinates naming in input
    input_coords = list(input_data.Ze.coords.keys())[:2]
    input_coords.append(list(input_data.cloud_base_height.coords.keys())[1])
    rename_map = {input_coords[0]: 'time',
                  input_coords[1]: 'range',
                  input_coords[2]: 'layer'}
    ds = input_data.rename(rename_map)

    # check input
    check_input_config(ds, config)

    # calculate top and base of each range-gate
    # this will be used to identify the range-gate index of CBH, CTH in radar data
    rgmid = ds.range.values
    rgedge = np.pad(np.diff(rgmid) / 2, pad_width=1, mode='edge')
    rgtop = rgmid + rgedge[1:]
    rgbase = rgmid - rgedge[:-1]

    # preprocess cbh data
    cbh, idx_lcl, idx_fill = layer_utils.process_cbh(ds, config=config)

    # Assign/ Initialize variables to work with
    cth = cbh.copy()*np.nan  # cloud-top height
    # initialize masks if with valid radar signal
    vmask = ~np.isnan(ds.Ze.values)  # flag_virga (True if Virga)
    cmask = vmask.copy()  # flag_cloud (True if cloud)

    # --------------------------------------
    # Apply  Virga-Sniffer on total profiles
    # --------------------------------------
    if config['mask_connect']:
        # Loop through cbh layers (bottom to top) and:
        # * remove from virga mask, if the signal connects on top of the cbh layer (probably cloud),
        #     but not to the next cbh layer
        # * remove cbh layer if virga connects through this layer to the next cbh layer
        # Checking connection of virga has a margin of "virga_max_gap" to ignore small gaps
        idxs = np.searchsorted(rgtop, cbh.values[:, :])  # find index of cbh layers in radar signal
        idxs[idxs == vmask.shape[1]] = -1  # if cbh == nan or above highest range gate, fill with -1
        # add virutal "CBH layer" at lowest range gate, so that the gap detection is also applied to
        # virga below lowest CBH layer
        idxs = np.concatenate((np.zeros(idxs.shape[0])[:, np.newaxis], idxs), axis=1).astype(int)
        # we need a second index, where cbh == nan or above highest range gate, filled with last index instead of -1
        # need bot idxs and idxs_tops for slicing later
        idxs_top = idxs.copy()
        idxs_top[idxs_top == -1] = vmask.shape[1] - 1

        # initialize temporal array
        mask_tmp = vmask.copy()
        # add True at cbh to fill nan directly on top of cbh layer
        np.put_along_axis(mask_tmp, idxs_top, values=True, axis=1)
        # switch to xarray to make use of their functions
        mask_tmp = xr.DataArray(mask_tmp, dims=('time', 'range'), coords={'time': ds.time.data, 'range': ds.range.data})
        mask_tmp = mask_tmp.where(mask_tmp)  # False to nan
        mask_int = mask_tmp.dropna(dim='time', thresh=2)  # remove nan for interpolation
        # interpolate mask, to fill small gaps (<layer_threshold) in virga
        mask_int = mask_int.interpolate_na(dim='range',
                                           method='nearest',
                                           max_gap=config['virga_max_gap'],
                                           bounds_error=False,
                                           fill_value=np.nan)
        # map back to original array
        mask_tmp = mask_int.combine_first(mask_tmp)
        # convert nan back to False, now we have a mask with True==virga, False==no-virga
        # but small gaps in original mask are filled with True
        mask_tmp = mask_tmp.fillna(False).values[:, 1:].astype(bool)
        # loop all timesteps i and cbh-layer l
        for itime in range(idxs.shape[0]):
            for ilayer in range(idxs.shape[1]):
                # identify index of cbh layer in radar signal
                # handling of nan values in cbh layer:
                #  * if nan choose the next higher or lower layer
                #  * lower cbh layer defaults to lowest range-gate
                #  * higher cbh layer defaults to highest range_gate
                # here we can use max and min functions as we set nan values to
                # * -1 in idxs
                # * vmask.shape[1]-1 in idxs_top
                ilow = np.max([np.max(idxs[itime, :ilayer + 1]), 0])
                if ilayer == idxs.shape[1] - 1:
                    itop = vmask.shape[1] - 1
                else:
                    itop = np.min(idxs_top[itime, ilayer + 1:])
                # just look at this part of the mask:
                # time i and between cbh layers
                mask_l = mask_tmp[itime, ilow:itop]
                if not np.all(mask_l):
                    # signal has a gap larger then layer_threshold
                    # consider everything below the first gap as cloud (no-virga)
                    gapidx = np.argwhere(~mask_l)
                    if gapidx.shape[0] > 0:
                        mask_l[:gapidx.ravel()[0]] = False
                        if ilayer != 0:
                            cth.values[itime, ilayer - 1] = rgtop[ilow + gapidx.ravel()[0]]
                    # update original mask
                    vmask[itime, ilow:itop] *= mask_l
                elif ilayer == 0:
                    # do nothing if we look at virtual added lowest layer
                    pass
                else:
                    # signal connects through cloud bases, so remove below cbh
                    # avoid added virtual cbh layer at lowest rangegate
                    cbh.values[itime, ilayer - 1] = np.nan

    cth = cth.where(~np.isnan(cbh))
    # can use additional smoothing
    cth = layer_utils.smooth(cth, window=config['smooth_window_cbh'])

    # Find index of layer data for range-gate data
    idxs_cbh = np.searchsorted(rgtop, cbh.values[:, :])  # find index of cbh layers in vmask
    idxs_cbh[idxs_cbh == ds.range.size] = -1  # if cbh == nan or above highest range gate, fill with -1
    idxs_cth = np.searchsorted(rgtop, cth.values[:, :])  # find index of cth layers in vmask
    idxs_cth[idxs_cth == ds.range.size] = -1  # if cbh == nan or above highest range gate, fill with -1

    if config['mask_below_cbh']:
        # Consider as virga if signal below a cbh
        # (If multiple cbh layer, this will include also clouds from the lower layers)
        cbhmask = utils.below_cloudbase(rgtop, cbh.values, config['require_cbh'])
        vmask *= cbhmask  # True below cbh

    if config['mask_vel']:
        # Consider as Virga if doppler velocity is below threshold
        # Velocity < 0 is considered as falling droplets
        vel_threshold = ds.vel.values < config['vel_thres']
        vmask *= vel_threshold

    if config['mask_clutter']:
        # Remove certain Signal+Velocity combinations.
        # Eq has to be fullfilled to be considered as virga: velocity > signal*(-m) + c
        # This is to remove combinations of low signal + high falling speeds,
        # which are mainly observed in situations of false signal.
        clutter_mask = (ds.vel.values + ds.Ze.values * (config['clutter_m'] / 60.)) > config['clutter_c']
        vmask *= clutter_mask

    # ----------------------------------
    # Apply layer depended Virga-sniffer
    # ----------------------------------
    # assign layered masks for layer depended masking
    vmask_layer = np.full((*vmask.shape, cbh.layer.size), False)
    cmask_layer = np.full((*vmask.shape, cbh.layer.size), False)
    for itime in range(vmask.shape[0]):
        ilowercloudtop = 0
        for ilayer in range(cbh.shape[1]):
            icloudbase = idxs_cbh[itime, ilayer]
            icloudtop = idxs_cth[itime, ilayer]
            cmask_layer[itime, icloudbase:icloudtop, ilayer] = cmask[itime, icloudbase:icloudtop]
            vmask_layer[itime, ilowercloudtop:icloudbase, ilayer] = vmask[itime, ilowercloudtop:icloudbase]
            # update cloudtop from lower level
            ilowercloudtop = icloudtop

    if config['mask_rain']:
        # Use ancillary rain sensor data to remove detected virga if rain
        # is observed at surface.
        # Apply only to lowest range-gate
        vmask_layer[:, :, 0] *= _expand_mask(~ds.flag_surface_rain.values, vmask.shape[1])  # True if no rain at sfc

    if config['mask_zet']:
        # Check if Signal of first range gate is below threshold.
        # e.g., if larger Signal, than this virga is considered rain
        # Apply only to lowest range-gate
        ze_threshold = ds.Ze.values[:, 0] < config['ze_thres']
        ze_threshold += np.isnan(ds.Ze.values[:, 0])  # add nans again
        vmask_layer[:, :, 0] *= _expand_mask(ze_threshold, vmask.shape[1])

    if config['mask_minrg']:
        # Remove from Virga mask if column number of range-gates
        # with a signal is lower than mask_minrg
        # At this stage whole column is checked a once.
        # Apply virga from different layer separately
        for ilayer in range(vmask_layer.shape[1]):
            c = np.count_nonzero(vmask_layer[:, :, ilayer], axis=1)
            minrg_mask = c > config['mask_minrg']
            vmask_layer[:, :, ilayer] *= _expand_mask(minrg_mask, vmask.shape[1])

    # Merge layered masks again
    vmask = np.sum(vmask_layer, axis=-1).astype(bool)
    cmask = np.sum(cmask_layer, axis=-1).astype(bool)

    # --------------
    # Prepare Output
    # --------------
    # assign additional layer infos
    virga_depth_rgmid = np.zeros(cbh.shape)
    virga_depth_rgedge = np.zeros(cbh.shape)
    idxs_vth = np.full(cbh.shape, -1)
    idxs_vbh = np.full(cbh.shape, -1)
    vth = np.full(cbh.shape, np.nan)
    vbh = np.full(cbh.shape, np.nan)
    for itime in range(vmask.shape[0]):
        for ilayer in range(cbh.shape[1]):
            # get idx where virga is True in a layer
            ivirga = np.argwhere(vmask_layer[itime, :, ilayer]).flatten()
            if len(ivirga) != 0:
                ivirgabase = np.min(ivirga)
                ivirgatop = np.max(ivirga)
                idxs_vbh[itime, ilayer] = ivirgabase
                idxs_vth[itime, ilayer] = ivirgatop
                vth[itime, ilayer] = rgtop[ivirgatop]
                vbh[itime, ilayer] = rgbase[ivirgabase]
                virga_depth_rgmid[itime, ilayer] = rgmid[ivirgatop] - rgmid[ivirgabase]
                virga_depth_rgedge[itime, ilayer] = rgtop[ivirgatop] - rgbase[ivirgabase]

    output_dataset = xr.Dataset({"flag_virga": (('time', 'range'), vmask),
                                 "flag_virga_layer": (('time', 'range', 'layer'), vmask_layer),
                                 "flag_cloud": (('time', 'range'), cmask),
                                 "flag_cloud_layer": (('time', 'range', 'layer'), cmask_layer),
                                 "virga_depth": (('time', 'layer'), virga_depth_rgedge),
                                 "virga_depth_rgmid": (('time', 'layer'), virga_depth_rgmid),
                                 "cloud_depth": (('time', 'layer'), (cth - cbh).data),
                                 "virga_top_rg": (('time', 'layer'), idxs_vth),
                                 "virga_base_rg": (('time', 'layer'), idxs_vbh),
                                 "cloud_top_rg": (('time', 'layer'), idxs_cth),
                                 "cloud_base_rg": (('time', 'layer'), idxs_cbh),
                                 'virga_top_height': (('time', 'layer'), vth),
                                 'virga_base_height': (('time', 'layer'), vbh),
                                 'cloud_top_height': (('time', 'layer'), cth.data),
                                 'cloud_base_height': cbh,
                                 'Ze': ds.Ze,
                                 'vel': ds.vel,
                                 'flag_lcl_filled': ('time', idx_lcl),
                                 'flag_cbh_interpolated': (('time', 'layer'), idx_fill),
                                 'flag_surface_rain': ds.flag_surface_rain},
                                coords={'range_top': ('range', rgtop),
                                        'range_base': ('range', rgbase)})

    # assign metadata
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "nc-config.json")
    with open(filename) as json_file:
        nc_meta = json.load(json_file)

    for var in output_dataset.keys():
        output_dataset[var].attrs.update(nc_meta[var])

    return output_dataset
