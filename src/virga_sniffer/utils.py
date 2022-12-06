"""
utils.py
===============
Basic utility functions used for virga_detection.py
"""
from typing import Iterable, Union, Tuple
from numpy.typing import NDArray
import xarray as xr
import numpy as np
from scipy.ndimage import median_filter
import scipy.special


def get_gapidx(mask, fill_value=None):
    """
    Retrieve the index of gaps (False values) in a boolean mask.

    Parameters
    ----------
    mask: array_like
        Boolean mask of shape (M,N)
    fill_value: int or None, optional
        This value is used to pad the output array (axis 1). The default is M-1.
    Returns
    -------
    gapsidx: array_like, (M,L)
        The index of gaps along dimension N

    """
    if fill_value is None:
        fill_value = mask.shape[1] - 1
    
    Ncounts = np.zeros(mask.shape[0], dtype=int)
    # get index where gaps (False values) occur
    # reshape to 2D array (time, gapidxs)
    wmask = np.argwhere(~mask)
    times, counts = np.unique(wmask[:,0], return_counts=True)
    Ncounts[times] = counts
    fill_mask = Ncounts[:,None] > np.arange(counts.max())
    gapsidx = np.full(fill_mask.shape, int(fill_value), dtype=int)
    gapsidx[fill_mask] = wmask[:,1]
    return gapsidx

def get_firstgap_dn(gapidx, idxs, fill_value=None):
    """ Get the closest gap idx below idx """
    if fill_value is None:
        fill_value = -1
        
    Nshape = gapidx.shape
    itim = np.arange(Nshape[0])

    firstgap_dn = np.full(idxs.shape, fill_value)
    ifirstgaptmp = np.full(Nshape[0],-1)

    for ilayer in range(idxs.shape[1]):
        # find closest index of a gap below a CBH index (idxs) 
        ifirstgap = (gapidx.T<idxs[:,ilayer]).sum(axis=0) - 1
        ifirstgap[ifirstgap>=Nshape[1]] = -1
        
        # selection, handling layer overlapp
        tsel = np.ones(Nshape[0]).astype(bool)
        # only assign if within a layer
        if ilayer > 0: # not first layer
            tsel = ifirstgap > idxs[:,ilayer-1]
        # not assign, if already in the lower layer
        tsel[ifirstgap==ifirstgaptmp] = False
        ifirstgaptmp = ifirstgap
        
        firstgap_dn[tsel,ilayer] = gapidx[itim[tsel],ifirstgap[tsel]] 

    return firstgap_dn

def get_firstgap_up(gapidx, idxs, fill_value=None):
    """ Get the closest gap idx above idx """
    if fill_value is None:
        fill_value = -1
    
    Nshape = gapidx.shape
    itim = np.arange(Nshape[0])
    firstgap_up = np.full(idxs.shape, fill_value)
    ifirstgaptmp = np.full(Nshape[0],-1)
    for ilayer in range(idxs.shape[1]):
        # find closest index of a gap above a CBH index (idxs) 
        ifirstgap = (gapidx.T<=idxs[:,ilayer]).sum(axis=0)
        ifirstgap[ifirstgap>=Nshape[1]] = Nshape[1]-1
        
        # selection, handling layer overlapp
        tsel = np.ones(Nshape[0]).astype(bool)
        # only assign if within a layer
        if ilayer != idxs.shape[1]-1: # not last layer
            tsel = ifirstgap <= idxs[:,ilayer+1]
        # not assign, if already in the lower layer
        tsel[ifirstgap==ifirstgaptmp] = False
        ifirstgaptmp = ifirstgap
            
        firstgap_up[tsel,ilayer] = gapidx[itim[tsel],ifirstgap[tsel]] -1
    return firstgap_up

def fill_mask_gaps(mask: NDArray[bool],
                   altitude: NDArray[float],
                   max_gap: float,
                   idxs_true: NDArray[int]) -> NDArray[bool]:
    """
    Fill vertical gaps in a boolean mask
    Parameters
    ----------
    mask
    altitude
    max_gap
    idxs_true

    Returns
    -------
    numpy.ndarray
    """

    np.put_along_axis(mask, idxs_true, values=True, axis=1)

    # switch to xarray to make use of their functions
    mask = xr.DataArray(mask, dims=('time', 'range'),
                        coords={'time': np.arange(mask.shape[0]),
                                'range': altitude})
    mask = mask.where(mask)  # False to nan
    mask_int = mask.dropna(dim='time', thresh=2)  # remove nan for interpolation

    # interpolate mask, to fill small gaps (<layer_threshold) in virga
    mask_int = mask_int.interpolate_na(dim='range',
                                       method='nearest',
                                       max_gap=max_gap,
                                       bounds_error=False,
                                       fill_value=np.nan)

    # map back to original array
    mask = mask_int.combine_first(mask)
    # convert nan back to False, now we have a mask with True==virga, False==no-virga
    # but small gaps in original mask are filled with True
    # mask = mask.fillna(False).values[:, 1:].astype(bool)
    mask = mask.fillna(False).values.astype(bool)
    return mask


def medfilt(input_data: NDArray, freq: float, window: float) -> NDArray:
    """
    Apply scipy.ndimage.median_filter of a certain time window to the data

    Parameters
    ----------
    input_data: array_like
        The input array of timewise continuous data
    freq: float
        The frequency of the data samples [s-1]
    window: float
        The time window to apply the filter [s].

    Returns
    -------
    numpy.ndarray
        Filtered array, with the same shape as input data.
    """
    # import numpy as np
    # from scipy.ndimage import median_filter
    # calculate window size
    window_size = int(np.round(freq * window, 0))
    # size have to be odd
    if window_size % 2 == 0:
        window_size += 1
    # apply filter
    data_filtered = median_filter(input_data, size=window_size)
    return data_filtered


def calc_lcl(p: Union[float, Iterable[float]],
             T: Union[float, Iterable[float]],
             rh:  Union[None, float, Iterable[float]] = None,
             rhl:  Union[None, float, Iterable[float]] = None,
             rhs:  Union[None, float, Iterable[float]] = None,
             return_ldl: bool = False,
             return_min_lcl_ldl: bool = False) -> Union[float, np.ndarray]:
    """
    Calculate lifting condensation level (LCL) from surface pressure, temperature and humidity observations.
    Adapted from Romps (2017) adding numpy array functionality.

    References
    ----------
    [1] Romps, D. M. (2017). Exact Expression for the Lifting Condensation Level, Journal of the Atmospheric Sciences, 74(12), 3891-3900. :doi:`10.1175/JAS-D-17-0102.1`

    Parameters
    ----------
    p: float or array_like
        Surface air pressure `p` in units Pascals [PA]
    T: float or array_like
        Surface Air temperature `T` in units Kelvin [K].
    rh, rhl, rhs: float or array_like or None, optional
        Exactly one of rh, rhl, and rhs must be specified.
         - `rh`: Relative humidity (dimensionless, from 0 to 1) with respect to liquid water if T >= 273.15 K and with respect to ice if T < 273.15 K.
         - `rhl`: Same as rh but solely with respect to liquid water.
         - `rhs`: Same as rh but solely with respect to ice.

         If array_like, requires same shape as `p` and `T`. The default is None.
    return_ldl: bool, optional
        If True, the lifting deposition level (LDL) is returned instead of LCL.
        The default is False.
    return_min_lcl_ldl: bool, optional
        If true, the minimum of the LCL and LDL is returned. The default is False.

    Returns
    -------
    float or numpy.ndarray
        Result equals:
            - LCL if `return_ldl` == False and `return_min_lcl_ldl` == False,
            - LDL if `return_ld` == True and `return_min_lcl_ldl` == False,
            - Min(LCL,LDL) if `return_min_lcl_ldl` == True and `return_ldl` == False.

        Same shape as `p`.
    """

    # Parameters
    Ttrip = 273.16  # K
    ptrip = 611.65  # Pa
    E0v = 2.3740e6  # J/kg
    E0s = 0.3337e6  # J/kg
    ggr = 9.81  # m/s^2
    rgasa = 287.04  # J/kg/K
    rgasv = 461  # J/kg/K
    cva = 719  # J/kg/K
    cvv = 1418  # J/kg/K
    cvl = 4119  # J/kg/K
    cvs = 1861  # J/kg/K
    cpa = cva + rgasa
    cpv = cvv + rgasv

    # The saturation vapor pressure over liquid water
    def pvstarl(T):
        satp = ptrip * (T / Ttrip) ** ((cpv - cvl) / rgasv)
        satp *= np.exp((E0v - (cvv - cvl) * Ttrip) / rgasv * (1 / Ttrip - 1 / T))
        return satp

    # The saturation vapor pressure over solid ice
    def pvstars(T):
        satp = ptrip * (T / Ttrip) ** ((cpv - cvs) / rgasv)
        satp *= np.exp((E0v + E0s - (cvv - cvs) * Ttrip) / rgasv * (1 / Ttrip - 1 / T))
        return satp

    p = np.array(p)
    T = np.array(T)

    # Calculate pv from rh, rhl, or rhs
    if not ((rh is not None) + (rhl is not None) + (rhs is not None)):
        exit('Error in lcl: Exactly one of rh, rhl, and rhs must be specified')
    rh_counter = 0

    # check temperature above triple point
    Ctrip = T > Ttrip

    if rh is not None:
        rh = np.array(rh)
        # The variable rh is assumed to be
        # with respect to liquid if T > Ttrip and
        # with respect to solid if T < Ttrip
        pv = np.ones(len(T)) * np.nan
        pv[Ctrip] = rh[Ctrip] * pvstarl(T[Ctrip])
        pv[~Ctrip] = rh[~Ctrip] * pvstars(T[~Ctrip])
        rhl = pv / pvstarl(T)
        rhs = pv / pvstars(T)

    elif rhl is not None:
        rhl = np.array(rhl)
        pv = rhl * pvstarl(T)
        rhs = pv / pvstars(T)
        rh = np.ones(len(T)) * np.nan
        rh[Ctrip] = rhl[Ctrip]
        rh[~Ctrip] = rhs[~Ctrip]

    elif rhs is not None:
        rhs = np.array(rhs)
        pv = rhs * pvstars(T)
        rhl = pv / pvstarl(T)
        rh = np.ones(len(T)) * np.nan
        rh[Ctrip] = rhl[Ctrip]
        rh[~Ctrip] = rhs[~Ctrip]
    else:
        raise Exception("At least one of rh, rhl or rhs must be not None")

    # Calculate lcl_liquid and lcl_solid
    qv = rgasa * pv / (rgasv * p + (rgasa - rgasv) * pv)
    rgasm = (1 - qv) * rgasa + qv * rgasv
    cpm = (1 - qv) * cpa + qv * cpv

    lcl0 = cpm * T / ggr
    lcl0[pv > p] = np.nan

    aL = -(cpv - cvl) / rgasv + cpm / rgasm
    bL = -(E0v - (cvv - cvl) * Ttrip) / (rgasv * T)
    cL = pv / pvstarl(T) * np.exp(-(E0v - (cvv - cvl) * Ttrip) / (rgasv * T))
    aS = -(cpv - cvs) / rgasv + cpm / rgasm
    bS = -(E0v + E0s - (cvv - cvs) * Ttrip) / (rgasv * T)
    cS = pv / pvstars(T) * np.exp(-(E0v + E0s - (cvv - cvs) * Ttrip) / (rgasv * T))

    lcl = lcl0 * (1 - bL / (aL * scipy.special.lambertw(bL / aL * cL ** (1 / aL), -1).real))
    ldl = lcl0 * (1 - bS / (aS * scipy.special.lambertw(bS / aS * cS ** (1 / aS), -1).real))

    lcl[rh == 0] = lcl0[rh == 0]
    ldl[rh == 0] = lcl0[rh == 0]

    # Return either lcl or ldl
    if return_ldl and return_min_lcl_ldl:
        exit('return_ldl and return_min_lcl_ldl cannot both be true')
    elif return_ldl:
        return ldl
    elif return_min_lcl_ldl:
        return np.min(np.vstack(lcl, ldl), axis=0)
    else:
        return lcl
