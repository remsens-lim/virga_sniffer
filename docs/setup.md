# Setup

- install via pip - github

```{code-block} python
---
lineno-start: 1
caption: Imports
---
import xarray as xr
import numpy as np
from virga_sniffer import virga_mask as mvirga
from virga_sniffer.utils import calc_lcl
```
```{code-block} python
---
lineno-start: 4
caption: Preparations
---
# Prepare the configuration
# The default configuration will be used if not set manually
config = {"ze_thres": -10}

# The lifting condensation level can be calculated using virga_sniffer.utils:
lcl = calc_lcl(p = surface_air_pressure.data, # [Pa]
               T = surface_air_temperature.data, # [K]
               rh = surfave_relative_humidity.data # [0-1]
               )

# Preparing the Input
input_ds = xr.Dataset({
      "Ze": (('time', 'range'), radar_reflectivity.data),
      # cloud_base data needs two dimensions, even if its only one layer
      "cloud_base_height": (('time', 'layer'), np.array(cloud_base.data)[:,np.newaxis]),
      "flag_surface_rain": (('time'), flag_rain_ground.data),
      "vel": (('time', 'range'), mean_doppler_velocity.data),
      "lcl": (('time'), lcl)
   },
   coords={
      "time": ('time', time.data),
      "range":('range', height.data),
      "layer":('layer',np.arange(cloud_base.data.shape[1]))
   })
```

```{code-block} python
---
lineno-start: 28
caption: Run virga detection
---
dsout = mvirga(input_ds)
# store to netCDF
dsout.to_netcdf("<path>.nc")

# Quicklooks
date = dsout.time.values[0]
for dtime in range(0,24,3):
    # Restrict plots to 3 hours
    ssdate = pd.to_datetime(np.datetime64(date)+np.timedelta64(dtime,'h'))
    eedate = pd.to_datetime(np.datetime64(date)+np.timedelta64(dtime+3,'h'))
    intime = dsout.time < np.datetime64(date)+np.timedelta64(dtime+3,'h')
    intime *= dsout.time >= np.datetime64(date)+np.timedelta64(dtime,'h')
    dsp = dsout.sel(time = slice(np.datetime64(date)+np.timedelta64(dtime,'h'),
                                 np.datetime64(date)+np.timedelta64(dtime+3,'h')))
    # use virga_sniffer plotting method
    fig,axs = dsp.vsplot.quicklook_full(radar='LIMRAD94')
    fig.savefig("<path>",dpi=100)
```
```{figure} ../docs/images/vs_demonstration_maxgap_multilayer.jpg
Example quicklook from virga_sniffer.vsplot.quicklook_flag_virga
```




