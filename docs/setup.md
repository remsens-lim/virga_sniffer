# Setup
## Virga-Sniffer

The Virga-Sniffer can be installed via pip and the github repository:
```
python -m pip install git+https://github.com/remsens-lim/virga_sniffer.git#egg=virga_sniffer
```
The following packages will be installed: 
 - virga_sniffer
 - numpy
 - scipy
 - pandas
 - xarray
 - netcdf4
 - bottleneck
 - matplotlib

After installing, the Virga-Sniffer module can be accessed as demonstrated below:

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
lineno-start: 5
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
lineno-start: 29
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



(cfg-demo)=
## Configuration Demonstration
To view the interactive demonstration of the Virga-Sniffer configuration, the example notebook can be run in
[binder](https://mybinder.org/v2/gh/remsens-lim/virga_sniffer/main?filepath=example/virga-sniffer_config_demo.ipynb).

To run the example jupyter notebook yourself, additional dependencies can be installed via:
```
python -m pip install git+https://github.com/remsens-lim/virga_sniffer.git#egg=virga_sniffer[example]
```
This will install additionally:
 - jupyter
 - jupyterlab
 - ipywidgets
 - ipympl

To run the demonstration of the virga-sniffer configuration, these files have to be downloaded from the repository:
 - [virga-sniffer_config_demo.ipynb](../example/virga-sniffer_config_demo.ipynb)
 - [vs_interactive.py](../example/vs_interactive.py)
 - [test_data](../example/test_data/2020-01-24_00_virga-sniffer_input.nc)
 
Or, the zipped data can be downloaded: 
```{eval-rst} 
:download:`example.zip <../example/example.zip>`
```

Once installed and downloaded, the jupyter notebook can be run. The configuration of the Virga-Sniffer can be tested with the interactive widgets:
```{figure} images/vs-jlab-demo.jpg
```

The result will be shown as a [full quicklook](vsplot).

## Building the docs
To build the docs, additional dependencies can be installed via:
```
python -m pip install git+https://github.com/remsens-lim/virga_sniffer.git#egg=virga_sniffer[docs]
```

Then, the docs can be build with 
(see [sphinx documentation](https://www.sphinx-doc.org/en/master/man/sphinx-build.html)):
```
cd <path-to-virga_sniffer>/docs/
make html
```
or
```
cd <path-to-virga_sniffer>/docs/
make latex
```







