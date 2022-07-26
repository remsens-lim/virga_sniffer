# Data
(input)=
## Input dataset
The Virga-Sniffer virga detection method receives as bare minimum input the *radar reflectivity* and values of cloud-base height. Ancillary data is accepted (and highly recommended) to refine the 
virga detection. Optional input data are the *doppler velocity*, *rain at surface flag*, *lifting condensation level*. The input dataset has to be provided in the form of a [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html)
(see e.g., [virga_mask](virga_detection)). In the following, the input 
variables, their dimensions and purpose are described:

### Radar reflectivity
 - **name**: *Ze*
 - **units**: dBz
 - **dimensions**: (*time*, *range*)
 - **optional**: False

 *Ze* is used for basic precipitation and cloud detection. The *Ze* valid value mask (True if not nan-value) is used along with *cloud_base_height*. If [mask_rain_ze](cfg_flag) is True, values 
 of *Ze* at the lowest range-gate are compared to [ze_thres](cfg_thres). If [mask_clutter](cfg_flag) is True, *Ze* and *vel* are used to [refine the virga mask](mvel).
```{note}
Actually the unit of *Ze* doesn't matter as long as [mask_rain_ze](cfg_flag) and [mask_clutter](cfg_flag) are False. Even if [mask_rain_ze](cfg_flag) is True, *Ze* is required to have the same 
unit as [ze_thres](cfg_thres). If [mask_clutter](cfg_flag) is True, [clutter_m](cfg_thres) has to be carefully chosen in order to match the unit of *Ze*.
```

### Cloud-base height
 - **name**: *cloud_base_height*
 - **units**: m
 - **dimensions**: (*time*, *layer*)
 - **optional**: False

The cloud-base height values are required to separate between cloud and precipitation in the [baseline detection process](detection). The dimension *layer* is required, even if only 1D data is 
available. Regardless, the data is smoothed and preprocessed based on the 
configuration setting  [cbh_smooth_window](cfg_thres) and [cbh_processing](cfg_spec), respectively. The processed cloud-base height data might have an increased number of layers (see 
[preprocessing](preprocessing)). 
```{note}
Cloud-base height preprocessing can be turned off by setting [cbh_smooth_window](cfg_thres) to 0 and [cbh_processing](cfg_spec) to [].
```

### Doppler velocity
 - **name**: *vel*
 - **units**: m s-1
 - **dimensions**: (*time*, *range*)
 - **optional**: True

The doppler velocity is required only if at least on of  [mask_vel](cfg_flag) or [mask_clutter](cfg_flag) is True. This data is used for [virga mask refinement](mvel).

### Surface rain flag
 - **name**: *flag_surface_rain*
 - **units**: bool
 - **dimensions**: (*time*)
 - **optional**: True

A surface rain flag is required only if  [mask_rain](cfg_flag) is True. This flag is used to drop the lowest layer of virga if [rain is detected at the surface](mrain). 

### Lifting condensation level
 - **name**: *lcl*
 - **units**: m
 - **dimensions**: (*time*)
 - **optional**: True

The lifting condensation level is required only if [cbh_processing](cfg_spec) includes a 3, which means that this data will be replacing nan values of the lowest layer of cloud base height. Adding 
this data helps to define a continuous cloud base to aid the virga detection. 
```{note}
The lifting condensation level can be calculated from surface observations of air pressure, temperature and humidity, using the [Virga-Sniffer utils](utils). 
```


(output)=
## Output dataset
The results of the virga and cloud detection are stored in the output dataset as boolean flags with the same dimensions as the radar reflectivity input data. In addition, the processed cloud-base and -top heights are stored, as well as some basic characteristics such as cloud and virga depth for each column. The output is provided as a
[xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) - names, units and description are provided in the following:

### Virga flag
 - **name**: *flag_virga*, *flag_virga_layer*
 - **units**: bool
 - **dimensions**: (*time*, *range*), (*time*, *range*, *layer*)

With the same dimensions as the input *Ze*, the boolean flags are True if virga is detected. In *flag_virga_layer*, the *flag_virga* mask is separated by each cloud layer, so that only virga 
attached to the cloud base of the respective layer is True. 
```
flag_virga = flag_virga.sum(axis=-1).astype(bool) 
```

### Cloud flag
 - **name**: *flag_cloud*, *flag_cloud_layer*
 - **units**: bool
 - **dimensions**: (*time*, *range*), (*time*, *range*, *layer*)

Similar to the virga flag, this flag is True if a cloud is detected.

### Virga depth
 - **name**: *virga_depth*, *virga_depth_maximum_extend*
 - **units**: m
 - **dimensions**: (*time*, *layer*)

*virga_depth* and *virga_depth_maximum_extend* are measures of the vertical extend of the virga, but calculated differently. *virga_depth* denotes the sum of the vertical extend of all range-gates 
where virga is detected, thus excluding gaps introduced by [virga_max_gap](cfg_thres). *virga_depth_maximum_extend* includes these gaps, by being calculated by ```virga_top_height - virga_base_height```. Therefore, *virga_depth* should be used, when calculating volumetric characteristics, such as the liquid water path, and *virga_depth_maximum_extend* for geometric characteristics.

### Cloud depth
 - **name**: *cloud_depth*
 - **units**: m
 - **dimensions**: (*time*, *layer*)

The vertical extend of the cloud, calculated by ```cloud_top_height - cloud_base_height```.

### Height values
 - **name**: *virga_top_height*, *virga_base_height*, *cloud_top_height*, *cloud_base_height*
 - **units**: m
 - **dimensions**: (*time*, *layer*)

The data of virga- and cloud-base and -tops as processed from the input data 
([cloud-base preprocessing](preprocessing) and [precip. and cloud detection](detection)).  

### Range-gate indices
 - **name**: *virga_top_rg*, *virga_base_rg*, *cloud_top_rg*, *cloud_base_rg*
 - **units**: -
 - **dimensions**: (*time*, *layer*)

The index of *virga_top_height*, *virga_base_height*, *cloud_top_height* and *cloud_base_height* in the *range* dimension.

### Passed input data
 - **name**: *Ze*, *vel*, *flag_surface_rain*
 - **units**: dBz, m s-1, bool
 - **dimensions**: (*time*, *range*), (*time*, *range*), (*time*)

Some of the input data is passed into the output dataset.

### Processing flags
 - **name**: *flag_lcl_filled*, *flag_cbh_interpolated*
 - **units**: bool, bool
 - **dimensions**: (*time*), (*time*, *layer*)

During the [cloud-base preprocessing](preprocessing), the lower layer of the cloud-base height is filled by the lifting condensation level (if provided), and gaps in cloud-base layers are filled 
according to the configuration. These flags indicate which data of the output *cloud_base_height* is filled with the lifting condensation level (*flag_lcl_filled*), or interpolated by the filling 
method (*flag_cbh_interpolated*).


(plotting)=
### Plotting
The output dataset is enhanced with 
[xarray.Datast accessors](https://docs.xarray.dev/en/stable/generated/xarray.register_dataset_accessor.html)
to add plotting capability (see [vsplot](vsplot)).

The methods can be accessed with:
```{code-block} python
from virga_sniffer import virga_mask as mvirga
# run Virga-Sniffer
dsout = mvirga(input_ds)
# plot a full quicklook
fig,axs = dsp.vsplot.quicklook_full(radar='LIMRAD94')
```

This produces:
```{figure} images/RV_Meteor_virga-timeseries_20200219_18-21_v0.3.4.png
```
