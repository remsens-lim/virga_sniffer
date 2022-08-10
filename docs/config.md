(configuration)=
# Configuration
The Virga-Sniffer utilizes a variety of flags and thresholds to detect virga from the given input data. The configuration can be set via the *config* dictionary, which will be merged with the 
default values. In the following all configuration flags, thresholds and settings, and their respective default values are described. 
See also the [configuration demo](https://mybinder.org/v2/gh/remsens-lim/virga_sniffer/main?filepath=example/virga-sniffer_config_demo.ipynb).

(cfg_flag)=
## Flags
Flags are boolean values which control certain functionality of the Virga-Sniffer:
 - **require_cbh = True**: If **True**, detected virga always have to be attributed to a cloud-base height (CBH) value. This prevents False-positive detection of virga which actually would be a cloud in case of data-gaps in the ceilometer CBH data. If **False**, this requirement is dropped and virga events, where the cloud is already gone, can be detected. The default is **True**.
   ```{note}
   It is highly recommended to use **True** when working with multi-layer CBH, as gaps in ceilometer data are introduced by clouds in the lower layers. To circumvent the issue of ceilometer data gaps, one can adjust CBH-layer fill (*cbh_layer_fill_...*) thresholds.
   ```
 - **mask_vel = True**: If **True**, switches on the restriction of the virga mask to data points where the Doppler velocity is below the value of **vel_thres** threshold (see [Doppler velocity 
   based refinement](mvel)). Therefore if 
   **True**, 
   the Doppler velocity (**vel**) has to be included in the input dataset. The default is **True**. 
 - **mask_clutter = True**: If **True**, switches on the restriction of the virga mask to data points where the following dependency is fulfilled:
   ```
   vel > -clutter_m * (Ze / 60 dBz) + clutter_c
   ```
   where *vel* and *Ze* denotes the input Doppler velocity [ms-1] and radar reflectivity [dBz], respectively (see [Doppler velocity based refinement](mvel)). The default is **True**, therefore, 
   the Doppler velocity (*vel*) has to be included in 
   the input dataset.
 - **mask_rain = True**: If **True**, this flag switches on the use of the *flag_surface_rain* variable from the input dataset in order to consider virga only, if no rain is observed at the 
   surface. This is applied to the lowest present cloud layer at any given time. Therefore, higher layer virga events will not be masked if rain is detected at the surface.
 - **mask_rain_ze**: Similar to **mask_rain**, but instead of using *flag_surface_rain* from the input data, the radar reflectivity at the lowest range gate is tested against the **ze_thres** 
   threshold in order to estimate if precipitation will reach the surface.

(cfg_thres)=
## Thresholds
Virga detection specific thresholds:
 - **minimum_rangegate_number = 2**:
 - **ze_max_gap = 150 m**: From each cloud-base layer, the detection of cloud advances upwards. Cloud-top heights are assigned below the first gap larger than **ze_max_gap**. The cloud mask is 
   always applied between cloud-base and detected cloud-top. The default value is 150m.
 - **virga_max_gap = 700 m**: From each cloud-base layer, the detection of virga advances downwards. Virga is detected until a gap (nan-value) of radar-reflectivity larger than **virga_max_gap** 
   occurs. The default value is 700m to also capture virga in fall streaks relatively far below the cloud base, but mask out any clutter or not identified cloud close to the surface or lower cloud 
   layer.
 - **vel_thres = 0 ms-1**: Defines the Doppler velocity threshold. If **vel_mask** is set to **True** (default), a datapoint is considered virga only if the Doppler velocity is below this threshold. 
   The 
   default value is 0 [ms-1], therefore only falling hydrometeors are considered virga.
 - **ze_thres = 0 dBz**: This threshold is applied if **mask_rain_ze = True**. If the value of radar reflectivity of the lowest range is larger than **ze_thres**, precipitation is assumed to reach 
   the ground and not considered virga in the lowest cloud layer. 
 - **clutter_m = 4 ms-1**: Slope of the linear masking dependency (see **mask_clutter**).
 - **clutter_c = -8 ms-1**: Intercept of the linear masking dependency (see **mask_clutter**).

Cloud-base preprocessing specific thresholds:
 - **cbh_smooth_window = 60 s**: Size of the window for the median-filter smoothing of cloud-base height and cloud-top values.
 - **lcl_smooth_window = 300 s**: Size of the window for the median-filter smoothing of the lifting condensation level data.
 - **cbh_layer_thres = 500 m**: Threshold used for splitting cloud-base layers during the [preprocessing](preprocessing).
 - **cbh_clean_thres = 0.05 [0-1]**: Threshold used for cleaning cloud-base layers during the [preprocessing](preprocessing).
 - **cbh_fill_limit = 60 s**: Defines the maximum gap within cloud layers to be filled by interpolation method **cbh_fill_method** (see [preprocessing](preprocessing)).

(cfg_spec)=
## Special configuration
Apart from thresholds, the cloud-base preprocessing is controlled by specialized configuration:
 - **cbh_fill_method = "slinear"**: This defines the method of filling cloud-base gaps smaller than **cbh_fill_limit**. Possible methods are
   - [ffill](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.ffill.html)
   - [bfill](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.bfill.html)
   - *nearest*, *zero*, *slinear*, *quadratic*, *cubic*, *polynomial* as accepted by
     [xarray.interpolate_na](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interpolate_na.html)
 - **cbh_processing = [1, 0, 2, 0, 3, 1, 0, 2, 0, 3, 4]**: This list defines the methods applied for [preprocessing](preprocessing). 

## Demonstration
An interactive demonstration of the Virga-Sniffer configuration is available as a jupyter notebook (check *examples* directory, see [Setup/Configuration-Demonstration](cfg-demo)).
The notebook can be viewed in
[binder](https://mybinder.org/v2/gh/remsens-lim/virga_sniffer/main?filepath=example/virga-sniffer_config_demo.ipynb).
