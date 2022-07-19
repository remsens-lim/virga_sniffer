# Setup

- install via pip - github

```{code-block} python
---
lineno-start: 1
caption: Imports
---
import xarray as xr
from virga_sniffer import virga_mask as mvirga
```



(configuration)=
## Configuration
The Virga-Sniffer utilizes a variety of flags and thresholds to detect virga from the given input data.

(cfg_flag)=
### Flags
Flags are boolean values which control certain functionality of the Virga-Sniffer:
 - **require_cbh = True**: If **True**, detected virga always have to be attributed to a cloud-base height (CBH) value. This prevents False-positive detection of virga which actually would be a cloud in case of data-gaps in the ceilometer CBH data. If **False**, this requirement is dropped and virga events, where the cloud is already gone, can be detected. It is highly recommended to use **True** when working with multi-layer CBH, as gaps in ceilometer data are introduced by clouds in the lower layers. To circumvent the issue of ceilometer data gaps, one can adjust CBH-layer fill (*cbh_layer_fill_...*) thresholds. The default is **True**.
 - **mask_vel = True**: If **True**, switches on the restriction of the virga mask to data points where the doppler velocity is below the value of **vel_thres** threshold (see {ref}`figure below <fig-ze-vs-vel>`). Therefore if 
   **True**, 
   the doppler velocity (**vel**) has to be included in the input dataset. The default is **True**. 
 - **mask_clutter = True**: If **True**, switches on the restriction of the virga mask to data points where the following dependency is fulfilled:
   ```
   vel > -clutter_m * (Ze / 60 dBz) + clutter_c
   ```
   where *vel* and *Ze* denotes the input doppler velocity [ms-1] and radar reflectivity [dBz], respectively. The reflectivity is scaled for convenience by 60 dBz (as -60dBz is minimum valid 
   value of LIMRAD94). The values **clutter_m** [ms-1] and **clutter_c** [ms-1] are 
   slope and intercept of the threshold line. A data point is considered virga, only if the above equation is fulfilled. With default configuration (**clutter_m = 4** and **clutter_c = -8**) unusual 
   combinations of low reflectivity while high falling speed hydro meteors are dropped (see {ref}`figure below <fig-ze-vs-vel>`). The default is **True**, therefore, the doppler velocity (*vel*) 
   has to be included in 
   the 
   input 
   dataset.
   ```{figure} ../docs/images/vs_demonstration_ze_vs_vel.png
   :name: fig-ze-vs-vel
   2D-Histrogram of radar reflectivity and doppler velocity of LIMRAD94 on RV-Meteor at EUREC4A campaign at 2020-01-17. Red shaded areas show, when virga is not considered due to **mask_vel** and 
   **mask_clutter** using the default configuration of **vel_thres=0**, **clutter_m=4** and **clutter_c=-8**.
   ```
   Therefore, **mask_vel** and **mask_clutter** can be used to explicitly mask a certain interval of interest of doppler velocity or radar reflectivity for virga statistics.
 - **mask_rain = True**: If **True**, this flag switches on the use of the *flag_surface_rain* variable from the input dataset in order to consider virga only, if the no rain is observed at the 
   surface. This is applied to the lowest present cloud layer at given time. Therefore, higher layer virga events will not be masked if rain is detected at the surface.
 - **mask_rain_ze**: Similar to **mask_rain**, but instead of using *flag_surface_rain* from the input data, the radar reflectivity at the lowest range-gate is tested against the **ze_thres** 
   threshold in order to estimate if precipitation will reach the surface.

(cfg_thres)=
### Thresholds
Virga detection specific thresholds:
 - **minimum_rangegate_number = 2**:
 - **ze_max_gap = 150 m**: From each cloud-base layer, the detection of cloud advances upwards. Cloud-top heights are assigned below the first gap larger than **ze_max_gap**. The cloud mask is 
   always applied between cloud-base and detected cloud-top. The default value is 150m.
 - **virga_max_gap = 700 m**: From each cloud-base layer, the detection of virga advances downwards. Virga is detected until a gap (nan-value) of radar-reflectivity larger than **virga_max_gap** 
   occurs. The default value is 700m to also capture virga in fall streaks relatively far below the cloud base, but mask out any clutter or not identified cloud close to the surface or lower cloud 
   layer.
 - **vel_thres = 0 ms-1**: Defines the doppler velocity threshold. If **vel_mask** is set to **True** (default), an datapoint is considered virga only if the doppler velocity is below this threshold. 
   The 
   default value is 0 [ms-1], therefore only falling hydro meteors are considered virga.
 - **ze_thres = 0 dBz**: This threshold is applied if **mask_rain_ze = True**. If the value of radar reflectivity of the lowest range is larger than **ze_thres**, precipitation is assumed to reach 
   the ground and not considered virga in the lowest cloud layer. 
 - **clutter_m = 4 ms-1**: Slope of the linear masking dependency (see **mask_clutter**).
 - **clutter_c = -8 ms-1**: Intercept of the linear masking dependency (see **mask_clutter**).

Cloud-base preprocessing specific thresholds:
 - **cbh_smooth_window = 60 s**: Size of the window for the median-filter smoothing of cloud-base height and cloud-top values.
 - **lcl_smooth_window = 300 s**: Size of the window for the median-filter smoothing of the lifting condensation level data.
 - **cbh_layer_thres = 500 m**: Threshold used for split
 - **cbh_clean_thres = 0.05 [0-1]**:
 - **cbh_fill_limit = 60 s**:

(cfg_spec)=
### Special configuration
Appart from thresholds, the cloud-base preprocessing is controlled by specialized configuration:
 - **cbh_fill_method = "slinear"**:
 - **cbh_processing = [1, 0, 2, 0, 3, 1, 0, 2, 0, 3, 4]**:

## Demonstration
An interactive demonstration of the virga-sniffer configuration is available as jupyter notebook (check *examples* directory) and viewable here: [nbviewer](https://nbviewer.
org/github/jonas-witthuhn/virga_sniffer/example/virga-sniffer_config_demo.ipynb).

