# Method description
The Virga-Sniffer is a tool to detect *virga* (precipitation which completely evaporates before reaching the surface).
As input source **radar reflectivity** and ceilometer **cloud-base height** observations are mandatory. Optionally but highly recommended are the additional information of **radar Doppler 
velocity**, **lifting condensation level** and **surface rain detection**.

The workflow of the detection scheme as summarized by the flowchart is structured in three parts:
 1. Preprocessing of cloud-base height
 2. Baseline precipitation and cloud detection
 3. (Optional) Virga detection refinement

```{figure} images/flowchart.svg
```

## Input data
The Virga-Sniffer virga detection method receives as bare minimum input the *radar reflectivity* and values of *cloud-base height*. Ancillary data is accepted (and highly recommended) to refine the 
virga detection. Optional input data are the *Doppler velocity*, *rain at surface flag* and *lifting condensation level*. For a detailed descriptions see [Input dataset](input).

(preprocessing)=
## Cloud-base preprocessing
The input cloud-base height (CBH) layer data is preprocessed before it is used for virga detection. Prior to the configurable processing the CBH data is smoothed, which corresponds to processing 
module 4. The remaining processing steps are controlled with the configuration setting of [cbh_processing](cfg_spec), which is a list of integer in the range of 0 to 4:
 - **0: clean and sort**: First each layer of the current CBH dataset with less than [cbh_clean_thres](cfg_thres) [%] number valid (not nan) datapoints are dropped (*clean*). From the remaining 
   layers the mean height of each layer is calculated. The CBH dataset is then re-indexed, sorting the layers in ascending order by mean height (*sort*).
 - **1: split**: The CBH dataset is iterated successively layer by layer. For each layer, outliers according to layer mean +- [cbh_layer_thres](cfg_thres) are identified and pushed to new created 
   layers above/below the current layer. This process is re-iterated until no new layers are created. 
 - **2: merge**: Merging CBH layer data by successively iterating all layers and comparing lower layers to all layers above them. If the distance of the compared layers is smaller than 
   [cbh_layer_thres](cfg_thres), upper layer data will be added to lower layer, or merged by mean value if both layers hold valid data.
 - **3: add lcl**: At the lowest CBH layer, nan-values are filled with lifting-condensation level data (*lcl*) from the input dataset.
   ```{note}
   The lifiting-condensation level input data is always smoothed by running median window of size [lcl_smooth_window](cfg_thres)
   ```
 - **4: smooth**: The CBH layer data is smoothed by applying a running-median window of the size [cbh_smooth_window](cfg_thres).

After all processing steps are applied, the processed CBH data is filled by interpolation according to the configuration [cbh_fill_limit](cfg_thres) and [cbh_fill_method](cfg_spec). This step 
can be disabled by either setting [cbh_fill_limit](cfg_thres) to 0 or [cbh_fill_method](cfg_spec) to None.

(detection)=
## Precipitation & cloud detection
After the preprocessing of CBH, the radar reflectivity values, specifically the boolean mask of valid reflectivity values, is used for the initial step of detecting precipitation, cloud and cloud-top 
heights. This is done by successively iterating the reflectivity mask starting from each cloud-base up- and downwards. Precipitation is detected at each range-gate of valid radar reflectivity 
iterating downward until a gap (nan-value in radar 
reflectivity) occurs, which is larger than the configuration threshold [virga_max_gap](cfg_thres).
Similarly, valid radar reflectivity values from the cloud base upward are marked as clouds until a gap larger than the configuration threshold [ze_max_gap](cfg_thres) occurs. The cloud top height 
value is 
assigned to the range gate value (top of range gate) of the last valid radar reflectivity value below the gap (see {ref}`example sketch <fig-sketch>`). 
```{note}
The detection of clouds is always limited to the area between cloud base and top, while 
virga or precipitation cannot be detected in this range.
```
```{note}
A special case is when there are no gaps in radar reflectivity between the cloud base layers. In this case, the intervening cloud layers are omitted. Therefore, the virga events are connected and 
in this case assigned to the highest contiguous cloud base and associated cloud.
```
The detected cloud-top values are smoothed as cloud-base values are smoothed prior to the cloud-base processing utilizing a rolling median filter of window size defined by the 
[cbh_smooth_window](cfg_thres) threshold.

After this processing step, an index mapping of CTH and CBH values to the upper edge of radar range-gate heights is conducted for further processing. This mapping is used to separate the cloud- and 
virga-mask into cloud layer components. 

Until this point, the identification of clouds and precipitation is solely based on the input variables *cloud_base_height* and *Ze* (radar reflectivity). The virga mask is refined by optional 
masking componentes, enabled via the configuration. The modules [mask_rain_ze](cfg_flag) and [minimum_rg_number](cfg_thres) can be used without the requirement of additional data. If one of [mask_clutter](cfg_flag), [mask_vel](cfg_flag), 
[mask_rain](cfg_flag) is set in the configuration, additional data is required.

```{note}
It is recommended to enable at least one of [mask_rain_ze](cfg_flag) and [mask_rain](cfg_flag), else the Virga-Sniffer turns into a Precipitation-Sniffer.  Using [mask_rain_ze](cfg_flag), virga 
detection can be done using only cloud-base height and radar reflectivity data.
```

Virga and cloud detection is sketched in the {ref}`figure below <fig-sketch>`. Special cases are:
 - **time = 2**: The gap (range-gate (rg) 7-8) is smaller than [virga_max_gap](cfg_thres) to count rg 6 as virga, but rg 6 is dropped due to [minimum_rangegate_number](cfg_thres)=2.
 - **time = 3**: The gap (rg 7-8) is smaller than [virga_max_gap](cfg_thres), therefore rg 3-6 are counted as virga.
 - **time = 4**: The gap (rg 7-11) is larger than [virga_max_gap](cfg_thres), therefore rg 3-6 are not counted as virga. In addition, the gap (rg 17-18) is larger than [ze_max_gap](cfg_thres), 
   therefore rg 19 is not counted as cloud.
 - **time = 5**: Rain is observed at the surface (either by [mask_rain](cfg_flag) or [mask_rain_ze](cfg_flag)), therefore no virga is assigned in this column. 
 - **time = 6**: Same as **time = 5**. In addition, the gap (rg 17) is smaller than [ze_max_gap](cfg_thres), therefore rg 18-19 are counted as cloud.

```{figure} images/example_detection.png
:name: fig-sketch
Example sketch of virga and cloud detection. 
```

## Virga mask refinement
The virga detection can be refined by enabling masking modules via the [Configuration](configuration), some of which require additional input data such as Doppler velocity or surface rain flag.
In order to actually detect virga and exclude rain events, one of *mask_rain* or *mask_rain_ze* have to be enabled. Other optional modules refine the mask by excluding events with undesired 
properties or clutter, so it's highly recommended to enable them.

(mrain)=
### Masking rain events
Rain events can be identified in two ways. First, by providing additional input data (*flag_surface_rain* - True if rain is detected at the surface) by setting [mask_rain](cfg_flag) to **True** in the 
[Configuration](configuration) dictionary. Second, by setting [mask_rain_ze](cfg_flag) to **True** and define a radar reflectivity threshold ([ze_thres](cfg_thres) [dBz]), which is tested against the lowest 
range-gate at every time step. If the threshold is exceeded, the precipitation is assumed to reach the surface.
```{warning}
The threshold [ze_thres](cfg_thres) is strongly depended on the radar system calibration and configuration. Therefore, do not use [mask_rain_ze](cfg_flag)=**True** without carefully choosing 
*ze_thres*.
```

### Count valid data
If [minimum_rangegate_number](cfg_thres) is set to a value larger than zero (default=2), virga events spanning a lower number of range gates between gaps of false values are dropped from the 
virga mask (see {ref}`this figure <fig-sketch>`)

(mvel)=
### Doppler velocity based refinement
If the Doppler velocity is provided in the input data  [mask_vel](cfg_flag) and [mask_clutter](cfg_flag) can be enabled. While [mask_vel](cfg_flag) can be used to restrict virga to only falling 
droplets (with default [vel_thres](cfg_thres)=0), [mask_clutter](cfg_flag) restricts virga to datapoints fulfilling:
```
vel > -clutter_m * (Ze / 60 dBz) + clutter_c
```
where *vel* and *Ze* denotes the input Doppler velocity [ms-1] and radar reflectivity [dBz], respectively.
For convenience, the reflectivity is scaled by 60 dBz (as -60dBz is minimum valid value of *LIMRAD94*). The values [clutter_m](cfg_thres) [ms-1] and [clutter_c](cfg_thres) [ms-1] are slope and 
intercept of the threshold line. A data point is considered virga, only if the above equation is fulfilled.With default configuration ([clutter_m](cfg_thres) = 4 and [clutter_c](cfg_thres) =  -8) 
unusual combinations of low reflectivity while high falling speed hydro meteors are dropped (see {ref}`figure below <fig-ze-vs-vel>`). 
```{figure} ../docs/images/vs_demonstration_ze_vs_vel.png
:name: fig-ze-vs-vel
2D-Histrogram of radar reflectivity and Doppler velocity of LIMRAD94 on RV-Meteor at EUREC4A campaign at 2020-01-17. Red shaded areas show, when virga is not considered due to **mask_vel** and 
**mask_clutter** using the default configuration of **vel_thres=0**, **clutter_m=4** and **clutter_c=-8**.
```
Therefore, [mask_vel](cfg_flag) and [mask_clutter](cfg_flag) can be used to explicitly mask a certain interval of interest of Doppler velocity or radar reflectivity for virga statistics.

## Output data
The results of the virga and cloud detection are stored as boolean flags on same dimensions of the radar reflectivity input data in the output dataset. In addition, the processed cloud-base and 
-top heights are stored, as well as some basic characteristics as cloud and virga depth for each column.  
For a detailed descriptions see [Output dataset](output). The output dataset has an implemented custom methods for plotting of the Virga-Sniffer output (*vsplot*). For a detailed description see 
[Plotting](plotting).
