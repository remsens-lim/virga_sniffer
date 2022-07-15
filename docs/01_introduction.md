# Introduction
The Virga-Sniffer is a tool to detect *virga* (precipitation which completely evaporates before reaching the surface).
As input source **radar reflectivity** and **ceilometer cloud-base height** observation are mandatory. Optionally but highly recommended are the additional information of **radar doppler velocity**, **lifting condensation level** and **surface rain detection**.

# Method description

```{graphviz} ../docs/flowchart.dot
```

## Cloud-base preprocessing
The input cloud-base height (CBH) layer data is preprocessed before it is used for virga detection. Prior to the configurable processing the CBH data is smoothed, which corresponds to processing 
module 4. The remaining processing steps are controlled with the configuration setting of **cbh_processing**, which is a list of integer in the range of 0 to 4:
 - **0: clean and sort**: First each layer of the current CBH dataset with less than **cbh_clean_thres** [%] number valid (not nan) datapoints are dropped (*clean*). From the remaining 
   layers the mean height of each layer is calculated. The CBH dataset is then re-indexed, sorting the layers in ascending order by mean height (*sort*).
 - **1: split**: The CBH dataset is iterated successively layer by layer. For each layer, outliers according to layer mean +- **cbh_layer_thres** are identified and pushed to new created 
   layers above/below the current layer. This process is re-iterated until no new layers are created. 
 - **2: merge**: Merging CBH layer data by successively iterating all layers and comparing lower layers to all layers above them. If the distance of the compared layers is smaller than 
   **cbh_layer_thres**, upper layer data will be added to lower layer, or merged by mean value if both layers hold valid data.
 - **3: add lcl**: At the lowest CBH layer, nan-values are filled with lifting-condensation level data (*lcl*) from the input dataset.
 - **4: smooth**: The CBH layer data is smoothed by applying a running-median window of the size **cbh_smooth_window**.

After the all processing steps are applied, the processed CBH data is filled by interpolation according to the configuration **cbh_fill_limit** and **cbh_fill_method**. This step can be disabled 
   either setting **cbh_fill_limit** to 0 or **cbh_fill_method** to None.

## Virga & cloud detection
After the preprocessing of CBH, the radar reflectivity values, specifically the boolean mask of valid reflectivity values, is used for the initial step of detecting virga, cloud and cloud-top 
heights. This is done by successively iterating the reflectivity mask starting from each cloud-base up- and downwards. Virga is detected at each range-gate of valid radar reflectivity 
iterating downward until a gap (nan-value in radar 
reflectivity) occurs, which is larger than the configuration threshold **virga_max_gap**.
Similarly, valid radar reflectivity values from the cloud base upward are marked as clouds until a gap larger than the configuration threshold **ze_max_gap** occurs. The cloud top height value is 
assigned to the range gate value (top of range gate) of the last valid radar reflectivity value below the gap. 
```{note}
The detection of clouds is always limited to the area between cloud base and top, while 
virga cannot be detected in this range.
```
```{note}
A special case is when there are no gaps in radar reflectivity between the cloud base layers. In this case, the intervening cloud layers are omitted. Therefore, the virga events are connected and 
in this case assigned to the highest contiguous cloud base and associated cloud.
```
The detected cloud-top values are smoothed as cloud-base values are smoothed prior to the cloud-base processing utilizing a rolling median filter of window size defined by the 
**smooth_window_cbh** threshold.

After this processing step, an index mapping of CTH and CBH values to the upper edge of radar range-gate heights is conducted for further processing. This mapping used to separate the cloud- and 
virga-mask into cloud layer components. 

Until this point, the identification of clouds and virga is solely based on the input variables *cloud_base_height* and *Ze* (radar reflectivity). The virga mask is refined by optional masking 
componentes, enabled via the configuration. The modules *mask_rain_ze* and *minimum_rg_number* can be used without the requirement of additional data. If one  of *mask_clutter*, *mask_vel*, 
*mask_rain* is set in the configuration, additional data is required (see [Configuration](configuration))

```{note}
It is recommended to enable at least one of *mask_rain_ze* and *mask_rain*, else the Virga-Sniffer turns into Precipitation-Sniffer.  Using *mask_rain_ze*, virga detection can be done using only 
cloud-base height and radar reflectivity data.
```

## Optional virga masking

## Output

# Caveats
The virga-sniffer is a column based detection scheme for virga events. The detection is strongly tuned and manually evaluated for best performance of the dataset (TODO:cite) it relies on threshold 
based tests which might not work in other environments or different input data setup. Some major caveats to have an eye on while using the virga-sniffer are outlined below:

```{figure} ../docs/images/vs_demonstration_maxgap_multilayer.jpg
:alt: ze_max_gap to small
:name: fig-ze-max-gap

Virga sniffer output at a situation to reveal some caveats. At 03:45 UTC - fractured radar signal; at 05:00 UTC - cloud detection; at 05:45 UTC - multi layer cloud transition.
```

## Fractured radar reflectivity signal
Virga events are associated with a certain cloud-base height (if **require_cbh = True**). Precipitation below a cloud-base height layer, which do not reach the surface is considered virga. In case 
of fractured radar signal (see {ref}`demonstration figure <fig-ze-max-gap>` at around 03:45 UTC). The virga associated with the cloud-base will far below if the gaps are small enough 
(**ze_max_gap** threshold). This should be kept in mind when using output values of **virga_base** and **virga_top** as they mark the maximum extend of virga plus the gaps, which is 
**virga_depth_maximum_extend**. The output value **virga_depth** is calculated by excluding these gaps and therefore should be used when calculating volumetric characteristics. Anyway, this caveat 
can be circumvented by not ignoring gaps in virga setting the **ignore_virga_gaps** flag to **False**, although this would mean to cut some virga (especially in fall streaks). 

## Cloud - virga differentiation (in upper layer clouds)
If clouds are present in multi levels, virga-detection is challenging, as only the cloud-base is known a priori and the vertical extend of the precipitating cloud is not. The Virga-Sniffer 
includes a cloud and cloud-top detection which is heavily sensitive to the **ze_max_gap** threshold and by essence defines the cloud top where gaps in radar reflectivity occur. This raises two 
issues if upper layer clouds are present:
 1. **ze_max_gap** is to small: Due to uncertainties in observational cloud-base height or radar reflectivity data, misalignment of both data or coarse resolution of radar range-gates, 
    ceilometer detected cloud bases might not connect directly to a valid radar signal. Assume, the cloud-base height value is below the first range-gate with valid radar signal and the gap is 
    larger than **ze_max_gap**, the cloud will not be detected and no cloud-top will be assigned. In turn, these range-gates which are not marked as cloud due to that will be potentially marked as 
    virga from higher level clouds if **ignore_virga_gaps** is True (default) and the precipitation of the upper layer is close to the lower layer cloud. See the {ref}`demonstration figure <fig-ze-max-gap>` at 
    around 05:00 UTC. 
 2. **ze_max_gap** is to large: Similarly, if the gap allowance is to large, clouds will expand over the precipitation from upper layer clouds when close to lower layer cloud top height. 

## Multi-layer cloud transition
The data points of radar reflectivity might connect (without gaps or gaps smaller **ze_max_gap** threshold) through multiple layer of clouds defined by the ceilometer observed cloud base heights. 
This is the case for example in  the {ref}`demonstration figure <fig-ze-max-gap>` at around 05:00 and 05:45 UTC. During processing with the Virga-Sniffer, these cases assumed connected and lower 
layer cloud-base height values are dropped. But this might result in sudden *jump* of virga extend, if gaps in upper layers of cloud-base height occur. These gaps might result when the ceilometer 
beam is attenuated by the lower level cloud to a large extend. Gaps in ceilometer data can be filled (**cbh_layer_fill = True**) by increasing the **layer_fill_limit** threshold.


