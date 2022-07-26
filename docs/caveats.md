# Caveats
The Virga-Sniffer is a column based detection scheme for virga events. The detection is strongly tuned and manually evaluated for best performance with the dataset (TODO:cite)
It relies on threshold based tests which might not work in other environments or with different input data setup. Some major caveats to have an eye on while using the Virga-Sniffer are outlined below:

```{figure} ../docs/images/vs_demonstration_maxgap_multilayer.jpg
:alt: ze_max_gap to small
:name: fig-ze-max-gap

Virga sniffer output at a situation to reveal some caveats. At 03:45 UTC - fractured radar signal; at 05:00 UTC - cloud detection; at 05:45 UTC - multi layer cloud transition.
```

## Fractured radar reflectivity signal
Virga events are associated with a certain cloud-base height (if **require_cbh = True**). Precipitation below a cloud-base height layer, which does not reach the surface is considered virga. In case 
of a fractured radar signal (see {ref}`demonstration figure <fig-ze-max-gap>` at around 03:45 UTC) the virga associated with the cloud-base will be far below the cloud if the gaps are small enough 
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
