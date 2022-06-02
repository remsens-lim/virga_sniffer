# Setup


## Configuration
The Virga-Sniffer utilizes a variety of flags and thresholds to detect virga from the given input data.

### Flags
Flags are boolean values which control certain functionality of the Virga-Sniffer:
 - **require_cbh**: If **True**, detected virga always have to be attributed to a cloud-base height (CBH) value. This prevents False-positive detection of virga which actually would be a cloud in case of data-gaps in the ceilometer CBH data. If **False**, this requirement is dropped and virga events, where the cloud is already gone, can be detected. It is highly recommended to use **True** when working with multi-layer CBH, as gaps in ceilometer data are introduced by clouds in the lower layers. To circumvent the issue of ceilometer data gaps, one can adjust CBH-layer fill (*cbh_layer_fill_...*) thresholds. The default is **True**.
 - 

### Thresholds