# Setup


## Configuration
The Virga-Sniffer utilizes a variety of flags and thresholds to detect virga from the given input data.

### Flags
Flags are boolean values which control certain functionality of the Virga-Sniffer:
 - **require_cbh = True**: If **True**, detected virga always have to be attributed to a cloud-base height (CBH) value. This prevents False-positive detection of virga which actually would be a cloud in case of data-gaps in the ceilometer CBH data. If **False**, this requirement is dropped and virga events, where the cloud is already gone, can be detected. It is highly recommended to use **True** when working with multi-layer CBH, as gaps in ceilometer data are introduced by clouds in the lower layers. To circumvent the issue of ceilometer data gaps, one can adjust CBH-layer fill (*cbh_layer_fill_...*) thresholds. The default is **True**.
 - **mask_connect = True**: This flag is used to switch on the detection of connected range-gates  of valid values of radar reflectivity. In addition, this enables the detection of cloud tops.
 - **mask_rain**:
 - **mask_zet**:

 - **mask_vel**:
 - **mask_clutter**:
---
 - **cbh_layer_fill**

### Thresholds
 - 

## Demonstration
An interactive demonstration of the virga-sniffer configuration is available as jupyter notebook (check *examples* directory) and viewable here: [nbviewer](https://nbviewer.
org/github/jonas-witthuhn/virga_sniffer/example/virga-sniffer_config_demo.ipynb).


