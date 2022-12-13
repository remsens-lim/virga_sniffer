[![DOI](https://zenodo.org/badge/468340916.svg)](https://zenodo.org/badge/latestdoi/468340916)
[![Documentation Status](https://readthedocs.org/projects/virga-sniffer/badge/?version=latest)](https://virga-sniffer.readthedocs.io/en/latest/?badge=latest)


# Virga-Sniffer
The Virga-Sniffer is a tool to detect *virga* (precipitation which completely evaporates before reaching the surface).
As input source **radar reflectivity** and **ceilometer cloud-base height** observation are mandatory.
Optionally but highly recommended are the additional information of **mean Doppler velocity**, **lifting condensation level** and **surface rain detection**.

# Documentation
The documentation is hosted at [readthedocs](https://virga-sniffer.readthedocs.io/en/latest/index.html).

# Demonstration
The example/virga-sniffer_config_demo.ipynb can be executet in 
[binder](https://mybinder.org/v2/gh/remsens-lim/virga_sniffer/main?filepath=example/virga-sniffer_config_demo.ipynb).

# Installation
The package can be installed via pip:
```
 python -m pip install git+https://github.com/remsens-lim/virga_sniffer.git#egg=virga_sniffer
```

To build the documentation, additional dependencies can be installed via:
```
 python -m pip install git+https://github.com/remsens-lim/virga_sniffer.git#egg=virga_sniffer[docs]
```

To run the config demonstration in the examples directory, additional dependencies can be installed via:
```
 python -m pip install git+https://github.com/remsens-lim/virga_sniffer.git#egg=virga_sniffer[example]
```

# Versions
Documentation of version changes can be found in [ChangeLog.md](ChangeLog.md).
- **v1.0.X**: Additional user configuration, optimized initial detection. Version referenced in https://doi.org/10.5194/amt-2022-252
- **v0.3.X**: Virga-Sniffer overhaul, generalized input data, multi-layer handling
- **v0.2.X**: improved base version, using heave corrected data
- **v0.1.X**: base version tailored to limrad94 Reflectivity, 1st ceilometer CBH, DWD rain sensor




