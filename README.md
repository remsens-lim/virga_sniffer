# Virga Sniffer
The Virga-Sniffer is a tool to detect *virga* (precipitation which completely evaporates before reaching the surface).
As input source **radar reflectivity** and **ceilometer cloud-base height** observation are mandatory. Optionally but highly recommended are the additional information of **radar doppler velocity**, **lifting condensation level** and **surface rain detection**.

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
- **v0.3.X**: Virga-Sniffer overhaul, generalized input data, multi-layer handling
- **v0.2.X**: improved base version, using heave corrected data
- **v0.1.X**: base version tailored to limrad94 Reflectivity, 1st ceilometer CBH, DWD rain sensor




