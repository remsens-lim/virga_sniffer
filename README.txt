Timeseries with virga properties at each timestep of the ceilometer
Column description:
unix_time: seconds since 1970-01-01 00:00:00 UTC
virga_flag: was there a virga in this time step? Only considers one timestep!
ceilo_cbh: ceilometer first detected cloud base height in meters
lowest_virga_range_gate: lowest virga range gate in meters extracted from the radar range gates (different range resolution than ceilometer)
virga_depth: difference between lowest and highest range gate in virga, using the radar range. The closest radar range gate to the ceilometer first cloud base height is used as the upper range gate.
cloud_depth: Difference between highest radar range gate detected as the cloud top and the ceilometer base height.
max_ze: maximum radar reflectivity in linear units of the virga
max_dbz: maximum radar reflectivtiy in dBZ of the virga
max_vel: maximum Doppler velocity in m/s of the virga
mean_vel: mean Doppler velocity in m/s of the virga
median_vel: median Doppler velocity in m/s of the virga
std_vel: standard deviation of the Doppler velocity of the virga
ze_gradient_mean: mean gradient of reflectivity in the profile
ze_gradient_median: median gradient of reflectivity in the profile
ze_gradient_std: standard deviation of ze differences between the consecutive range gates in linear units starting at the top.
dbz_gradient_mean: mean gradient of reflectivity in dBZ in profile
dbz_gradient_median: median gradient of reflectivity in dBZ in profile
dbz_gradient_std: standard deviation of differences between the consecutive range gates in dBZ units starting at the top. This formula is used to account for the transformation to the logarithmic space: 10*log10(1 + std(differences) / np.mean(differences))
vel_gradient_mean: mean velocity gradient in profile in m/s
vel_gradient_median: median velocity gradient in profile in m/s
vel_gradient_std: standard deviation of vel differences between the consecutive range gates in linear units starting at the top.
first_gate: flag whether the first virga range gate corresponds to the first radar range gate -> virga could be longer

Things noticed in first processing:
- cloud depth is not always bigger than virga depth, especially for small virga which are only present for one or two time steps
- cloud depth can also be negative which is due to the different range resolutions of the instruments and the fact that the closest cloud tops and bottoms to the ceilometer time steps are selected

Changes in Version 2:
- Do not read in first radar range gate as it contains unfiltered clutter.  
- Use heave corrected radar data.  
-> This introduced a problem with the velocity data. Since the final step in the heave correction is to apply a rolling mean over the Doppler velocity, not every Ze value has a corresponding Doppler velocity. Thus, the statistics on the velocity are sometimes all nan, even though there is reflectivity observed.  
- The masked values (no signal = -999) in the reflectivity and velocity data are replaced by 0 to minimize their influence on the interpolation on the ceilometer time.  
- Instead of interpolating the cloud_bases_tops variable the function is applied again after the interpolation.  
- The function find_bases_tops had a bug which prevented it from detecting a cloud base for a cloud which started in the first range gate. This has been fixed. This was probably the cause for some weird statistics as in such cases the last (upper most) range gate was classified as cloud top and no cloud base was classified.  
- Add more profile statistics for reflectivity (dBZ and linear), velocity and the gradient of both.  
- Increase minimum virga time to 80 timesteps (~4 minutes for first chirp table/~2 minutes for later chirp table).  
