# Virga Sniffer
Detect virgae from radar and ceilometer data

Use ceilometer and radar data to detect virga occurences in each ceilometer time step (timeseries csv output) and calculates statistics for each virga case (one row per virga csv output). It also produces 3 hourly quicklooks.

# EUREC4A RV-Meteor Virga Detection and Analyses

**Used Instruments and Variables**

* LIMRAD94, 94 GHz Doppler cloud radar $\rightarrow$ Radar Reflectivity
* CHM15k, ceilometer $\rightarrow$ cloud base heigth of first detected cloud base
* WS100-UMB, DWD rain sensor $\rightarrow$ rain flag

### Virga Detection

Good Morning everyone, I spent yesterday making a virga sniffer for the campaign by utilizing the ceilometer cloud base height and the radar reflectivity Ze. The first runnable script is done and I will open a new repository with it in our remsens-lim Github account at one point. The routine goes through 5 steps:

1. Find virga in each time step
2. create virga mask
3. detect single virga and define borders
4. evaluate statistcally -> output like cloud sniffer in csv file
5. plot radar Ze with virga in boxes (optionally)

A virga is thereby defined when the ceilometer cloudbase differs from the height of first radar range gate, which shows a signal.  Their difference has to be at least 23m, which is the range resolution in the first chirp.  Furthermore, I included a rainflag from the DWD sensor. If rain is detected in a time step, no virga can be present. Also 10 minutes after the last rain the data is flagged. As a given the radar has to detect a signal at all. 

In the second step I then create a mask with pixels defined as virga = 1 and 0 otherwise. The third step takes that mask and loops through it. A virga has 

- a minimum vertical extent of 3 range gates (70 - 120 m depending on chirp)
- a minimum horizontal extent of 20 time steps (60 - 80 s depending on chirptable)
- a maximum gap of 10 timesteps is allowed inside the virga (30 - 40 s depending on chirptable)

I then save the borders as a tuple (time, lower range gate, upper range gate) and in a few different formats for easier plotting.  Using these borders I select each virga and compute some statistics about it and save everything to a csv file, similar to the cloud sniffer files from Johannes Bühl.

The analysis runs on a daily basis. Daily csv files and plots can be found here: `/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/virga_sniffer` I included two nice examples. (I always plot half a day for visibility)

Further steps can (must) be:

- I just discovered an error, when I have rain free days, need to fix that (Done, was an error in reading in the DWD csv file)
- merge daily csv files and derive statistics for whole campaign such as how often do we have virga, average depth, max/min Ze, relation between number of clouds and number of virga
- adjust routine to be run for any campaign using different rainflags (HATPRO)
- virga detection in upper chirps with 2nd and 3rd cloud base

![RV-Meteor_radar-Ze_virga-masked_20200125_2_first_try](./images/RV-Meteor_radar-Ze_virga-masked_20200125_2_first_try.png)

### First Comments Heike

In all but one occasion I would call it a virga: Feb25, 12:45 UTC: LIMRAD94 Ze is near saturation so it seems strange that there should be a virga. maybe such erroneous virga classifications can be filtered by applyling a radar Ze-gradient filter: If Ze does not decrease from virga top to bottom, then it shouldn't be defined as virga. probably/maybe some vertical range gate smoothing in the Ze is needed for that. Other option: if there is Ze in first range gate and Ze(first_range_gate) > threshold (0 dbZ? 5dBZ?), then there is also no virga but some other reason why first echo of Ze is below first ceilometer CBH. 

### First Corrections

#### 1. Using a gradient filter

checked if Ze increased from bottom to top of virga, if not -> not a virga

![RV-Meteor_radar-Ze_virga-masked_20200125_2_gradient_filter](./images/RV-Meteor_radar-Ze_virga-masked_20200125_2_gradient_filter.png)

**Result:** Basically no virga would be left.

#### 2. Using a Ze threshold filter 

A virga should have a low reflectivity in general. Thus, the first Ze detected should be below $0$ dBZ.

![RV-Meteor_radar-Ze_virga-masked_20200125_2_ze-threshold_filter](./images/RV-Meteor_radar-Ze_virga-masked_20200125_2_ze-threshold_filter.png)

**Result:** Looks good. The erroneous virga is removed. But the others are kept.

### Additions for Sebastian Los

Sebastian Los (University of New Mexico) can make use of a few extra variables for each profile, thus an extra csv file is created.

### Meeting with Heike and Sebastian 23. November 2021

Virga Sniffer v2  
  
Implemented improvements after discussion with Heike and Sebastian on 23. November 2021.  
  
- Do not read in first radar range gate as it contains unfiltered clutter.  
- Use heave corrected radar data.  
-> This introduced a problem with the velocity data. Since the final step in the heave correction is to apply a rolling mean over the Doppler velocity, not every Ze value has a corresponding Doppler velocity. Thus, the statistics on the velocity are sometimes all nan, even though there is reflectivity observed.  
- The masked values (no signal = -999) in the reflectivity and velocity data are replaced by 0 to minimize their influence on the interpolation on the ceilometer time.  
- Instead of interpolating the cloud_bases_tops variable the function is applied again after the interpolation.  
- The function find_bases_tops had a bug which prevented it from detecting a cloud base for a cloud which started in the first range gate. This has been fixed. This was probably the cause for some weird statistics as in such cases the last (upper most) range gate was classified as cloud top and no cloud base was classified.  
- Using the new heave corrected radar data brought  a change in unit. Reflectivity is now read in as dBZ already and not in linear units. -> Everything is adjusted to that but the variable names stay the same.  
- Add more profile statistics for reflectivity (dBZ and linear), velocity and the gradient of both.  
- Change the time interval on the temporary quality check quicklooks.  
- Increase minimum virga time to 80 timesteps (~4 minutes for first chirp table/~2 minutes for later chirp table).  
- Add start and end date (unix, formatted) to virga statistics.  
- Add version to quicklooks.
- Included new profile statistics in csv file
