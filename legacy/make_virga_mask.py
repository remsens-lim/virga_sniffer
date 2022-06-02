#!/usr/bin/env python
"""script to analyse virgae during eurec4a
Idea: compare height of first radar echo and cloud base height detected by ceilometer
Step 1: find virga in each time step
Step 2: create virga mask
Step 3: detect single virga and define borders
Step 4: evaluate statistcally -> output like cloud sniffer in csv file
Step 5: plot radar Ze with virga in boxes (optionally)
Result:
    - depth of virga
    - maximum Ze in virga
    - dataset
"""

# %% import modules and set up logging
import sys
sys.path.append("/projekt1/remsens/work/jroettenbacher/Base/larda")
import pyLARDA
import pyLARDA.helpers as h
import functions_jr as jr
import datetime as dt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.patches import Polygon, Patch
import pandas as pd
import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# %% gather command line arguments and set options
method_name, args, kwargs = h._method_info_from_argv(sys.argv)

version = 2
save_fig = True  # plot the two virga masks? saves to ./tmp/
save_csv = True
plot_data = True  # plot radar Ze together with virga polygons
csv_outpath = '/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/virga_sniffer'
csv_outpath2 = '/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/virga_sniffer/timeseries'
larda = pyLARDA.LARDA().connect("eurec4a")

if 'date' in kwargs:
    date = str(kwargs['date'])
    begin_dt = dt.datetime.strptime(date, "%Y%m%d")
else:
    begin_dt = dt.datetime(2020, 2, 15, 0, 0, 0)

end_dt = begin_dt + dt.timedelta(hours=23, minutes=59, seconds=59)
time_interval = [begin_dt, end_dt]

# %% read in data
system = "LIMRAD94_cni_hc_ca"
# system = "LIMRAD94_cn_input"
# leave out lowest range gate (315m) because of clutter
radar_ze = larda.read(system, "Ze", time_interval, [330, 'max'])  # already in dBZ
radar_vel = larda.read(system, "Vel", time_interval, [330, 'max'])
# turn -999 (masked values) to 0 to minimize their impact on the interpolation
# the interpolated values will be masked again
radar_ze["var"][radar_ze["mask"]] = 0
radar_vel["var"][radar_vel["mask"]] = 0

ceilo_cbh = larda.read("CEILO", "cbh", time_interval)
rainrate = jr.read_rainrate()  # read in rain rate from RV-Meteor DWD rain sensor
rainrate = rainrate[time_interval[0]:time_interval[1]]  # sort index and select time interval

# %% make a rain flag, extend rain flag x minutes after last rain to account for wet radome
rain_flag_dwd = rainrate.Dauer > 0  # set rain flag if rain duration is greater 0 seconds
# get a one dimensional array with the indices where rainflag turns from True to False or vice versa
indices = np.asarray(np.where(np.diff(rain_flag_dwd))).flatten()
# get indices where rainflag turns from True to False only -> index where it stops to rain
rain_indices = np.asarray([idx for idx in indices if rain_flag_dwd[idx]])
# from the end of each rain event add 10 minutes of masked values
minutes = 10  # just for readability
for i in rain_indices:
    rain_flag_dwd[i:(i+minutes)] = True

# %% interpolate radar and rain rate on ceilo time
radar_ze_ip = pyLARDA.Transformations.interpolate2d(radar_ze, new_time=ceilo_cbh['ts'])
radar_ze_ip['mask'] = radar_ze_ip['mask'] == 1  # turn mask from integer to bool
radar_ze_ip['var'][radar_ze_ip['mask']] = np.nan  # mask values again
radar_vel_ip = pyLARDA.Transformations.interpolate2d(radar_vel, new_time=ceilo_cbh['ts'])
radar_vel_ip['mask'] = radar_vel_ip['mask'] == 1  # turn mask from integer to bool
radar_vel_ip['var'][radar_vel_ip['mask']] = np.nan  # mask values again
# interpolating (select_closest) the cloud mask does not work -> create new cloud mask from interpolated radar data
_, radar_cloud_mask_ip = jr.find_bases_tops(radar_ze_ip["mask"], radar_ze_ip["rg"])

f_rr = interp1d(h.dt_to_ts(rain_flag_dwd.index), rain_flag_dwd, kind='nearest', fill_value="extrapolate")
rain_flag_dwd_ip = f_rr(ceilo_cbh['ts'])  # interpolate DWD RR to ceilo time values
rain_flag_dwd_ip = rain_flag_dwd_ip == 1  # turn mask from integer to bool

# %% get height of first ceilo cloud base and radar echo
h_ceilo = ceilo_cbh['var'][:, 0]
# get arrays which have the indices at which a signal is measured in a timestep, each array corresponds to one timestep
rg_radar_all = [np.asarray(~radar_ze_ip['mask'][t, :]).nonzero()[0] for t in range(radar_ze_ip['ts'].shape[0])]

# loop through arrays and select first element which corresponds to the first range gate with a signal
# convert the range gate index into its corresponding height
# if the time stamp has no signal an empty array is returned, append a -1 for those steps to keep size of time dimension
# check for the minimum vertical extent to already exclude such cases from the mask
min_vert_ext = 3  # minimum vertical extent: 3 radar range gates (70 - 120m) depending on chirp
h_radar, first_radar_ze, max_ze, max_vel, depth = list(), list(), list(), list(), list()
for i in range(len(rg_radar_all)):
    try:
        rg = rg_radar_all[i][0]
        if len(rg_radar_all[i]) >= min_vert_ext:
            # TODO: add a height threshold -> 1000 m to exclude low level liquid clouds
            h_radar.append(radar_ze_ip['rg'][rg])
            first_radar_ze.append(radar_ze_ip['var'][i, rg])  # save reflectivity at lowest radar range gate
        else:
            h_radar.append(-1)
            first_radar_ze.append(np.nan)
    except IndexError:
        h_radar.append(-1)
        first_radar_ze.append(np.nan)

########################################################################################################################
# %% Step 1: Is there a virga in the time step
########################################################################################################################
h_radar, first_radar_ze = np.asarray(h_radar), np.asarray(first_radar_ze)  # convert list to numpy array
cloudy = h_radar != -1  # does the radar see a cloud?
# since both instruments have different range resolutions compare their heights and decide if their are equal within a
# tolerance of 23m (approximate range resolution of first radar chirp)
h_diff = ~np.isclose(h_ceilo, h_radar, atol=23)  # is the ceilometer cloud base different from the first radar echo height?
virga_flag = h_ceilo > h_radar  # is the ceilometer cloud base higher than the first radar echo?
ze_threshold = first_radar_ze < 0  # is the reflectivity in the first radar range gate below 0 dBZ?
# combine all masks
# is a virga present in the time step?, exclude rainy profiles
virga_flag = cloudy & h_diff & virga_flag & ~rain_flag_dwd_ip & ze_threshold

########################################################################################################################
# %% Step 2: Create Virga Mask
########################################################################################################################
# virga mask on ceilo resolution
# if timestep has virga, mask all radar range gates between first radar echo and cbh from ceilo as virga
# find equivalent range gate to ceilo cbh
virga_mask = np.zeros(radar_ze_ip['var'].shape, dtype=bool)
# initialize arrays for time series csv output
# array for the standard deviation of the Ze gradient inside the virga
ze_gradient_std = np.full(ceilo_cbh['ts'].shape, np.nan)
ze_gradient_mean = np.full(ceilo_cbh['ts'].shape, np.nan)
ze_gradient_median = np.full(ceilo_cbh['ts'].shape, np.nan)
dbz_gradient_std = np.full(ceilo_cbh['ts'].shape, np.nan)
dbz_gradient_mean = np.full(ceilo_cbh['ts'].shape, np.nan)
dbz_gradient_median = np.full(ceilo_cbh['ts'].shape, np.nan)
vel_gradient_std = np.full(ceilo_cbh['ts'].shape, np.nan)  # same as above but for the velocity gradient
vel_gradient_mean = np.full(ceilo_cbh['ts'].shape, np.nan)
vel_gradient_median = np.full(ceilo_cbh['ts'].shape, np.nan)
max_ze = np.full(ceilo_cbh['ts'].shape, np.nan)
max_vel = np.full(ceilo_cbh['ts'].shape, np.nan)
mean_vel = np.full(ceilo_cbh['ts'].shape, np.nan)
median_vel = np.full(ceilo_cbh['ts'].shape, np.nan)
std_vel = np.full(ceilo_cbh['ts'].shape, np.nan)
virga_depth = np.full(ceilo_cbh['ts'].shape, np.nan)
cloud_depth = np.full(ceilo_cbh['ts'].shape, np.nan)
lowest_rg = np.full(ceilo_cbh['ts'].shape, np.nan)
first_gate = np.full(ceilo_cbh['ts'].shape, False)

for i in np.where(virga_flag)[0]:
    lower_rg = rg_radar_all[i][0]
    upper_rg = h.argnearest(radar_ze_ip['rg'], h_ceilo[i])
    # double check for minimal vertical extent
    if (upper_rg - lower_rg) >= min_vert_ext:
        virga_mask[i, lower_rg:upper_rg] = True
        ze_slice = radar_ze_ip['var'][i, lower_rg:upper_rg]  # get radar ze slice
        vel_slice = radar_vel_ip['var'][i, lower_rg:upper_rg]  # get radar vel slice

        ze_gradient = np.diff(h.z2lin(ze_slice[~np.isnan(ze_slice)]))  # gradient of non nan values
        dbz_gradient = np.diff(ze_slice[~np.isnan(ze_slice)])
        vel_gradient = np.diff(vel_slice[~np.isnan(vel_slice)])

        max_ze[i] = np.max(ze_slice)  # save maximum reflectivity in virga slice
        ze_gradient_mean[i] = np.mean(ze_gradient)
        ze_gradient_median[i] = np.median(ze_gradient)
        ze_gradient_std[i] = np.std(ze_gradient)  # take standard deviation of diff non nan values
        dbz_gradient_mean[i] = np.mean(dbz_gradient)
        dbz_gradient_median[i] = np.median(dbz_gradient)
        dbz_gradient_std[i] = h.lin2z(1 + np.std(ze_gradient) / np.mean(ze_gradient))
        # the heave correction processing applies a running mean to the Doppler velocity, thus the borders get filtered
        # therefore not every Ze value corresponds with a velocity value
        # this will raise a RunTimeWarning here because nan is returned from all functions
        max_vel[i] = np.max(vel_slice)  # save max Doppler velocity in virga slice
        mean_vel[i] = np.mean(vel_slice)
        median_vel[i] = np.median(vel_slice)
        std_vel[i] = np.std(vel_slice)
        vel_gradient_mean[i] = np.mean(vel_gradient)
        vel_gradient_median[i] = np.median(vel_gradient)
        vel_gradient_std[i] = np.std(vel_gradient)  # take standard deviation of diff non nan values


        virga_depth[i] = radar_ze_ip['rg'][upper_rg] - radar_ze_ip['rg'][lower_rg]
        # get the range index of the lowest cloud top
        # TODO: Double check what is calculated here -> many negative values
        # TODO: minimum time for virga -> 40 time steps
        # TODO: Remove first radar range gate due to unremoved noise there
        # TODO: rolling mean of cbh to remove jumps
        # TODO: use only virga which is filled with 50% data
        cloud_top_idx = np.min(np.where(radar_cloud_mask_ip[i, :] == 1)[0])
        cloud_depth[i] = radar_ze_ip['rg'][cloud_top_idx] - ceilo_cbh['var'][i, 0]
        lowest_rg[i] = radar_ze_ip['rg'][lower_rg]
        first_gate[i] = lower_rg == 0  # check if lowest virga range gate corresponds to first radar range gate
    else:
        continue

# %% make a larda container with the mask
virga = h.put_in_container(virga_mask, radar_ze_ip, name="virga_mask", paramkey="virga", var_unit="-", var_lims=[0, 1])
location = virga['paraminfo']['location']
if save_fig:
    fig, ax = pyLARDA.Transformations.plot_timeheight2(virga, range_interval=[0, 3000],
                                                       time_interval=[time_interval[0],
                                                                      time_interval[0]+dt.timedelta(hours=3)])
    virga_dt = [h.ts_to_dt(t) for t in virga['ts']]
    ax.scatter(virga_dt, h_ceilo, s=0.2, color="purple")
    figname = f"./tmp/{location}_virga_ceilo-res_{time_interval[0]:%Y%m%d}_v{version}.png"
    fig.savefig(figname)
    plt.close()
    log.info(f"Saved {figname}")

# %% virga mask on radar resolution

ts_list = list()
for t in ceilo_cbh['ts']:
    id_diff_min = h.argnearest(radar_ze['ts'], t)  # find index of nearest radar time step to ceilo time step
    ts_list.append(id_diff_min)  # append index to list

virga_mask_hr = np.zeros_like(radar_ze['mask'])
for j in range(len(ts_list)-1):
    ts1 = ts_list[j]
    ts2 = ts_list[j+1]
    if any(virga_mask[j]) and any(virga_mask[j+1]):
        rg1 = np.where(virga_mask[j])[0][0]  # select first masked range gate
        rg2 = np.where(virga_mask[j+1])[0][-1]  # select last masked range gate
        virga_mask_hr[ts1:ts2, rg1:rg2] = True  # interpolate mask to radar time resolution

virga_hr = h.put_in_container(virga_mask_hr, radar_ze, name="virga_mask", paramkey="virga", var_unit="-",
                              var_lims=[0, 1])
if save_fig:
    fig, ax = pyLARDA.Transformations.plot_timeheight2(virga_hr, range_interval=[0, 3000],
                                                       time_interval=[time_interval[0],
                                                                      time_interval[0]+dt.timedelta(hours=3)])
    virga_dt = [h.ts_to_dt(t) for t in virga['ts']]
    ax.scatter(virga_dt, h_ceilo, s=0.2, color="purple")
    figname = f"./tmp/{location}_virga_radar-res_{time_interval[0]:%Y%m%d}_v{version}.png"
    fig.savefig(figname)
    plt.close()
    log.info(f"Saved {figname}")

########################################################################################################################
# %% Step 3: define single virga borders (corners) and update virga mask
########################################################################################################################
min_hori_ext = 80  # virga needs to be present for at least 2 minutes (80*1.5s) to be counted
max_hori_gap = 10  # maximum horizontal gap: 10 radar time steps (40 - 30s) depending on chirp table
virgae = dict(ID=list(), start_unix=list(), end_unix=list(), start_date=list(), end_date=list(),
              virga_thickness_avg=list(), virga_thickness_med=list(), virga_thickness_std=list(),
              max_Ze=list(), min_Ze=list(), avg_height=list(), max_height=list(), min_height=list(),
              idx=list(), borders=list(), points_b=list(), points_t=list())
real_virgas_hr = np.zeros(virga_mask_hr.shape, dtype=bool)
t_idx = 0
while t_idx < len(virga_hr['ts']):
    # check if a virga was detected in this time step
    if any(virga_hr['var'][t_idx, :]):
        v, b, p_b, p_t = list(), list(), list(), list()
        # as long as a virga is detected within the maximum horizontal gap add the borders to v
        while virga_hr['var'][t_idx:(t_idx+max_hori_gap), :].any():
            h_ids = np.where(virga_hr['var'][t_idx, :])[0]
            if len(h_ids) > 0:
                # TODO: add check for maximum vertical gap -> there can be no values between cbh and first radar echo,
                # TODO: leading to a mask that covers lots of nan values. This introduces a bias in the thickness stat!
                if (h_ids[-1] - h_ids[0]) > min_vert_ext:
                    real_virgas_hr[t_idx, h_ids] = True  # set virga mask with only big virgas
                    v.append((t_idx, h_ids[0], h_ids[-1]))
                    b.append((virga_hr['ts'][t_idx], virga_hr['rg'][h_ids[0]], virga_hr['rg'][h_ids[-1]]))
                    p_b.append(np.array([date2num(h.ts_to_dt(virga_hr['ts'][t_idx])), virga_hr['rg'][h_ids[0]]]))
                    p_t.append(np.array([date2num(h.ts_to_dt(virga_hr['ts'][t_idx])), virga_hr['rg'][h_ids[-1]]]))
            t_idx += 1
        # when the virga is finished add the list of borders to the output list
        if len(v) > min_hori_ext:
            virgae['idx'].append(v)
            virgae['borders'].append(b)
            virgae['points_b'].append(p_b)
            virgae['points_t'].append(p_t)
    else:
        t_idx += 1

# %% put new virga mask in container
real_virgas = h.put_in_container(real_virgas_hr, radar_ze, name="virga_mask", paramkey="virga", var_unit="-",
                                 var_lims=[0, 1])

if save_fig:
    fig, ax = pyLARDA.Transformations.plot_timeheight2(real_virgas, range_interval=[0, 3000],
                                                       time_interval=[time_interval[0],
                                                                      time_interval[0]+dt.timedelta(hours=3)])
    virga_dt = [h.ts_to_dt(t) for t in virga['ts']]
    ax.scatter(virga_dt, h_ceilo, s=0.2, color="purple")
    figname = f"./tmp/{location}_real_virga_radar-res_{time_interval[0]:%Y%m%d}_v{version}.png"
    fig.savefig(figname)
    plt.close()
    log.info(f"Saved {figname}")
########################################################################################################################
# %% Step 4: get statistics of each virga and save to csv file
########################################################################################################################
# loop through virgas, select radar pixels, get stats
for v in virgae['idx']:
    time_slice = [v[0][0], v[-1][0]]
    # get only range borders
    rgs = [k[1:] for k in v]
    range_slice = [np.min(rgs), np.max(rgs)]
    virga_ze = pyLARDA.Transformations.slice_container(radar_ze, index={'time': time_slice, 'range': range_slice})
    mask = pyLARDA.Transformations.slice_container(real_virgas, index={'time': time_slice, 'range': range_slice})['var']
    # calculate thickness in each timestep
    thickness = list()
    for idx in range(len(v)):
        rg = radar_ze['rg']
        thickness.append(rg[v[idx][2]] - rg[v[idx][1]])
    # add stats do dictionary
    virgae["start_unix"].append(virga_ze["ts"][0])
    virgae["end_unix"].append(virga_ze["ts"][-1])
    virgae["start_date"].append(dt.datetime.utcfromtimestamp(virga_ze["ts"][0]).strftime("%Y-%m-%d %H:%M:%S"))
    virgae["end_date"].append(dt.datetime.utcfromtimestamp(virga_ze["ts"][-1]).strftime("%Y-%m-%d %H:%M:%S"))
    virgae['max_Ze'].append(h.lin2z(np.max(virga_ze['var'][mask])))
    virgae['min_Ze'].append(h.lin2z(np.min(virga_ze['var'][mask])))
    virgae['avg_height'].append(np.mean(virga_ze['rg']))
    virgae['max_height'].append(np.max(virga_ze['rg']))
    virgae['min_height'].append(np.min(virga_ze['rg']))
    virgae['virga_thickness_avg'].append(np.mean(thickness))
    virgae['virga_thickness_med'].append(np.median(thickness))
    virgae['virga_thickness_std'].append(np.std(thickness))
    virgae['ID'].append(dt.datetime.strftime(h.ts_to_dt(virga_ze['ts'][0]), "%Y%m%d%H%M%S%f"))

if save_csv:
    # write to csv file
    csv_out = pd.DataFrame(virgae)
    csv_name = f"{csv_outpath}/{location}_virga-collection_{time_interval[0]:%Y%m%d}_v{version}.csv"
    csv_out.to_csv(csv_name, sep=';', index=False)
    log.info(f"Saved {csv_name}")
    # write a timeseries csv file with attributes for each timestep (ceilometer resolution)
    ts_out = dict(unix_time=ceilo_cbh['ts'], virga_flag=virga_flag, ceilo_cbh=ceilo_cbh['var'][:, 0],
                  lowest_virga_range_gate=lowest_rg, virga_depth=virga_depth, cloud_depth=cloud_depth,
                  max_ze=h.z2lin(max_ze), max_dbz=max_ze,
                  max_vel=max_vel, mean_vel=mean_vel, median_vel=median_vel, std_vel=std_vel,
                  ze_gradient_mean=ze_gradient_mean, ze_gradient_median=ze_gradient_median, ze_gradient_std=ze_gradient_std,
                  dbz_gradient_mean=dbz_gradient_mean, dbz_gradient_median=dbz_gradient_median,
                  dbz_gradient_std=dbz_gradient_std,
                  vel_gradient_mean=vel_gradient_mean, vel_gradient_median=vel_gradient_median,
                  vel_gradient_std=vel_gradient_std,
                  first_gate=first_gate)
    ts_name = f"{csv_outpath}/time_series/RV-Meteor_virga-timeseries_{time_interval[0]:%Y%m%d}_v{version}.csv"
    pd.DataFrame(ts_out).to_csv(ts_name, sep=';', index=False)
    log.info(f"Saved {ts_name}")


########################################################################################################################
# %% Step 5: plot radar Ze with virga in boxes (optionally)
########################################################################################################################
if plot_data:
    radar_ze.update(var_unit="dBZ", var_lims=[-60, 20])
    title = f"{location} Cloud Radar Reflectivity with Virga Detection {time_interval[0]:%Y-%m-%d}"
    t = [begin_dt, begin_dt + dt.timedelta(hours=3), begin_dt + dt.timedelta(hours=6),
         begin_dt + dt.timedelta(hours=9), begin_dt + dt.timedelta(hours=12),
         begin_dt + dt.timedelta(hours=15), begin_dt + dt.timedelta(hours=18),
         begin_dt + dt.timedelta(hours=21), end_dt]
    for i in range(len(t)-1):
        fig, ax = pyLARDA.Transformations.plot_timeheight2(radar_ze, range_interval=[0, 3000],
                                                           time_interval=[t[i], t[i+1]],
                                                           rg_converter=False,
                                                           title=f"{title} {i+1}")
        # add cloud base height
        time_list = [h.ts_to_dt(ts) for ts in ceilo_cbh['ts']]
        ax.scatter(time_list, ceilo_cbh['var'][:, 0], s=0.2, color='k', label='first ceilometer cloud base')
        lgd1 = ax.legend(loc=1)
        lgd1.legendHandles[0]._sizes = [30]
        for points_b, points_t in zip(virgae['points_b'], virgae['points_t']):
            # append the top points to the bottom points in reverse order for drawing a polygon
            points = points_b + points_t[::-1]
            ax.add_patch(Polygon(points, closed=True, fc='pink', ec='black', alpha=0.5))
        # add legend for virga polygons
        virga_lgd = [Patch(facecolor='pink', edgecolor='black', label="Virga")]
        lgd2 = ax.legend(handles=virga_lgd, bbox_to_anchor=[1, -0.1], loc="lower left",  prop={'size': 12})
        ax.add_artist(lgd1)
        figname = f"{csv_outpath}/{location}_radar-Ze_virga-masked_{time_interval[0]:%Y%m%d}_{i+1}_v{version}.png"
        fig.savefig(figname)
        plt.close()
        log.info(f"Saved {figname}")

# # custom virga plot
# t1 = dt.datetime(2020, 2, 9, 15, 55, 0)
# t2 = dt.datetime(2020, 2, 9, 16, 15, 0)
# fig, ax = pyLARDA.Transformations.plot_timeheight2(radar_ze, range_interval=[0, 3000],
#                                                    time_interval=[t1, t2],
#                                                    rg_converter=False)
# # add cloud base height
# time_list = [h.ts_to_dt(ts) for ts in ceilo_cbh['ts']]
# ax.scatter(time_list, ceilo_cbh['var'][:, 0], s=3, color='k', label='first ceilometer cloud base')
# lgd1 = ax.legend(loc=1)
# lgd1.legendHandles[0]._sizes = [30]
# for points_b, points_t in zip(virgae['points_b'], virgae['points_t']):
#     # append the top points to the bottom points in reverse order for drawing a polygon
#     points = points_b + points_t[::-1]
#     ax.add_patch(Polygon(points, closed=True, fc='pink', ec='black', alpha=0.5))
# # add legend for virga polygons
# virga_lgd = [Patch(facecolor='pink', edgecolor='black', label="Virga")]
# lgd2 = ax.legend(handles=virga_lgd, bbox_to_anchor=[1, -0.1], loc="lower left", prop={'size': 12})
# ax.add_artist(lgd1)
# figname = f"../plots/{location}_radar-Ze_virga-masked_{time_interval[0]:%Y%m%d}_custom.png"
# fig.savefig(figname)
# plt.close()
# log.info(f"Saved {figname}")

# # plot rainrate and used rainflag
# rainrate["Flag"] = rain_flag_dwd
# rr_plot = rainrate["20200125 12":'20200125 15']
# fig, ax = plt.subplots(figsize=[10, 6])
# ax.plot(date2num(rr_plot.index), rr_plot.Flag, label="Flag")
# ax.plot(date2num(rr_plot.index), rr_plot.Dauer)
# plt.title("RV-Meteor DWD Rain Sensor - Rain Duration per Minute in Seconds")
# plt.ylabel("Rain Duration [s]")
# plt.xlabel("Time [UTC]")
# time_extend = rr_plot.index[-1] - rr_plot.index[0]
# ax = pyLARDA.Transformations._set_xticks_and_xlabels(ax, time_extend)
# ax.legend()
# plt.savefig(f"./tmp/{location}_DWD-rainrate_{time_interval[0]:%Y%m%d}.png")
# plt.close()
