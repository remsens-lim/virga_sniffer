import sys
sys.path.append('../src/')
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import Label

from virga_sniffer import virga_mask as vm
from virga_sniffer import vsplot

# read default config
with open("../src/virga_sniffer/config_vsdefault.json") as json_file:
    default_config = json.load(json_file)

def vmask_interact(input_file,
                   smooth_window_cbh,
                   smooth_window_lcl,
                   require_cbh,
                   # mask_below_cbh,
                   mask_ze_threshold,
                   ze_threshold,
                   mask_rain,
                   mask_connect2cbh,
                   mask_minimum_rg,
                   ymax,mask_vel,
                   vel_threshold,
                   mask_clutter,
                   clutter_m,
                   clutter_c,
                   clean_threshold,
                   layer_threshold,
                   cbh_layer_fill,
                   ze_max_gap,
                   cbh_fill_method,
                   layer_fill_limit,
                   cbh_ident_function):
    plt.close('all')
    if layer_fill_limit >= 5*60:
        layer_fill_limit=None

    if len(cbh_ident_function)==0:
        cbh_ident_function=[]
    else:
        cbh_ident_function = np.array(list(cbh_ident_function.strip().split(','))).astype(int)

    if len(input_file) == 0:
        input_data = xr.open_dataset("../example/test_data/2020-01-24_00_virga-sniffer_input.nc")
    else:
        input_data = xr.open_dataset(input_file)

    config = dict(smooth_window_cbh=smooth_window_cbh,
                  smooth_window_lcl=smooth_window_lcl,
                  require_cbh=require_cbh,
                  # mask_below_cbh=mask_below_cbh,
                  mask_zet=mask_ze_threshold,
                  ze_thres=ze_threshold,
                  mask_rain=mask_rain,
                  mask_connect=mask_connect2cbh,
                  mask_minrg=mask_minimum_rg,
                  mask_vel=mask_vel,
                  vel_thres=vel_threshold,
                  mask_clutter=mask_clutter,
                  clutter_m=clutter_m,
                  clutter_c=clutter_c,
                  cbh_ident_function=cbh_ident_function,
                  clean_threshold=clean_threshold/100.,
                  layer_threshold=layer_threshold,
                  cbh_layer_fill=cbh_layer_fill,
                  virga_max_gap=ze_max_gap,
                  cbh_fill_method=cbh_fill_method,
                  layer_fill_limit=layer_fill_limit)
    vsout = vm(input_data, config=config)

    fig,axs = vsout.vsplot.quicklook_full(ylim=ymax)

    colors = [
        "k",
        '#e41a1c',
        '#ff7f00',
        '#ffff33',
        '#377eb8',
        '#a65628',
    ]
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    tim = pd.to_datetime(vsout.time.values)
    for ilayer in vsout.layer.values:
        ax.plot(tim, vsout.sel(layer=ilayer).virga_depth,
                color=colors[ilayer])
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    tim = pd.to_datetime(vsout.time.values)
    for ilayer in vsout.layer.values:
        ax.plot(tim, vsout.sel(layer=ilayer).virga_depth_maximum_extend,
                color=colors[ilayer])

    return 0

def vsinteractive():

    style = {'description_width': 'initial'}
    input_file = widgets.FileUpload(accept='.nc', multiple=False)
    input_file = widgets.Text(
        value="../example/test_data/2020-01-24_00_virga-sniffer_input.nc",
        placeholder='path to input file')
    ymax = widgets.IntSlider(min=1000, max=15000, step=1000, value=6000,
                             continuous_update=False)

    smooth_window_cbh = widgets.IntSlider(min=0, max=300, step=15,
                                          value=default_config['smooth_window_cbh'],
                                          continuous_update=False)
    smooth_window_lcl = widgets.IntSlider(min=0, max=600, step=60,
                                          value=default_config['smooth_window_lcl'],
                                          continuous_update=False)
    # mask_below_cbh = widgets.Checkbox(value=default_config['mask_below_cbh'], indent=False)
    require_cbh = widgets.Checkbox(value=default_config['require_cbh'], indent=False)
    mask_rain = widgets.Checkbox(value=default_config['mask_rain'], indent=False)
    mask_zet = widgets.Checkbox(value=default_config['mask_zet'], indent=False)
    ze_thres = widgets.IntSlider(min=-40, max=20, step=2,
                                 value=default_config['ze_thres'],
                                 continuous_update=False)
    mask_connect = widgets.Checkbox(value=default_config['mask_connect'], indent=False)
    mask_minrg = widgets.IntSlider(min=0, max=10, step=1,
                                   value=default_config['mask_minrg'],
                                   continuous_update=False)
    mask_vel = widgets.Checkbox(value=default_config['mask_vel'], indent=False)
    vel_thres = widgets.FloatSlider(min=-7, max=7, step=0.5,
                                    value=default_config['vel_thres'], intent=False,
                                    continuous_update=False)
    mask_clutter = widgets.Checkbox(value=default_config['mask_clutter'], indent=False)
    clutter_m = widgets.IntSlider(min=0, max=10, step=1, value=default_config['clutter_m'],
                                  continuous_update=False)
    clutter_c = widgets.IntSlider(min=-12, max=0, step=1, value=default_config['clutter_c'],
                                  continuous_update=False)

    layer_threshold = widgets.IntSlider(min=0, max=1000, step=100,
                                        value=default_config['layer_threshold'],
                                        continuous_update=False)
    ze_max_gap = widgets.IntSlider(min=0, max=500, step=50,
                                      value=default_config['ze_max_gap'],
                                      continuous_update=False)
    clean_threshold = widgets.IntSlider(min=0, max=100, step=1,
                                        value=int(100*default_config['clean_threshold']),
                                        intent=False,continuous_update=False)
    cbh_layer_fill = widgets.Checkbox(value=default_config['cbh_layer_fill'], indent=False)
    cbh_ident_function = widgets.Text(
        value=str(default_config['cbh_ident_function']).strip('[]'),
        placeholder='numbers in 0 - 4',
        description='',
        disabled=False
    )



    layer_fill_limit=widgets.IntSlider(min=0, max=5*60, step=1,
                                       value=default_config['layer_fill_limit'])
    cbh_fill_method = widgets.Dropdown(
        options=['ffill','bfill','nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial'],
        value=default_config['cbh_fill_method'],
        description='',
        disabled=False,
    )

    ui = widgets.VBox([
        Label("CBH Processing: -------------------------------------------------------"),
        widgets.HBox([Label("CBH smooth window [s]:"),smooth_window_cbh]),
        widgets.HBox([Label("CBH layer threshold [m]:"),layer_threshold]),
        widgets.HBox([Label("CBH layer clean threshold [%]:"),clean_threshold]),
        widgets.HBox([widgets.VBox([Label("CBH Layer processing:"),Label("0-clean, 1-split, 2-merge, 3-LCL, 4-smooth (default  == 1,0,2,0,3,1,0,2,0,3,4)")]),
                      cbh_ident_function]),
        widgets.HBox([Label("Layer max filling [min]:"),cbh_fill_method,layer_fill_limit,cbh_layer_fill]),
        # widgets.HBox([Label("Merge LCL:"),merge_LCL]),
        widgets.HBox([Label("LCL smooth window [s]:"),smooth_window_lcl]),
        Label("Virga Mask: -----------------------------------------------------------"),
        # widgets.HBox([Label("Mask below CBH:"),mask_below_cbh]),#Label('Restrict Virga below CBH layer, 0=off')
        widgets.HBox([Label("Require CBH          :"),require_cbh]),#Label('Detection of Virga requires CBH value in column')
        widgets.HBox([Label("Rain Mask (Ze) [dBz] :"),  ze_thres,mask_zet]),#Label("No Virga if Ze of lowest range-gate is above threshold [dBz]")
        widgets.HBox([Label("Rain Mask (DWD)      :"),mask_rain]),#, Label("If DWD detects Rain => no virga")
        widgets.HBox([Label("Doppler Velocity Mask [ms-1]:"),vel_thres,mask_vel]),#Label("Virga, if Dopplervelocity below threshold [m s-1]"
        widgets.HBox([Label("Mask Virga gaps (based on ze max gap)"), mask_connect]),#Label("Requires this number of range-gates connected to CBH to be counted as virga."
        widgets.HBox([Label("Ze max gap [m]:"), ze_max_gap]),
        widgets.HBox([Label("Number of RG required:"), mask_minrg]),#Label("Requires this number of virga range-gates to be counted as virga."
        widgets.HBox([Label("Masking Ze+Vel relation:"),clutter_m,clutter_c, mask_clutter]),
        Label("Data chooser: ----------------------------------------------------------"),
        widgets.HBox([Label("Choose alternative Input-file:"),input_file]),
        widgets.HBox([Label("Max Altitude [m]     :"),ymax]),
        ])

    out = widgets.interactive_output(vmask_interact,
                                     dict(input_file=input_file,
                                          smooth_window_cbh=smooth_window_cbh,
                                          smooth_window_lcl=smooth_window_lcl,
                                          require_cbh=require_cbh,
                                          # mask_below_cbh=mask_below_cbh,
                                          mask_ze_threshold=mask_zet,
                                          ze_threshold=ze_thres,
                                          mask_rain=mask_rain,
                                          mask_connect2cbh=mask_connect,
                                          mask_minimum_rg=mask_minrg,
                                          ymax=ymax,
                                          mask_vel=mask_vel,
                                          vel_threshold=vel_thres,
                                          mask_clutter=mask_clutter,
                                          clutter_m=clutter_m,
                                          clutter_c=clutter_c,
                                          cbh_layer_fill=cbh_layer_fill,
                                          ze_max_gap=ze_max_gap,
                                          cbh_fill_method=cbh_fill_method,
                                          cbh_ident_function=cbh_ident_function,
                                          clean_threshold=clean_threshold,
                                          layer_threshold=layer_threshold,
                                          layer_fill_limit=layer_fill_limit))
    return ui, out

