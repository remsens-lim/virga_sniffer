
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
                   cbh_smooth_window,
                   lcl_smooth_window,
                   require_cbh,
                   mask_rain_ze,
                   ze_thres,
                   mask_rain,
                   minimum_rangegate_number,
                   ymax, mask_vel,
                   vel_thres,
                   mask_clutter,
                   clutter_m,
                   clutter_c,
                   cbh_clean_thres,
                   cbh_layer_thres,
                   ze_max_gap,
                   virga_max_gap,
                   cbh_fill_method,
                   cbh_fill_limit,
                   cbh_processing):
    plt.close('all')
    if cbh_fill_limit >= 5*60:
        cbh_fill_limit=None

    if len(cbh_processing) == 0:
        cbh_processing=[]
    else:
        cbh_processing = np.array(list(cbh_processing.strip().split(','))).astype(int)

    if len(input_file) == 0:
        input_data = xr.open_dataset("../example/test_data/2020-01-24_00_virga-sniffer_input.nc")
    else:
        input_data = xr.open_dataset(input_file)

    config = dict(cbh_smooth_window=cbh_smooth_window,
                  lcl_smooth_window=lcl_smooth_window,
                  require_cbh=require_cbh,
                  mask_rain_ze=mask_rain_ze,
                  ze_thres=ze_thres,
                  mask_rain=mask_rain,
                  minimum_rangegate_number=minimum_rangegate_number,
                  mask_vel=mask_vel,
                  vel_thres=vel_thres,
                  mask_clutter=mask_clutter,
                  clutter_m=clutter_m,
                  clutter_c=clutter_c,
                  cbh_processing=cbh_processing,
                  cbh_clean_thres=cbh_clean_thres/100.,
                  cbh_layer_thres=cbh_layer_thres,
                  ze_max_gap=ze_max_gap,
                  virga_max_gap=virga_max_gap,
                  cbh_fill_method=cbh_fill_method,
                  cbh_fill_limit=cbh_fill_limit)
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

    return 0

def vsinteractive():
    style = {'description_width': 'initial'}
    input_file = widgets.Text(
        value="../example/test_data/2020-01-24_00_virga-sniffer_input.nc",
        placeholder='path to input file')

    ymax = widgets.IntSlider(min=1000, max=15000, step=1000, value=6000,
                             continuous_update=False)

    cbh_smooth_window = widgets.IntSlider(min=0, max=300, step=15,
                                          value=default_config['cbh_smooth_window'],
                                          continuous_update=False)
    lcl_smooth_window = widgets.IntSlider(min=0, max=600, step=60,
                                          value=default_config['lcl_smooth_window'],
                                          continuous_update=False)
    require_cbh = widgets.Checkbox(value=default_config['require_cbh'], indent=False)
    mask_rain = widgets.Checkbox(value=default_config['mask_rain'], indent=False)
    mask_rain_ze = widgets.Checkbox(value=default_config['mask_rain_ze'], indent=False)
    ze_thres = widgets.IntSlider(min=-40, max=20, step=2,
                                 value=default_config['ze_thres'],
                                 continuous_update=False)
    minimum_rangegate_number = widgets.IntSlider(min=0, max=10, step=1,
                                   value=default_config['minimum_rangegate_number'],
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

    cbh_layer_thres = widgets.IntSlider(min=0, max=1000, step=100,
                                        value=default_config['cbh_layer_thres'],
                                        continuous_update=False)
    ze_max_gap = widgets.IntSlider(min=0, max=500, step=50,
                                      value=default_config['ze_max_gap'],
                                      continuous_update=False)
    virga_max_gap = widgets.IntSlider(min=0, max=2000, step=100,
                                      value=default_config['virga_max_gap'],
                                      continuous_update=False)
    cbh_clean_thres = widgets.IntSlider(min=0, max=100, step=1,
                                        value=int(100*default_config['cbh_clean_thres']),
                                        intent=False,continuous_update=False)
    cbh_processing = widgets.Text(
        value=str(default_config['cbh_processing']).strip('[]'),
        placeholder='numbers in 0 - 4',
        description='',
        disabled=False
    )




    cbh_fill_limit=widgets.IntSlider(min=0, max=5*60, step=1,
                                       value=default_config['cbh_fill_limit'])
    cbh_fill_method = widgets.Dropdown(
        options=['ffill', 'bfill', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial'],
        value=default_config['cbh_fill_method'],
        description='',
        disabled=False,
    )

    ui = widgets.VBox([
        Label("CBH Processing: -------------------------------------------------------"),
        widgets.HBox([Label("CBH smooth window [s]:"),cbh_smooth_window]),
        widgets.HBox([Label("CBH layer threshold [m]:"),cbh_layer_thres]),
        widgets.HBox([Label("CBH layer clean threshold [%]:"),cbh_clean_thres]),
        widgets.HBox([widgets.VBox([Label("CBH Layer processing:"),Label("0-clean, 1-split, 2-merge, 3-LCL, 4-smooth (default  == 1,0,2,0,3,1,0,2,0,3,4)")]),
                      cbh_processing]),
        widgets.HBox([Label("Layer max filling [min]:"),cbh_fill_method,cbh_fill_limit]),
        widgets.HBox([Label("LCL smooth window [s]:"),lcl_smooth_window]),
        Label("Virga Mask: -----------------------------------------------------------"),
        widgets.HBox([Label("Require CBH          :"), require_cbh]),#Label('Detection of Virga requires CBH value in column')
        widgets.HBox([Label("Rain Mask (Ze) [dBz] :"),  ze_thres,mask_rain_ze]),#Label("No Virga if Ze of lowest range-gate is above threshold [dBz]")
        widgets.HBox([Label("Rain Mask (DWD)      :"), mask_rain]),#, Label("If DWD detects Rain => no virga")
        widgets.HBox([Label("Doppler Velocity Mask [ms-1]:"),vel_thres,mask_vel]),#Label("Virga, if Dopplervelocity below threshold [m s-1]"
        widgets.HBox([Label("Virga max gap [m]"), virga_max_gap]),#Label("Requires this number of range-gates connected to CBH to be counted as virga."
        widgets.HBox([Label("Ze max gap [m]:"), ze_max_gap]),
        widgets.HBox([Label("Number of RG required:"), minimum_rangegate_number]),#Label("Requires this number of virga range-gates to be counted as virga."
        widgets.HBox([Label("Masking Ze+Vel relation:"), clutter_m, clutter_c, mask_clutter]),
        Label("Data chooser: ----------------------------------------------------------"),
        widgets.HBox([Label("Choose alternative Input-file:"), input_file]),
        widgets.HBox([Label("Max Altitude [m]     :"), ymax]),
        ])

    out = widgets.interactive_output(vmask_interact,
                                     dict(input_file=input_file,
                                          cbh_smooth_window=cbh_smooth_window,
                                          lcl_smooth_window=lcl_smooth_window,
                                          require_cbh=require_cbh,
                                          mask_rain_ze=mask_rain_ze,
                                          ze_thres=ze_thres,
                                          mask_rain=mask_rain,
                                          minimum_rangegate_number=minimum_rangegate_number,
                                          ymax=ymax,
                                          mask_vel=mask_vel,
                                          vel_thres=vel_thres,
                                          mask_clutter=mask_clutter,
                                          clutter_m=clutter_m,
                                          clutter_c=clutter_c,
                                          ze_max_gap=ze_max_gap,
                                          virga_max_gap=virga_max_gap,
                                          cbh_fill_method=cbh_fill_method,
                                          cbh_processing=cbh_processing,
                                          cbh_clean_thres=cbh_clean_thres,
                                          cbh_layer_thres=cbh_layer_thres,
                                          cbh_fill_limit=cbh_fill_limit))
    return ui, out

