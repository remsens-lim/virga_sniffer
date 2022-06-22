"""
vsplot.py
====================================
Methods for plotting Virga-Sniffer output, in the form of xarray dataset accessor "vsplot".

"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.dates import DateFormatter


@xr.register_dataset_accessor("vsplot")
class VirgaSnifferPlotAccessor:
    """
    Add plotting functionality to output xr.Dataset of virga-sniffer as xr.Dataset.vsplot.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._time = pd.to_datetime(self._obj.time)

    def plot_cbh(self,
                 ax=None,
                 colors=None,
                 cbh=True,
                 cth=True,
                 lcl=True,
                 fill=True,
                 colorbar=True,
                 # color_lcl = "#beaed4",
                 label_lcl="LCL from sfc-obs",
                 # color_fill = "#984ea3",
                 label_fill="cloud base interpolated",
                 ) -> None:
        """
        Plot all cloud layer related output: cloud-base height, cloud-top height, lifitng-condensation-level,
         and filled layer.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes to assign the lines. If None uses matplotlib.pyplot.gca().
        colors: list(str), optional
            List of colors as matplotlib string for cloud-layers. If list is shorter than layer-number, it gets repreated,
            if list is longer, it gets truncated.
        cbh: bool, optional
            Plot cloud-base heights if True. The default is True.
        cth: bool, optional
            Plot cloud-top heights if True. The default is True.
        lcl: bool, optional
            Plot lifitng-condensation level as filled in cloud-base heights if True. The default is True.
        fill: bool, optional
            Plot filled/interpolated data of cloud-base heights if True. The default is True.
        colorbar: bool, optional
            Add a colorbar to `ax` if True. The default is True.
        label_lcl: str, optional
            Label of lifing-condensation level. The default is "LCL from sfc-obs".
        label_fill: str, optional
            Label of filled cloud-base heights. The default is "cloud base interpolated".

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()
        if colors is None:
            colors = [
                "k",
                '#e41a1c',
                '#ff7f00',
                '#ffff33',
                '#377eb8',
                '#a65628',
            ]
        # pad colors to length
        Npad = self._obj.layer.size - len(colors)
        if Npad > 0:
            colors = np.pad(colors, Npad, mode='wrap')[Npad:]
        elif Npad < 0:
            colors = colors[:Npad]
        else:
            pass

        if colorbar:
            # Using contourf to provide my colorbar info, then clearing the figure
            Z = np.zeros((2, 2))
            levels = np.arange(len(colors) + 1)
            pl1 = ax.contourf([self._time[0], self._time[0]],
                              [self._obj.range[0], self._obj.range[0]],
                              Z, levels, cmap=mcolors.ListedColormap(colors))

            cbar = plt.colorbar(pl1, ax=ax, fraction=0.13, pad=0.025)
            cbar.ax.set_ylabel(f"cloud base/top layer [-]", fontsize=14)
            cbar.set_ticks(np.arange(len(colors)) + 0.5)
            cbar.ax.set_yticklabels(np.arange(len(colors)))
            cbar.ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=4)
            cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)

        idx_fill = self._obj.flag_cbh_interpolated.values
        idx_lcl = self._obj.flag_lcl_filled.values
        cbhlayer = self._obj.cloud_base_height.values.copy()
        cbhlayer[idx_fill] = np.nan
        cbhlayer[idx_lcl, 0] = np.nan

        lcllayer = self._obj.cloud_base_height.values.copy()[:, 0]
        lcllayer[~idx_lcl] = np.nan

        filllayer = self._obj.cloud_base_height.values.copy()
        filllayer[~idx_fill] = np.nan

        for l in range(cbhlayer.shape[1]):
            c = colors[l]
            # plot cloud bases
            if cbh:
                ax.plot(self._time, cbhlayer[:, l],
                        linewidth=3,
                        label=f'cloud base {l + 1}',
                        color=c)
            # plot cloud tops
            if cth:
                cthv = self._obj.cloud_top_height.values[:, l]
                ax.plot(self._time, cthv,
                        linewidth=2,
                        label=f'cloud top {l + 1}',
                        linestyle=':',
                        color=c)

        # plot lcl
        if lcl:
            ax.plot(self._time, lcllayer,
                    linewidth=2,
                    # color=color_lcl,
                    color=colors[0],
                    linestyle='--',
                    label=label_lcl)

        # plot interpolated cloud base parts
        if fill:
            for l in range(cbhlayer.shape[1]):
                ax.plot(self._time, filllayer[:, l],
                        linewidth=2,
                        # color=color_fill,
                        color=colors[l],
                        linestyle='--',
                        label=label_fill if l == 0 else "")

    def plot_flag_virga(self,
                        ax=None,
                        color="#fb9a99"):
        """
        Plot Virga flag.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes to assign the lines. If None uses matplotlib.pyplot.gca().
        color: str, optional
            Matplotlib color, the default is "#fb9a99".

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()

        plt_vmask = self._obj.flag_virga.values.astype(float)
        plt_vmask[plt_vmask == 0] = np.nan

        ax.contourf(self._time,
                    self._obj.range,
                    plt_vmask.T,
                    levels=1,
                    colors=[color],
                    alpha=0.8)

    def plot_flag_cloud(self,
                        ax=None,
                        color="#a6cee3"):
        """
         Plot cloud flag.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         color: str, optional
             Matplotlib color, the default is "#a6cee3".

         Returns
         -------
         None

         """

        if ax is None:
            ax = plt.gca()

        plt_cmask = self._obj.flag_cloud.values.astype(float)
        plt_cmask[plt_cmask == 0] = np.nan

        ax.contourf(self._time,
                    self._obj.range,
                    plt_cmask.T,
                    levels=1,
                    colors=[color],
                    alpha=0.8)

    def plot_flag_ze(self,
                     ax=None,
                     color="#7fc97f"):
        """
         Plot radar signal flag.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         color: str, optional
             Matplotlib color, the default is "#7fc97f".

         Returns
         -------
         None

         """
        if ax is None:
            ax = plt.gca()

        plt_radar = (~np.isnan(self._obj.Ze.values)).astype(float)
        plt_radar[plt_radar == 0] = np.nan
        ax.contourf(self._time,
                    self._obj.range,
                    plt_radar.T,
                    levels=1,
                    colors=[color])

    def plot_flag_surface_rain(self,
                               ax=None,
                               scale=None,
                               color="k"):
        """
         Plot rain at surface flag.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         scale: float, optional
             Scaling the flag value, e.g. set maximum. The default is 0.9*min(output.range).
         color: str, optional
             Matplotlib color, the default is "k.

         Returns
         -------
         None

         """
        if ax is None:
            ax = plt.gca()
        if scale is None:
            scale = self._obj.range.values[0] * 0.9

        ax.axhline(scale, color=(0.3, 0.3, 0.3), linestyle=':', linewidth=1)
        ax.text(self._time[0], 0, "flag_rain", fontsize=12)
        ax.fill_between(self._time,
                        self._obj.flag_surface_rain.values * scale, color=color)

    def quicklook_ze(self,
                     ax=None,
                     ylim=None,
                     radar='LIMRAD94'):
        """
         Plot formatted quicklook of radar reflectivity.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         ylim: float, optional
             Limit y-axis to altitude [m]. The default is np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3.
         radar: str, optional
             Add radar name to colobar label, the default is "LIMRAD94".

         Returns
         -------
         None

         """
        if ax is None:
            ax = plt.gca()
        if ylim is None:
            # limit altitude to next full km above max cloud top height
            ylim = np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3

        pl1 = ax.pcolormesh(self._time,
                            self._obj.range,
                            self._obj.Ze.values.T,
                            cmap='jet', vmin=-60, vmax=20)
        self.plot_cbh(ax=ax, colorbar=False)
        ax.set_ylim([0, ylim])
        ax.set_yticks(ax.get_yticks(), ax.get_yticks() * 1e-3)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

        ax.set_ylabel('Altitude [km]', fontsize=15)
        ax.set_xlabel('Time UTC', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

        # cax = ax1.inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax1.transAxes)
        cbar = plt.colorbar(pl1, ax=ax, fraction=0.13, pad=0.025)
        cbar.ax.set_ylabel(f"{radar} Ze [dBZ]", fontsize=14)

        cbar.set_ticks(np.arange(-60, 30, 10))
        cbar.ax.set_yticklabels(np.arange(-60, 30, 10))
        cbar.ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=4)
        cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)

    def quicklook_vel(self,
                      ax=None,
                      ylim=None,
                      radar='LIMRAD94'):
        """
         Plot formatted quicklook of radar doppler velocity.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         ylim: float, optional
             Limit y-axis to altitude [m]. The default is np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3.
         radar: str, optional
             Add radar name to colobar label, the default is "LIMRAD94".

         Returns
         -------
         None

         """
        if ax is None:
            ax = plt.gca()
        if ylim is None:
            # limit altitude to next full km above max cloud top height
            ylim = np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3

        pl2 = ax.pcolormesh(self._time,
                            self._obj.range,
                            self._obj.vel.values.T,
                            cmap='jet',
                            vmin=-4, vmax=3)
        self.plot_cbh(ax=ax, colorbar=False)

        ax.set_ylim([0, ylim])
        ax.set_yticks(ax.get_yticks(), ax.get_yticks() * 1e-3)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.set_ylabel('Altitude [km]', fontsize=15)
        ax.set_xlabel('Time UTC', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

        # cax = ax1.inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax1.transAxes)
        cbar = plt.colorbar(pl2, ax=ax, fraction=0.13, pad=0.025)
        cbar.ax.set_ylabel(f"{radar} Mean Doppler Velocity [m/s]", fontsize=14)
        cbar.set_ticks(np.arange(-4, 4, 1))
        cbar.ax.set_yticklabels(np.arange(-4, 4, 1))
        cbar.ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=4)
        cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)

    def quicklook_flag_virga(self,
                             ax=None,
                             ylim=None, legend=True):
        """
         Plot formatted quicklook of virga-sniffer output.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         ylim: float, optional
             Limit y-axis to altitude [m]. The default is np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3.
         legend: bool, optional
             If True add legend to plot. The default is True.

         Returns
         -------
         None

         """
        if ax is None:
            ax = plt.gca()
        if ylim is None:
            # limit altitude to next full km above max cloud top height
            # add one km to make room for legends
            ylim = (1. + np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3)) * 1e3

        self.plot_flag_ze(ax=ax)
        self.plot_flag_virga(ax=ax)
        self.plot_flag_cloud(ax=ax)
        self.plot_cbh(ax=ax)
        self.plot_flag_surface_rain(ax=ax)
        if legend:
            virga_patch = mpatches.Patch(color="#fb9a99", label='flag_virga')
            cloud_patch = mpatches.Patch(color="#a6cee3", label='flag_cloud')
            radar_patch = mpatches.Patch(color="#7fc97f", label='radar-signal')
            cloud_base_line = Line2D([0], [0], color='k', lw=2)
            cloud_layer_fill = Line2D([0], [0], color='k', lw=2, ls='--')
            cloud_top_line = Line2D([0], [0], color='k', lw=2, ls=':')

            ax.legend([cloud_base_line, cloud_top_line, cloud_layer_fill,
                       radar_patch, virga_patch, cloud_patch],
                      ['cloud-base', 'cloud-top', 'filled cloud-base',
                       'radar-signal', 'flag_virga', 'flag_cloud'],
                      fontsize=14,
                      ncol=2)
        ax.set_ylim([0, ylim])
        ax.set_yticks(ax.get_yticks(), ax.get_yticks() * 1e-3)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.set_ylabel('Altitude [km]', fontsize=15)
        ax.set_xlabel('Time UTC', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

    def quicklook_full(self,
                       axs=None,
                       ylim=None,
                       radar='LIMRAD94'):
        """
         3-Panel combined formatted quicklook of radar reflectivity, doppler velocity and virga-sniffer output.

         Parameters
         ----------
         axs: list(matplotlib.axes.Axes) of length 3, optional
             The axes to assign the lines. If None uses plt.subplots(3, 1, figsize=(15, 15), constrained_layout=True)
         ylim: float, optional
             Limit y-axis to altitude [m]. The default is np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3.
         radar: str, optional
             Add radar name to colobar label, the default is "LIMRAD94".

         Returns
         -------
         None

         """
        if axs is None:
            # ,figsize=(10,8)
            fig, axs = plt.subplots(3, 1, figsize=(15, 15), constrained_layout=True)

        if ylim is None:
            # limit altitude to next full km above max cloud top height
            ylim = np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3

        stime = pd.to_datetime(self._time.values[0])
        etime = pd.to_datetime(self._time.values[-1])
        axs[0].set_title(f"{stime:%d.%m.%Y %H:%M} UTC - {etime:%d.%m.%Y %H:%M} UTC",
                         fontsize=24, fontweight='bold')

        self.quicklook_ze(ax=axs[0],
                          ylim=ylim,
                          radar=radar)
        self.quicklook_vel(ax=axs[1],
                           ylim=ylim,
                           radar=radar)
        self.quicklook_flag_virga(ax=axs[2],
                                  ylim=ylim)

        axs[0].set_xlabel("")
        axs[1].set_xlabel("")
        axs[2].set_xlabel('Time UTC', fontsize=15)
        fig = axs[0].get_figure()
        fig.autofmt_xdate()
        return fig, axs
