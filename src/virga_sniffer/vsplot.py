"""
vsplot.py
====================================
Methods for plotting Virga-Sniffer output, in the form of xarray dataset accessor "vsplot".

"""
from typing import Tuple, Any
import xarray as xr
import numpy as np
import pandas as pd
# import matplotlib
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
                 fontsize=None,
                 # color_lcl = "#beaed4",
                 label_lcl="LCL from sfc-obs",
                 # color_fill = "#984ea3",
                 label_fill="cloud base interpolated",
                 ):
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
        fontsize: dict, optional
            Configuration of fontsizes. Keywords and Defaults: "cbar_label"=16, "cbar_ticklabel"=14
        label_lcl: str, optional
            Label of lifing-condensation level. The default is "LCL from sfc-obs".
        label_fill: str, optional
            Label of filled cloud-base heights. The default is "cloud base interpolated".

        Returns
        -------
        matplotlib.axes.Axes, matplotlib.colorbar.Colorbar

        """
        fontsize_default = dict(
            cbar_label=16,
            cbar_ticklabel=14,
        )
        if fontsize is None:
            fontsize = fontsize_default
        fontsize = {**fontsize_default, **fontsize}

        if ax is None:
            ax = plt.gca()
        if colors is None:
            # colors = [
            #     "k",
            #     '#e41a1c',
            #     '#ff7f00',
            #     '#ffff33',
            #     '#377eb8',
            #     '#a65628',
            # ]
            colors = [
                'k',
                '#fdc086',
                '#7fc97f',
                '#beaed4',
                '#386cb0',
                '#ffff99',
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
            cbar.ax.set_ylabel(f"cloud base/top layer number", fontsize=fontsize['cbar_label'])
            cbar.set_ticks(np.arange(len(colors)) + 0.5)
            cbar.ax.set_yticklabels(np.arange(len(colors)))
            cbar.ax.tick_params(axis='both', which='major',
                                labelsize=fontsize['cbar_ticklabel'], width=2, length=4)
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

        for ilayer in range(cbhlayer.shape[1]):
            c = colors[ilayer]
            # plot cloud bases
            if cbh:
                ax.plot(self._time, cbhlayer[:, ilayer],
                        linewidth=3,
                        label=f'cloud base {ilayer + 1}',
                        color=c)
            # plot cloud tops
            if cth:
                cthv = self._obj.cloud_top_height.values[:, ilayer]
                ax.plot(self._time, cthv,
                        linewidth=2,
                        label=f'cloud top {ilayer + 1}',
                        linestyle='--',
                        color=c)

        # plot lcl
        if lcl:
            ax.plot(self._time, lcllayer,
                    linewidth=2,
                    # color=color_lcl,
                    color=colors[0],
                    linestyle=':',
                    label=label_lcl)

        # plot interpolated cloud base parts
        if fill:
            for ilayer in range(cbhlayer.shape[1]):
                ax.plot(self._time, filllayer[:, ilayer],
                        linewidth=2,
                        # color=color_fill,
                        color=colors[ilayer],
                        linestyle=':',
                        label=label_fill if ilayer == 0 else "")
        if colorbar:
            return ax, cbar
        else:
            return ax, 0

    def plot_virga_mask(self,
                        ax=None,
                        color="#d95f02"):
        """
        Plot Virga flag.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes to assign the lines. If None uses matplotlib.pyplot.gca().
        color: str, optional
            Matplotlib color, the default is "#d95f02".

        Returns
        -------
        matplotlib.axes.Axes

        """
        if ax is None:
            ax = plt.gca()

        plt_vmask = self._obj.mask_virga.values.astype(float)
        plt_vmask[plt_vmask == 0] = np.nan

        ax.contourf(self._time,
                    self._obj.range,
                    plt_vmask.T,
                    levels=1,
                    colors=[color],
                    alpha=1)
        return ax
    
    def plot_surface_precip(self,
                            ax=None,
                            color="#1dabed"):
        if ax is None:
            ax = plt.gca()

        plt_pmask = self._obj.mask_precip_layer.values[:,:,0].astype(float)
        plt_pmask[plt_pmask == 0] = np.nan
        ax.contourf(self._time,
                self._obj.range,
                plt_pmask.T,
                levels=1,
                colors=[color],
                alpha=1)

        return ax

    def plot_cloud_mask(self,
                        ax=None,
                        color="#7570b3"):
        """
         Plot cloud flag.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         color: str, optional
             Matplotlib color, the default is "#7570b3".

         Returns
         -------
         matplotlib.axes.Axes

         """

        if ax is None:
            ax = plt.gca()

        plt_cmask = self._obj.mask_cloud.values.astype(float)
        plt_cmask[plt_cmask == 0] = np.nan

        ax.contourf(self._time,
                    self._obj.range,
                    plt_cmask.T,
                    levels=1,
                    colors=[color],
                    alpha=1)
        return ax

    def plot_ze_mask(self,
                     ax=None,
                     color="#bdbdbd"):
        """
         Plot radar signal flag.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         color: str, optional
             Matplotlib color, the default is "#969696".

         Returns
         -------
         matplotlib.axes.Axes

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
        return ax

    def plot_flag_rain(self,
                       ax=None,
                       scale=None,
                       fontsize=None,
                       color="k"):
        """
         Plot rain at surface flag.

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         scale: float, optional
             Scaling the flag value, e.g. set maximum. The default is 0.9*min(output.range).
         fontsize: dict, optional
             Configuration of fontsizes. Keywords and Defaults: "legend"=16
         color: str, optional
             Matplotlib color, the default is "k.

         Returns
         -------
         matplotlib.axes.Axes
         """
        fontsize_default = dict(
            legend=16,
        )
        if fontsize is None:
            fontsize = fontsize_default
        fontsize = {**fontsize_default, **fontsize}
        if ax is None:
            ax = plt.gca()
        if scale is None:
            scale = self._obj.range.values[0] * 0.9

        # ax.axhline(scale, color=(0.3, 0.3, 0.3), linestyle=':', linewidth=1)
        # ax.text(self._time[0], 0, "flag_rain", fontsize=fontsize['legend'])
        ax.text(0.005, 0.01, "Rain flag:",
                fontsize=fontsize['legend'],
                ha="left", va="bottom", transform=ax.transAxes,
                bbox=dict(facecolor='w', edgecolor='k', alpha=0.7,boxstyle="Square, pad=0.1"))
        rainflag = self._obj.flag_surface_rain.values
        rainflag += self._obj.flag_lowest_rg_rain.values
        ax.fill_between(self._time, rainflag*scale, ec=color,hatch='.')
        return ax

    def quicklook_ze(self,
                     ax=None,
                     vmin=-40,
                     vmax=20,
                     ylim=None,
                     fontsize=None,
                     rasterized=True):
        """
         Plot formatted quicklook of radar reflectivity.
         The reflectivity is plotted using the `pride` colormap from CMasher (https://doi.org/10.21105/joss.02004).

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         vmin: float, optional
             Lower limit of the colorbar. If None, vmin is set to np.floor(np.nanmin(`vel`)).
             The default is -40.
         vmax: float, optional
             Upper limit of the colorbar. If None, vmax is set to np.ceil(np.nanmax(`vel`)).
             The default is 20.
         ylim: float, optional
             Limit y-axis to altitude [m].
             The default is np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3.
         fontsize: dict, optional
            Configuration of fontsizes.
            Keywords and Defaults: "ax_label"=16, "ax_ticklabel"=14, "cbar_label"=16,
            "cbar_ticklabel"=14
         rasterized: bool, optional
            Turn on rasterization of matplotlib.pyplot.pcolormesh. The default is True.

         Returns
         -------
         matplotlib.axes.Axes, matplotlib.colorbar.Colorbar

         """
        from .cmap import pride

        fontsize_default = dict(
            ax_label=16,
            ax_ticklabel=14,
            cbar_label=16,
            cbar_ticklabel=14,
        )
        if fontsize is None:
            fontsize = fontsize_default
        fontsize = {**fontsize_default, **fontsize}

        if ax is None:
            ax = plt.gca()
        if ylim is None:
            # limit altitude to next full km above max cloud top height
            # fallback is cloud-base height, if no cloud top is detected in the scene
            ylim = np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3
            if np.isnan(ylim):
                ylim = np.ceil(np.nanmax(self._obj.cloud_base_height) * 1e-3) * 1e3
        if vmin is None:
            vmin = np.floor(0.1*np.nanmin(self._obj.Ze.values))*10
        if vmax is None:
            vmax = np.ceil(0.1*np.nanmax(self._obj.Ze.values))*10
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        pl1 = ax.pcolormesh(self._time,
                            self._obj.range,
                            self._obj.Ze.values.T,
                            cmap='cmr.pride', norm=divnorm,
                            rasterized=rasterized)
        self.plot_cbh(ax=ax, colorbar=False, colors=['k'])
        ax.set_ylim([0, ylim])
        ax.set_yticks(ax.get_yticks(), ax.get_yticks() * 1e-3)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

        ax.set_ylabel('Height (km)', fontsize=fontsize['ax_label'])
        ax.set_xlabel('Time (UTC)', fontsize=fontsize['ax_label'])
        ax.tick_params(axis='both', which='major', labelsize=fontsize['ax_ticklabel'])
        ax.grid(True)

        ax.text(0.02,0.96,"Radar reflectivity factor",
                fontsize=fontsize['legend'],
                ha="left", va="top",transform=ax.transAxes,
                bbox=dict(facecolor='w',edgecolor='k',alpha=0.7))

        cbar = plt.colorbar(pl1, ax=ax, fraction=0.13, pad=0.025, extend='both')
        cbar.ax.set_yscale('linear')
        cbar.ax.set_ylabel(f"dBz", fontsize=fontsize['cbar_label'])

        cbar.ax.tick_params(axis='both', which='major',
                            labelsize=fontsize['cbar_ticklabel'],
                            width=2, length=4)
        cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)
        return ax,cbar

    def quicklook_vel(self,
                      ax=None,
                      vmin=-4,
                      vmax=3,
                      ylim=None,
                      fontsize=None,
                      rasterized=True):
        """
         Plot formatted quicklook of radar mean Doppler velocity.
         The mean Doppler velocity is plotted using the `holly` colormap from CMasher
         (https://doi.org/10.21105/joss.02004).

         Parameters
         ----------
         ax: matplotlib.axes.Axes, optional
             The axes to assign the lines. If None uses matplotlib.pyplot.gca().
         vmin: float, optional
             Lower limit of the colorbar. If None, vmin is set to np.floor(np.nanmin(`vel`)).
             The default is -4.
         vmax: float, optional
             Upper limit of the colorbar. If None, vmax is set to np.ceil(np.nanmax(`vel`)).
             The default is 3.
         ylim: float, optional
             Limit y-axis to altitude [m].
             The default is np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3.
         fontsize: dict, optional
            Configuration of fontsizes.
            Keywords and Defaults: "legend"=16, "ax_label"=16, "ax_ticklabel"=14,
            "cbar_label"=16, "cbar_ticklabel"=14
         rasterized: bool, optional
            Turn on rasterization of matplotlib.pyplot.pcolormesh. The default is True.
         Returns
         -------
         matplotlib.axes.Axes, matplotlib.colorbar.Colorbar

         """
        from .cmap import holly
        fontsize_default = dict(
            legend=16,
            ax_label=16,
            ax_ticklabel=14,
            cbar_label=16,
            cbar_ticklabel=14,
        )
        if fontsize is None:
            fontsize = fontsize_default
        fontsize = {**fontsize_default, **fontsize}

        if ax is None:
            ax = plt.gca()
        if ylim is None:
            # limit altitude to next full km above max cloud top height
            # fallback is cloud-base height, if no cloud top is detected in the scene
            ylim = np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3
            if np.isnan(ylim):
                ylim = np.ceil(np.nanmax(self._obj.cloud_base_height) * 1e-3) * 1e3
                
        if vmin is None:
            vmin = np.floor(np.nanmin(self._obj.vel.values))
        if vmax is None:
            vmax = np.ceil(np.nanmax(self._obj.vel.values))
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        pl2 = ax.pcolormesh(self._time,
                            self._obj.range,
                            self._obj.vel.values.T,
                            norm=divnorm,
                            cmap='cmr.holly',
                            rasterized=rasterized)
        self.plot_cbh(ax=ax, fontsize=fontsize, colorbar=False, colors=['k'])
 
        ax.set_ylim([0, ylim])
        ax.set_yticks(ax.get_yticks(), ax.get_yticks() * 1e-3)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.set_ylabel('Height (km)', fontsize=fontsize['ax_label'])
        ax.set_xlabel('Time (UTC)', fontsize=fontsize['ax_label'])
        ax.tick_params(axis='both', which='major', labelsize=fontsize['ax_ticklabel'])
        ax.grid(True)
        ax.text(0.02, 0.96, "Mean Doppler velocity",
                fontsize=fontsize['legend'],
                ha="left", va="top", transform=ax.transAxes,
                bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

        cbar = plt.colorbar(pl2, ax=ax, fraction=0.13, pad=0.025, extend='both')
        cbar.ax.set_yscale('linear')
        cbar.ax.set_ylabel(f"m/s", fontsize=fontsize['cbar_label'])
        cbar.ax.tick_params(axis='both', which='major', labelsize=fontsize['cbar_ticklabel'], width=2, length=4)
        cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)
        return ax, cbar

    def quicklook_virga_mask(self,
                             ax=None,
                             ylim=None,
                             legend=True,
                             fontsize=None,
                             plot_flags=None,
                            ):
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
         fontsize: dict, optional
             Configuration of fontsizes. Keywords and Defaults: "legend"=16, "ax_label"=16, "ax_ticklabel"=14,
             "cbar_label"=16, "cbar_ticklabel"=14
         plot_flags: dict, optional
             Components to plot. Keywords and Defaults: "ze"=True, "virga"=True, "cloud"=True, "rainflag"=True, "surface_precip"=True

         Returns
         -------
         matplotlib.axes.Axes, matplotlib.colorbar.Colorbar

         """
        plot_flags_default = dict(
            ze=True,
            virga=True,
            cloud=True,
            rainflag=True,
            surface_precip=True
        )
        fontsize_default = dict(
            legend=16,
            ax_label=16,
            ax_ticklabel=14,
            cbar_label=16,
            cbar_ticklabel=14,
        )
        if fontsize is None:
            fontsize = fontsize_default
        if plot_flags is None:
            plot_flags = plot_flags_default

        fontsize = {**fontsize_default, **fontsize}
        plot_flags = {**plot_flags_default, **plot_flags}

        if ax is None:
            ax = plt.gca()
        if ylim is None:
            # limit altitude to next full km above max cloud top height
            # fallback is cloud-base height, if no cloud top is detected in the scene
            ylim = np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3
            if np.isnan(ylim):
                ylim = np.ceil(np.nanmax(self._obj.cloud_base_height) * 1e-3) * 1e3

        if plot_flags['ze']:
            self.plot_ze_mask(ax=ax)
        if plot_flags['surface_precip']:
            self.plot_surface_precip(ax=ax)
        if plot_flags['virga']:
            self.plot_virga_mask(ax=ax)
        if plot_flags['cloud']:
            self.plot_cloud_mask(ax=ax)
        if plot_flags['rainflag']:
            self.plot_flag_rain(ax=ax,fontsize=fontsize)
        ax, cbar = self.plot_cbh(ax=ax,fontsize=fontsize)
        
        if legend:
            # virga_patch = mpatches.Patch(color="#fb9a99", label='virga mask')
            # cloud_patch = mpatches.Patch(color="#a6cee3", label='cloud mask')
            # radar_patch = mpatches.Patch(color="#7fc97f", label='radar signal')
            virga_patch = mpatches.Patch(color="#d95f02", label='virga mask')
            sprecip_patch = mpatches.Patch(color="#1dabed", label='precip mask')
            cloud_patch = mpatches.Patch(color="#7570b3", label='cloud mask')
            radar_patch = mpatches.Patch(color="#bdbdbd", label='unclassified')
            empty_patch = mpatches.Patch(color="#ffffff00", label='empty')
            cloud_base_line = Line2D([0], [0], color='k', lw=2)
            cloud_layer_fill = Line2D([0], [0], color='k', lw=2, ls=':')
            cloud_top_line = Line2D([0], [0], color='k', lw=2, ls='--')

            ax.legend([cloud_base_line, cloud_top_line, cloud_layer_fill,empty_patch,
                       radar_patch, virga_patch, sprecip_patch, cloud_patch],
                      ['cloud base', 'cloud top', 'filled cloud base','',
                       'unclassified', 'virga mask', 'precip mask', 'cloud mask'],
                      fontsize=fontsize['legend'],
                      ncol=2)
        ax.set_ylim([0, ylim])
        ax.set_yticks(ax.get_yticks(), ax.get_yticks() * 1e-3)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.set_ylabel('Height (km)', fontsize=fontsize['ax_label'])
        ax.set_xlabel('Time (UTC)', fontsize=fontsize['ax_label'])
        ax.tick_params(axis='both', which='major', labelsize=fontsize['ax_ticklabel'])
        ax.grid(True)
        ax.text(0.02, 0.96, "Virga-Sniffer output",
                fontsize=fontsize['legend'],
                ha="left", va="top", transform=ax.transAxes,
                bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
        return ax,cbar

    def quicklook_full(self,
                       axs=None,
                       ylim=None,
                       radar='LIMRAD94',
                       fontsize=None,
                       plot_flags=None):
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
         fontsize: dict, optional
             Configuration of fontsizes. Keywords and Defaults: "title"=18, "legend"=16, "ax_label"=16,
             "ax_ticklabel"=14, "cbar_label"=16, "cbar_ticklabel"=14
         plot_flags: dict, optional
             Components to plot. Keywords and Defaults: "ze"=True, "virga"=True, "cloud"=True, "rainflag"=True

         Returns
         -------
         matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.colorbar.Colorbar

         """
        plot_flags_default = dict(
            ze=True,
            virga=True,
            cloud=True,
            rainflag=True,
            surface_precip=True,
        )
        fontsize_default = dict(
            title=18,
            legend=16,
            ax_label=16,
            ax_ticklabel=14,
            cbar_label=16,
            cbar_ticklabel=14,
        )
        if fontsize is None:
            fontsize = fontsize_default
        if plot_flags is None:
            plot_flags = plot_flags_default

        fontsize = {**fontsize_default, **fontsize}
        plot_flags = {**plot_flags_default, **plot_flags}

        if axs is None:
            # ,figsize=(10,8)
            fig, axs = plt.subplots(3, 1, figsize=(15, 15), constrained_layout=True)

        if ylim is None:
            # limit altitude to next full km above max cloud top height
            # fallback is cloud-base height, if no cloud top is detected in the scene
            ylim = np.ceil(np.nanmax(self._obj.cloud_top_height) * 1e-3) * 1e3
            if np.isnan(ylim):
                ylim = np.ceil(np.nanmax(self._obj.cloud_base_height) * 1e-3) * 1e3

        stime = pd.to_datetime(self._time.values[0])
        etime = pd.to_datetime(self._time.values[-1])
        # axs[0].set_title(f"{stime:%d.%m.%Y %H:%M} UTC - {etime:%d.%m.%Y %H:%M} UTC",
        #                  fontsize=24, fontweight='bold')
        axs[0].set_title(f"{radar} {stime:%d %B %Y}",
                         fontsize=fontsize['title'])
        cbars = [0, 0, 0]
        axs[0], cbars[0] = self.quicklook_ze(ax=axs[0],
                                             ylim=ylim,
                                             fontsize=fontsize)
        axs[1], cbars[1] = self.quicklook_vel(ax=axs[1],
                                              ylim=ylim,
                                              fontsize=fontsize)
        axs[2], cbars[2] = self.quicklook_virga_mask(ax=axs[2],
                                                     fontsize=fontsize,
                                                     plot_flags=plot_flags,
                                                     ylim=ylim)

        axs[0].set_xlabel("")
        axs[1].set_xlabel("")
        axs[2].set_xlabel('Time (UTC)', fontsize=fontsize['ax_label'])
        fig = axs[0].get_figure()
        fig.autofmt_xdate()
        return fig, axs, cbars
