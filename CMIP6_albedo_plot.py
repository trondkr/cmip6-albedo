import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import box, mapping
import dateutil
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
#matplotlib.use('Qt5Agg')
import logging

class CMIP6_albedo_plot():

    def plot_spectral_irradiance(self, spectra,latitude):
        fig, ax = plt.subplots()
        ax.plot(spectra['wavelength'], spectra['poa_global'][:,0])

        plt.xlim(200, 2700)
        plt.ylim(0, 1.8)
        plt.title(r"Day 80 1984, $\tau=0.1$, Wv=0.5 cm lat {}".format(latitude))
        plt.ylabel(r"Irradiance ($W m^{-2} nm^{-1}$)")
        plt.xlabel(r"Wavelength ($nm$)")
       # time_labels = times.strftime("%H:%M %p")

       # plt.legend(labels)
        plt.savefig("spectral_test_{}.png".format(latitude))

    def create_plots(self, lon, lat, model_object, sisnconc=None, sisnthick=None, sithick=None, siconc=None, \
                     clt=None, chl=None, rads=None, irradiance_water=None, wind=None, OSA=None, OSA_UV=None, \
                     OSA_VIS=None, OSA_NIR=None, albedo=None, direct_sw=None, uvi=None, plotname_postfix=None):
        # create_streamplot(dr_out_uas,dr_out_vas,wind,lon[0,:],lat[:,0],"wind",nlevels=None)
        # self.create_plot(wind,lon[0,:],lat[:,0],"wind",model_object,regional=True)
        #self.create_plot(sisnconc,lon[0,:],lat[:,0],"sisnconc",model_object,regional=True)
        #self.create_plot(sisnthick,lon[0,:],lat[:,0],"sisnthick",model_object,regional=True)
        if siconc is not None: self.create_plot(siconc,lon[0,:],lat[:,0],"siconc",model_object,regional=True)
        if sithick is not None: self.create_plot(sithick,lon[0,:],lat[:,0],"sithick",model_object,regional=True)
        if direct_sw is not None: self.create_plot(direct_sw, lon[0, :], lat[:, 0], "direct_sw", model_object,
                                                   regional=False, plotname_postfix=plotname_postfix)
        if uvi is not None: self.create_plot(uvi, lon[0, :], lat[:, 0], "UVI", model_object,
                                                   regional=False, plotname_postfix=plotname_postfix)
        # self.create_plot(clt,lon[0,:],lat[:,0],"clouds",model_object,regional=True)
        # self.create_plot(np.log(chl),lon[0,:],lat[:,0],"chl (np.lon)",model_object,regional=True)
        #self.create_plot(OSA[:,:,0],lon[0,:],lat[:,0],"OSA_direct_broadband",model_object,regional=True)

        # self.create_plot(OSA[:, :, 4], lon[0, :], lat[:, 0], "OSA_direct_uv", model_object, nlevels=np.arange(0.01, 0.04, 0.001), regional=True)
        # self.create_plot(OSA[:, :, 2], lon[0, :], lat[:, 0], "OSA_direct_par", model_object, nlevels=np.arange(0.01, 0.04, 0.001), regional=True)
        #self.create_plot(OSA[:,:,1],lon[0,:],lat[:,0],"OSA_diffuse_broadband",model_object,regional=True)
        # self.create_plot(rads[:,:,0],lon[0,:],lat[:,0],"Direct radiation",model_object,regional=True)
        # self.create_plot(rads[:,:,1],lon[0,:],lat[:,0],"Diffuse radiation",model_object, regional=True)
        # self.create_plot(rads[:,:,2],lon[0,:],lat[:,0],"Apparent zenith",model_object, regional=True)

        if OSA_VIS is not None:  self.create_plot(OSA_VIS[:,:,0], lon[0, :], lat[:, 0], "OSA_VIS_DIRECT", model_object, regional=True)
      #  self.create_plot(OSA_UV[:,:,0], lon[0, :], lat[:, 0], "OSA_UV_DIRECT", model_object, regional=True)
        if OSA_VIS is not None:  self.create_plot(OSA_VIS[:, :, 1], lon[0, :], lat[:, 0], "OSA_VIS_DIFFUSE", model_object, regional=True)
      #  self.create_plot(OSA_UV[:, :, 1], lon[0, :], lat[:, 0], "OSA_UV_DIFFUSE", model_object, regional=True)

        # self.create_plot(irradiance_water, lon[0, :], lat[:, 0], "irradiance_water", model_object, regional=True)
        if albedo is not None: self.create_plot(albedo, lon[0, :], lat[:, 0], "albedo", model_object, nlevels=[0.02,0.025, 0.03, 0.035, 0.04,0.045, 0.05, 0.06], regional=True)

    def create_streamplot(self, indata_u, indata_v, uv, lon, lat, name, nlevels=None):
        # Make data cyclic around dateline
        fig = plt.figure(figsize=(12, 12))
        proj = ccrs.PlateCarree()
        ax = plt.axes(projection=proj)

        indata_u_cyclic, lon_cyclic = add_cyclic_point(indata_u, coord=lon)
        indata_v_cyclic, lon_cyclic = add_cyclic_point(indata_v, coord=lon)

        cf = ax.contourf(lon_cyclic, lat, uv, 10,
                         cmap='RdYlBu_r',
                         extend='both',
                         transform=ccrs.PlateCarree())

        land_110m = cfeature.NaturalEarthFeature('physical', 'land', '110m')
        sp = ax.streamplot(lon_cyclic, lat, indata_u_cyclic, indata_v_cyclic,
                           linewidth=0.5,
                           arrowsize=0.4,
                           density=10,
                           color='k',
                           transform=ccrs.PlateCarree())

        cb = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50)
        cb.ax.set_title('Wind speed [m/s]')
        ax.add_feature(land_110m, color="lightgrey")
        ax.add_feature(cfeature.COASTLINE, edgecolor="black")
        plt.show()

    def create_plot(self, indata, lon, lat, name, model_object, nlevels=None, regional=False, logscale=False, plotname_postfix=None):
        plt.interactive(False)
        logging.info("[CMIP6_albedo_plot] Plotting variable {} ({})".format(name,np.shape(indata)))
        plt.clf()
        proj = ccrs.NorthPolarStereo(true_scale_latitude=70)
        ax = plt.axes(projection=proj)
        land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m')
        ax.add_feature(land_10m, color="lightgrey", edgecolor="black")
        ax.coastlines(resolution='10m', linewidth=1.5, color='black', alpha=0.8, zorder=4)

        #  if regional:
        #      land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m')
        #      ax.add_feature(land_10m, color="lightgrey", edgecolor="black")
        #      ax.set_extent([-180, 180, 45, 90])

        # indata_cyclic=indata
        # lon_cyclic = lon
        indata_cyclic, lon = add_cyclic_point(indata, coord=lon)
        if nlevels is None:
            if logscale:
                from matplotlib import ticker, cm
                cs = ax.contourf(lon, lat, indata_cyclic, 10,
                                 transform=ccrs.PlateCarree(),
                                 cmap='RdYlBu_r', locator=ticker.LogLocator(subs=range(1, 5)), extend='both')
            else:
                cs = ax.contourf(lon, lat, indata_cyclic, 20, transform=ccrs.PlateCarree(), cmap='RdYlBu_r',
                                 extend='both')
        else:
            cs = ax.contourf(lon, lat, indata_cyclic, nlevels, transform=ccrs.PlateCarree(), cmap='RdYlBu_r',
                             extend='both')
        plt.title("{}".format(name))

        #   if regional:
        #       land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m')
        #       ax.add_feature(land_10m, color="lightgrey", edgecolor="black")
        #       ax.set_extent([-180, 180, 45, 90])
        #   else:
        #       land_110m = cfeature.NaturalEarthFeature('physical', 'land', '110m')
        #       ax.add_feature(land_110m, color="lightgrey", edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        plt.colorbar(cs, shrink=0.5)
      #  plt.show(block=True)
        if not os.path.exists("Figures"):
            os.mkdir("Figures")
        plotfilename = "Figures/{}_{}_{}_{}_{}.png".format(name,
                                        model_object.name,
                                        model_object.current_member_id,
                                        model_object.current_time,
                                                           plotname_postfix)

        if os.path.exists(plotfilename):os.remove(plotfilename)
        plt.savefig(plotfilename, dpi=150, bbox_inches='tight')

    def create_plots_compare(self, ds2020, ds2050):  # only used for comparisons

        ds_diff = ds2050 - ds2020

        notnull = pd.notnull(ds2050['chl'][0])
        land_110m = cfeature.NaturalEarthFeature('physical', 'land', '110m')
        proj = ccrs.PlateCarree()
        extent = [-20, 20, 50, 80]

        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 16), subplot_kw={'projection': proj})
        for i, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
            ds2020['chl'].sel(season=season).where(notnull).plot.pcolormesh(
                ax=axes[i, 0], cmap='Spectral_r', transform=ccrs.PlateCarree(),  # vmin=-30, vmax=30, cmap='Spectral_r',
                add_colorbar=True, extend='both')
            axes[i, 0].set_extent(extent, crs=proj)
            axes[i, 0].add_feature(land_110m, color="lightgrey")
            axes[i, 0].add_feature(cfeature.COASTLINE, edgecolor="black")
            axes[i, 0].add_feature(cfeature.BORDERS, linestyle=':')

            ds2050['chl'].sel(season=season).where(notnull).plot.pcolormesh(
                ax=axes[i, 1], cmap='Spectral_r', transform=ccrs.PlateCarree(),  # vmin=-30, vmax=30, cmap='Spectral_r',
                add_colorbar=True, extend='both')
            axes[i, 1].set_extent(extent, crs=proj)
            axes[i, 1].add_feature(land_110m, color="lightgrey")
            axes[i, 1].add_feature(cfeature.COASTLINE, edgecolor="black")
            axes[i, 1].add_feature(cfeature.BORDERS, linestyle=':')

            ds_diff['chl'].sel(season=season).where(notnull).plot.pcolormesh(
                ax=axes[i, 2], cmap='Spectral_r', transform=ccrs.PlateCarree(),  # vmin=-0.1, vmax=.1, cmap='RdBu_r',
                add_colorbar=True, extend='both')
            axes[i, 2].set_extent(extent, crs=proj)
            axes[i, 2].add_feature(land_110m, color="lightgrey")
            axes[i, 2].add_feature(cfeature.COASTLINE, edgecolor="black")
            axes[i, 2].add_feature(cfeature.BORDERS, linestyle=':')

            axes[i, 0].set_ylabel(season)
            axes[i, 1].set_ylabel('')
            axes[i, 2].set_ylabel('')

        for ax in axes.flat:
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])
            ax.axes.axis('tight')
            ax.set_xlabel('')

        axes[0, 0].set_title('ds2020')
        axes[0, 1].set_title('ds2050')
        axes[0, 2].set_title('Difference')

        plt.tight_layout()

        fig.suptitle('Seasonal Chlorophyll', fontsize=16, y=1.02)
        plt.show()
