import datetime
import logging
import os
from typing import List, Any

# Computational modules
import dask
import numpy as np
import pandas as pd
import pvlib
import xarray as xr
import xesmf as xe
from distributed import Client, LocalCluster

import CMIP6_IO
import CMIP6_albedo_plot
import CMIP6_albedo_utils
import CMIP6_config
import CMIP6_date_tools
import CMIP6_regrid
import CMIP6_ccsm3
from CMIP6_model import CMIP6_MODEL


class CMIP6_light:

    def __init__(self):

        self.config = CMIP6_config.Config_albedo()
        if not self.config.use_local_CMIP6_files:
            self.config.read_cmip6_repository()
        self.cmip6_models: List[Any] = []

    # Required setup for doing light calculations, but only required once per timestep.
    def setup_pv_system(self, month, hour_of_day):
        offset = 0  # int(lon_180/15.)
        when = [datetime.datetime(2006, month, 15, hour_of_day, 0, 0,
                                  tzinfo=datetime.timezone(datetime.timedelta(hours=offset)))]
        time = pd.DatetimeIndex(when)

        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

        module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
        inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        pv_system = {'module': module, 'inverter': inverter,
                     'surface_azimuth': 180}

        return time, pv_system

    def calculate_zenith(self, latitude, ctime):

        longitude = 0.0
        # get_solar-position returns a Pandas dataframe with index so we convert the value
        # to numpy after calculating
        solpos = pvlib.solarposition.get_solarposition(ctime, latitude, longitude)
        return np.squeeze(solpos['zenith'])

    def radiation(self, cloud_covers, latitude, ctime, system, albedo):
        results = np.zeros((np.shape(cloud_covers)[0], 3))
        altitude = 0.0

        # Some calculations are done only on Greenwhich meridian line as they are identical around the globe at the
        # same latitude. For that reason longitude is set to Greenwhich meridian and do not change. The only reason
        # to use longitude would be to have the local sun position for given time but since we calculate position at
        # the same time of the day (hour_of_day) and month (month) we can assume its the same across all longitudes,
        # and only change with latitude.

        longitude = 0.0
        # get_solar-position returns a Pandas dataframe with index so we convert the value
        # to numpy after calculating

        solpos = pvlib.solarposition.get_solarposition(ctime, latitude, longitude)

        airmass_relative = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'].to_numpy(), model='kasten1966')
        pressure = pvlib.atmosphere.alt2pres(altitude)
        airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_relative, pressure)

        airmass_abs_array = np.ones((np.shape(cloud_covers)))
        airmass__abs_array = airmass_abs_array * airmass_abs

        am_rel_array = np.ones((np.shape(cloud_covers)))
        am_rel_array = am_rel_array * airmass_relative

        apparent_zenith = np.ones((np.shape(cloud_covers)))
        apparent_zenith = apparent_zenith * solpos['apparent_zenith'].to_numpy()

        azimuth = np.ones((np.shape(cloud_covers)))
        azimuth = azimuth * solpos['azimuth'].to_numpy()

        surface_tilt = np.ones((np.shape(cloud_covers)))
        surface_tilt = surface_tilt * latitude

        surface_azimuth = np.ones((np.shape(cloud_covers)))
        surface_azimuth = surface_azimuth * system['surface_azimuth']


        # cloud cover in fraction units here
        transmittance = (1.0 - cloud_covers) * 0.75
        # irrads is a DataFrame containing ghi, dni, dhi

        irrads = pvlib.irradiance.liujordan(solpos['apparent_zenith'].to_numpy(), transmittance,
                                                         airmass_abs_array)

        print("LIUJ: dni: {} ghi: {} dhi: {}".format(np.nanmean(irrads['dni']),
              np.nanmean(irrads['ghi']),
              np.nanmean(irrads['dhi'])))

        total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt,
                                                            surface_azimuth,
                                                            apparent_zenith,
                                                            azimuth,
                                                            irrads['dni'],
                                                            irrads['ghi'],
                                                            irrads['dhi'],
                                                            model='isotropic',
                                                            albedo=albedo)
        results[:, 0] = total_irrad['poa_direct']
        results[:, 1] = total_irrad['poa_diffuse']
        results[:, 2] = solpos['zenith']
        return results

    def season_mean(self, ds, calendar='standard'):
        # Make a DataArray of season/year groups

        year_season = xr.DataArray(ds.time.to_index().to_period(freq='Q-NOV').to_timestamp(how='E'),
                                   coords=[ds.time], name='year_season')

        # Make a DataArray with the number of days in each month, size = len(time)
        date_tool = CMIP6_date_tools.CMIP6_date_tools()
        month_length = xr.DataArray(date_tool.get_dpm(ds.time.to_index(), calendar=calendar),
                                    coords=[ds.time], name='month_length')
        # Calculate the weights by grouping by 'time.season'
        weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

        # Test that the sum of the weights for each season is 1.0
        np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

        # Calculate the weighted average
        return (ds * weights).groupby('time.season').sum(dim='time')

    def create_chlorophyll_avg_for_year(self, year, ds_in, ds_out):
        start_date = '{}-01-01'.format(year)
        end_date = '{}-12-31'.format(year)
        ds_chl_2020 = ds_in.sel(time=slice(start_date, end_date))  # .mean('time')

        year2 = 2050
        start_date = '{}-01-01'.format(year2)
        end_date = '{}-12-31'.format(year2)
        ds_chl_2050 = ds_in.sel(time=slice(start_date, end_date))
        re = CMIP6_regrid.CMIP6_regrid()
        chl2020 = re.regrid_variable("chl", ds_chl_2020, ds_out)
        chl2050 = re.regrid_variable("chl", ds_chl_2050, ds_out)

        ds_2020 = chl2020.to_dataset()
        ds_2050 = chl2050.to_dataset()

        lat = ds_2020.lat.values
        lon = ds_2020.lon.values

        weighted_average_2020 = self.season_mean(ds_2020, calendar="noleap")
        weighted_average_2050 = self.season_mean(ds_2050, calendar="noleap")

        ds_diff = 100 * (weighted_average_2050 - weighted_average_2020) / weighted_average_2020
        chl2020 = weighted_average_2020.sel(season="MAM").chl.values
        chl2050 = weighted_average_2050.sel(season="MAM").chl.values
        chl2050_diff = ds_diff.sel(season="MAM").chl.values

        # kg/m3 to mg/m3 multiply by 1e6
        plotter = CMIP6_albedo_plot.CMIP6_albedo_plot()

        plotter.create_plot((chl2020 * 1.e6), lon[0, :], lat[:, 0], "chl2020", nlevels=np.arange(0, 5, 0.2),
                            regional=True,
                            logscale=True)
        plotter.create_plot((chl2050 * 1.e6), lon[0, :], lat[:, 0], "chl2050", nlevels=np.arange(0, 5, 0.2),
                            regional=True,
                            logscale=True)
        plotter.create_plot(chl2050_diff, lon[0, :], lat[:, 0], "chl2050-2020",
                            nlevels=np.arange(-101, 101, 1), regional=True)

    """
    Regrid to cartesian grid:
    For any Amon related variables (wind, clouds), the resolution from CMIP6 models is less than
    1 degree longitude x latitude. To interpolate to a 1x1 degree grid we therefore first interpolate to a
    2x2 degrees grid and then subsequently to a 1x1 degree grid.
    """

    def extract_dataset_and_regrid(self, model_obj, t_index):
        extracted: dict = {}
        if self.config.use_local_CMIP6_files:
            for key in model_obj.ds_sets[model_obj.current_member_id].keys():
                extracted[key] = model_obj.ds_sets[model_obj.current_member_id][key].isel(time=t_index).to_array()
            return extracted

        ds_out_amon = xe.util.grid_2d(self.config.min_lon,
                                      self.config.max_lon, 2,
                                      self.config.min_lat,
                                      self.config.max_lat, 2)
        ds_out = xe.util.grid_2d(self.config.min_lon,
                                 self.config.max_lon, 1,
                                 self.config.min_lat,
                                 self.config.max_lat, 1)

        re = CMIP6_regrid.CMIP6_regrid()
        for key in model_obj.ds_sets[model_obj.current_member_id].keys():

            current_ds = model_obj.ds_sets[model_obj.current_member_id][key].isel(time=t_index).sel(
                y=slice(self.config.min_lat, self.config.max_lat),
                x=slice(self.config.min_lon, self.config.max_lon))

            if key in ["chl", "sithick", "siconc", "sisnthick", "sisnconc"]:
                ds_trans = current_ds.transpose('bnds', 'vertex', 'y', 'x')
            else:
                ds_trans = current_ds.transpose('bnds', 'y', 'x')

            if key in ["uas", "vas", "clt"]:
                out_amon = re.regrid_variable(key,
                                              ds_trans,
                                              ds_out_amon,
                                              interpolation_method=self.config.interp,
                                              use_esmf_v801=self.config.use_esmf_v801).to_dataset()

                out = re.regrid_variable(key, out_amon, ds_out,
                                         interpolation_method=self.config.interp,
                                         use_esmf_v801=self.config.use_esmf_v801)
            else:
                out = re.regrid_variable(key, ds_trans,
                                         ds_out,
                                         interpolation_method=self.config.interp,
                                         use_esmf_v801=self.config.use_esmf_v801)
            extracted[key] = out
        return extracted

    def values_for_timestep(self, extracted_ds, selected_time):

        lat = np.squeeze(extracted_ds["uas"].lat.values)
        lon = np.squeeze(extracted_ds["uas"].lon.values)
        clt = np.squeeze(extracted_ds["clt"].values)
        chl = np.squeeze(extracted_ds["chl"].values)
        sisnconc = np.squeeze(extracted_ds["sisnconc"].values)
        sisnthick = np.squeeze(extracted_ds["sisnthick"].values)
        siconc = np.squeeze(extracted_ds["siconc"].values)
        sithick = np.squeeze(extracted_ds["sithick"].values)
        uas = np.squeeze(extracted_ds["uas"].values)
        vas = np.squeeze(extracted_ds["vas"].values)
        tas = np.squeeze(extracted_ds["tas"].values)
        toz = np.squeeze(extracted_ds["toz"].values)

        percentage_to_ratio = 1 / 100.

        if np.nanmax(sisnconc) > 5:
            sisnconc = sisnconc * percentage_to_ratio
        if np.nanmax(siconc) > 5:
            siconc = siconc * percentage_to_ratio
        if np.max(clt) > 5:
            clt = clt * percentage_to_ratio

        # Calculate scalar wind and organize the data arrays to be used for  given time-step (month-year)
        wind = np.sqrt(uas ** 2 + vas ** 2)
        m = len(wind[:, 0])
        n = len(wind[0, :])

        assert np.nanmax(clt) <= 1.1, "Clouds needs to be scaled to between 0 and 1"
        assert np.nanmax(sisnconc) <= 1.1, "Sea-ice snow concentration needs to be scaled to between 0 and 1"
        assert np.nanmax(siconc) <= 1.1, "Sea-ice needs to be scaled to between 0 and 1"
        assert np.nanmax(tas) <= 45, "Temperature needs to be in Celsisus"
        assert np.nanmax(toz) <= 45, "Ozone needs to be in cm (usually 0.0-0.8cm)"

        return wind, lat, lon, clt, chl, sisnconc, sisnthick, siconc, sithick, tas, toz, m, n

    def calculate_radiation(self,
                            hour_of_day: int,
                            model_object: CMIP6_MODEL,
                            clt: np.ndarray,
                            direct_OSA: np.ndarray,
                            lat: np.ndarray,
                            m: int, n: int) -> np.ndarray:
        logging.info("[CMIP6_light] Running for hour {}".format(hour_of_day))

        ctime, pv_system = self.setup_pv_system(model_object.current_time.month, hour_of_day)
        calc_radiation = [dask.delayed(self.radiation)(clt[j, :], lat[j, 0], ctime, pv_system, direct_OSA[j, :]) for j
                          in range(m)]

        # https://github.com/dask/dask/issues/5464
        rad = dask.compute(calc_radiation)
        rads = np.asarray(rad).reshape((m, n, 3))

        return rads

    def calculate_the_effect_of_ozone(self, dr_uv, toz):
        """
        Method that estimates the effect of ozone on total UV assuming
        simple relationship from here:
        https://www.sciencedirect.com/science/article/pii/S1309104215305419
        """
        dr_uv = dr_uv * 57 * (toz) ** (-1.05)
        return dr_uv

    def perform_light_calculations(self, model_object):

        times = model_object.ds_sets[model_object.current_member_id]["uas"].time
        data_list = []

        for selected_time in range(0, len(times)):
            model_object.current_time = pd.to_datetime(times[selected_time].values)

            extracted_ds = self.extract_dataset_and_regrid(model_object, selected_time)
            logging.info("[CMIP6_light] Running for timestep {} model {}".format(model_object.current_time,
                                                                                 model_object.name))

            wind, lat, lon, clt, chl, sisnconc, sisnthick, siconc, sithick, tas, toz, m, n = self.values_for_timestep(
                extracted_ds, selected_time)

            # TODO: get ozone

            for hour_of_day in range(12, 13, 1):
                # Calculate zenith for each grid point
                # Not currently used...
                #  albedo_simple = self.cmip6_ccsm3.calculate_diffuse_albedo_per_grid_point(sisnconc=sisnconc,
                #                                                               siconc=siconc)

                ctime, pv_system = self.setup_pv_system(model_object.current_time.month, hour_of_day)
                calc_zenith = [dask.delayed(self.calculate_zenith)(lat[j, 0], ctime) for j in range(m)]
                zenith = dask.compute(calc_zenith)
                zeniths = np.asarray(zenith).reshape(m)

                scenarios = ["osa"] #, "no_chl", "no_wind", "normal"]
                for scenario in scenarios:
                    if scenario == "no_chl":
                        chl_scale = 0.0
                    else:
                        chl_scale = 1.0

                    if scenario == "no_wind":
                        wind_scale = 0.0
                    else:
                        wind_scale = 1.0
                    logging.info("[CMIP6_light] Running scenario: {}".format(scenario))
                    # Calculate OSA for each grid point (this is without the effect of sea ice and snow)
                    zr = [CMIP6_albedo_utils.calculate_OSA(zeniths[i], wind[i, j] * wind_scale,
                                                           chl[i, j] * chl_scale,
                                                           self.config.wavelengths,
                                                           self.config.refractive_indexes,
                                                           self.config.alpha_chl,
                                                           self.config.alpha_w,
                                                           self.config.beta_w,
                                                           self.config.alpha_wc,
                                                           self.config.solar_energy) for i in range(m) for j in
                          range(n)]

                    res = np.squeeze(np.asarray(dask.compute(zr)))
                    OSA = res[:, 0, :].reshape((m, n, 2))
                    direct_OSA = np.squeeze(OSA[:, :, 0])
                    diffuse_OSA = np.squeeze(OSA[:, :, 1])
                    OSA_UV = res[:, 1, :].reshape((m, n, 2))
                    OSA_VIS = res[:, 2, :].reshape((m, n, 2))
                    OSA_NIR = res[:, 3, :].reshape((m, n, 2))

                    logging.info("[CMIP6_light] UV {} VIS {} NIR {}".format(np.nanmean(OSA_UV[:, :, 0]),
                                                                            np.nanmean(OSA_VIS[:, :, 0]),
                                                                            np.nanmean(OSA_NIR[:, :, 0])))
                    if scenario == "normal":
                        direct_OSA = np.where(direct_OSA < 0.08, 0.06, direct_OSA)
                    # Calculate radiation calculation uses the direct_OSA to calculate the diffuse radiation
                    rads = self.calculate_radiation(hour_of_day, model_object, clt, direct_OSA, lat, m, n)

                    tas = direct_OSA * 0.0 + 2.0
                    # Initialize the ccsm3 object for calculating effect of snow and ice. We want to calculate the
                    # albedo for visible and for UV light in two steps.
                    albedo_druv = OSA_UV[:, :, 0]
                    albedo_dfuv = OSA_UV[:, :, 1]
                    albedo_drvis = OSA_VIS[:, :, 0]
                    albedo_dfvis = OSA_VIS[:, :, 1]

                    self.cmip6_ccsm3 = CMIP6_ccsm3.CMIP6_CCSM3()
                    direct_sw = rads[:, :, 0]
                    diffuse_sw = rads[:, :, 1]

                    # Calculate shortwave radiation into the ocean accounting for the effect of snow and ice to the direct
                    # and diffuse albedos and for attenutation (no scattering). The final product adds diffuse and direct
                    # light for the spectrum in question (vis or uv).
                    direct_sw_albedo_ice_snow_corrected_vis = self.cmip6_ccsm3.compute_surface_solar_for_specific_wavelength_band(
                        albedo_drvis,
                        albedo_dfvis,
                        direct_sw*np.sum(self.config.fractions_shortwave_vis),
                        diffuse_sw*np.sum(self.config.fractions_shortwave_vis),
                        chl * chl_scale,
                        sisnthick,
                        sithick,
                        siconc,
                        sisnconc,
                        tas,
                        spectrum="vis")
                    print("self.config.fractions_shortwave_uv",self.config.fractions_shortwave_uv)
                    print("Unscaled {} scaled {}".format(np.nansum(direct_sw), np.nansum(direct_sw*np.nansum(self.config.fractions_shortwave_uv))))
                    direct_sw_albedo_ice_snow_corrected_uv = self.cmip6_ccsm3.compute_surface_solar_for_specific_wavelength_band(
                        albedo_druv,
                        albedo_dfuv,
                        direct_sw*np.sum(self.config.fractions_shortwave_uv),
                        diffuse_sw*np.sum(self.config.fractions_shortwave_uv),
                        chl * chl_scale,
                        sisnthick,
                        sithick,
                        siconc,
                        sisnconc,
                        tas,
                        spectrum="uv")

                    dr_uv = self.calculate_the_effect_of_ozone(direct_sw_albedo_ice_snow_corrected_uv)

                    uvi = self.cmip6_ccsm3.calculate_uvi(dr_uv)
                    print("UVI mean: {} range: {} to {}".format(np.nanmean(uvi),np.nanmin(uvi),np.nanmax(uvi)))

                    plotter = CMIP6_albedo_plot.CMIP6_albedo_plot()
                    plotter.create_plots(lon, lat, model_object,
                                         uvi=uvi,
                                         plotname_postfix="_uvi_{}".format(scenario))

                 #   plotter.create_plots(lon, lat, model_object,
                 #                        direct_sw=direct_sw_albedo_ice_snow_corrected_uv,
                 #                        plotname_postfix="_uv_{}".format(scenario))

                    coords = {'lat': lat[:, 0], 'lon': lon[0, :], 'time': model_object.current_time}
                    if scenario == scenarios[0]:
                        data_array = xr.DataArray(name="irradiance_vis_{}".format(scenario),
                                                  data=direct_sw_albedo_ice_snow_corrected_vis, coords=coords,
                                                  dims=['lat', 'lon'])
                    else:
                        data_array["irradiance_vis_{}".format(scenario)] = (
                            ['lat', 'lon'], direct_sw_albedo_ice_snow_corrected_vis)
                    data_array["irradiance_uv_{}".format(scenario)] = (
                    ['lat', 'lon'], dr_uv)

                    data_list.append(data_array)

        self.save_irradiance_to_netcdf(model_object.name,
                                       model_object.current_member_id,
                                       data_list)

    def save_irradiance_to_netcdf(self, model_name, member_id, data_list):
        out = self.config.outdir + "ncfiles/"
        result_file = out + "irradiance_{}_{}_{}-{}.nc".format(model_name,
                                                               member_id,
                                                               self.config.start_date,
                                                               self.config.end_date)

        if not os.path.exists(out): os.makedirs(out, exist_ok=True)
        if os.path.exists(result_file): os.remove(result_file)
        logging.info("[CMIP6_light] Wrote results to {}".format(result_file))
        expanded_da = xr.concat(data_list, 'time')
        expanded_da.to_netcdf(result_file, 'w')

    def calculate_light(self):

        io = CMIP6_IO.CMIP6_IO()
        if self.config.use_local_CMIP6_files:
            io.organize_cmip6_netcdf_files_into_datasets(self.config)
        else:
            io.organize_cmip6_datasets(self.config)
        io.print_table_of_models_and_members()

        self.cmip6_models = io.models
        logging.info("[CMIP6_light] Light calculations will involve {} CMIP6 model(s)".format(
            len(self.cmip6_models)))

        for ind, model in enumerate(self.cmip6_models):
            logging.info("[CMIP6_light] {} : {}".format(ind, model.name))
            for member_id in model.member_ids:
                logging.info("[CMIP6_light] Members : {}".format(member_id))

        for model in self.cmip6_models:
            for member_id in model.member_ids:
                model.current_member_id = member_id
                if not self.config.use_local_CMIP6_files:
                    io.extract_dataset_and_save_to_netcdf(model, self.config)
                if self.config.perform_light_calculations:
                    self.perform_light_calculations(model)


def main():
    light = CMIP6_light()
    light.config.setup_logging()
    light.config.setup_parameters()
    logging.info("[CMIP6_config] logging started")
    light.calculate_light()


if __name__ == '__main__':
    np.warnings.filterwarnings('ignore')
    # https://docs.dask.org/en/latest/diagnostics-distributed.html
    from dask.distributed import Client

    dask.config.set(scheduler='processes')

    client = Client()
    status = client.scheduler_info()['services']
    print("Dask started with status at: http://localhost:{}/status".format(status["dashboard"]))
    print(client)

    main()
