import datetime
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


class CMIP6_light:

    def __init__(self):

        self.config = CMIP6_config.Config_albedo()
        self.cmip6_models: List[Any] = []

    def radiation(self, cloud_covers, latitude, month, hour_of_day):
        results = np.zeros((np.shape(cloud_covers)[0], 3))
        offset = 0  # int(lon_180/15.)
        when = [datetime.datetime(2006, month, 15, hour_of_day, 0, 0,
                                  tzinfo=datetime.timezone(datetime.timedelta(hours=offset)))]
        time = pd.DatetimeIndex(when)
        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

        module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
        inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        system = {'module': module, 'inverter': inverter,
                  'surface_azimuth': 180}
        altitude = 0.0
        system['surface_tilt'] = latitude

        # Some calculations are done only on greenwhich meridian line as they are identical around the globe
        # at the same latitude. For that reason longitude is set to greenwhich meridian and do not change. The only reason
        # to use longitude would be to have the local sun position for given time but since we calculate position at the same
        # time of the day (hour_of_day) and month (month) we can assume its the same across all longitudes,
        # and only change with latitude.

        longitude = 0.0
        solpos = pvlib.solarposition.get_solarposition(time, latitude, longitude)

        dni_extra = pvlib.irradiance.get_extra_radiation(time)
        airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])

        pressure = pvlib.atmosphere.alt2pres(altitude)
        am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
        tl = pvlib.clearsky.lookup_linke_turbidity(time, latitude, longitude)

        for cloud_index, cloud_cover in enumerate(cloud_covers):
            # cloud cover in percentage units here
            transmittance = ((100.0 - cloud_cover) / 100.0) * 0.75
            # irrads is a DataFrame containing ghi, dni, dhi
            irrads = pvlib.irradiance.liujordan(solpos['apparent_zenith'], transmittance, am_abs)

            aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'],
                                       solpos['apparent_zenith'], solpos['azimuth'])

            total_irrad = pvlib.irradiance.get_total_irradiance(system['surface_tilt'],
                                                                system['surface_azimuth'],
                                                                solpos['apparent_zenith'],
                                                                solpos['azimuth'],
                                                                irrads['dni'], irrads['ghi'], irrads['dhi'],
                                                                dni_extra=dni_extra,
                                                                model='haydavies')

            results[cloud_index, 0] = total_irrad['poa_direct']
            results[cloud_index, 1] = total_irrad['poa_diffuse']
            results[cloud_index, 2] = solpos['zenith']

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

    def extract_dataset_and_regrid(self, model_obj,t_index,
                                   min_lat: float = None,
                                   max_lat: float = None,
                                   min_lon: float = None,
                                   max_lon: float = None):
        extracted: dict = {}

        ds_out_amon = xe.util.grid_2d(min_lon, max_lon, 2, min_lat, max_lat, 2)
        ds_out = xe.util.grid_2d(min_lon, max_lon, 1, min_lat, max_lat, 1)

        re = CMIP6_regrid.CMIP6_regrid()
        for key in model_obj.ds_sets[model_obj.current_member_id].keys():
            print(model_obj.ds_sets[model_obj.current_member_id][key],key)
            current_ds = model_obj.ds_sets[model_obj.current_member_id][key].isel(time=t_index).sel(y=slice(min_lat, max_lat),
                                                                         x=slice(min_lon, max_lon))

            if key in ["chl", "sithick", "siconc", "sisnthick", "sisnconc"]:
              #  ds_trans = current_ds.chunk({'time': -1}).transpose('bnds', 'vertex', 'y', 'x')
                ds_trans = current_ds.transpose('bnds', 'vertex', 'y', 'x')
            else:
               # ds_trans = current_ds.chunk({'time': -1}).transpose('bnds', 'y', 'x')
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

            #   outfile = "{}_{}_{}.nc".format(key,model_obj.name,model_obj.current_member_id)
            #   if os.path.exists(outfile): os.remove(outfile)
            #  out.to_dataset().to_netcdf(outfile)
            print("[CMIP6_light] wrote variable {} to file".format(key))

            extracted[key] = out
        return extracted

    def values_for_timestep(self, extracted_ds, selected_time):

        lat = extracted_ds["uas"].lat.values
        lon = extracted_ds["uas"].lon.values
        clt = extracted_ds["clt"].values
        chl = extracted_ds["chl"].values
        sisnconc = extracted_ds["sisnconc"].values
        sisnthick = extracted_ds["sisnthick"].values
        siconc = extracted_ds["siconc"].values
        sithick = extracted_ds["sithick"].values
        uas = extracted_ds["uas"].values
        vas = extracted_ds["vas"].values

        # Calculate scalar wind and organize the data arrays to be used for  given time-step (month-year)
        wind = np.sqrt(uas ** 2 + vas ** 2)
        m = len(wind[:, 0])
        n = len(wind[0, :])
        return wind, lat, lon, clt, chl, sisnconc, sisnthick, siconc, sithick, m, n

    def perform_light_calculations(self, model_object):
        startdate = datetime.datetime.now()

        times = model_object.ds_sets[model_object.current_member_id]["uas"].time
        data_list = []

        for selected_time in range(0, len(times)):
            model_object.current_time = pd.to_datetime(times[selected_time].values)

            extracted_ds = self.extract_dataset_and_regrid(model_object,selected_time,
                                                           min_lat=30,
                                                           max_lat=90,
                                                           min_lon=0,
                                                           max_lon=360)


            print("[CMIP6_light] Running for timestep {} model {}".format(model_object.current_time,
                                                                          model_object.name))

            wind, lat, lon, clt, chl, sisnconc, sisnthick, siconc, sithick, m, n = self.values_for_timestep(
                extracted_ds, selected_time)

            for hour_of_day in range(12, 13, 1):
                print("[CMIP6_light] Running for hour {}".format(hour_of_day))

                calc_radiation = [
                    dask.delayed(self.radiation)(clt[j, :], lat[j, 0], model_object.current_time.month, hour_of_day) for
                    j in range(m)]
                # https://github.com/dask/dask/issues/5464
                rad = dask.compute(calc_radiation, scheduler='processes')
                rads = np.asarray(rad).reshape((m, n, 3))

                print("[CMIP6_light] Time to finish with radiation{}".format(datetime.datetime.now() - startdate))

                zr = [CMIP6_albedo_utils.calculate_OSA(rads[i, j, 2], wind[i, j], chl[i, j],
                                                       self.config.wavelengths,
                                                       self.config.refractive_indexes,
                                                       self.config.alpha_chl,
                                                       self.config.alpha_w,
                                                       self.config.beta_w,
                                                       self.config.alpha_wc,
                                                       self.config.solar_energy)
                      for i in range(m)
                      for j in range(n)]

                OSA = np.asarray(dask.compute(zr)).reshape((m, n, 2))

                irradiance_water = (rads[:, :, 0] * OSA[:, :, 0] + rads[:, :, 1] * OSA[:, :, 1]) / (
                        OSA[:, :, 0] + OSA[:, :, 1])

                print("[CMIP6_light] Time to finish {} with mean OSA {}".format(datetime.datetime.now() - startdate,
                                                                                np.nanmean(irradiance_water)))

                # Write to file

            #    plotter = CMIP6_albedo_plot.CMIP6_albedo_plot()
            #    plotter.create_plots(sisnconc, sisnthick, sithick, siconc, clt, chl, rads,
            #                         irradiance_water, wind, OSA,
            #                         lon, lat, model_object)

                coords = {'lat': lat[:, 0], 'lon': lon[0, :], 'time': model_object.current_time}
                data_array = xr.DataArray(name="irradiance", data=irradiance_water, coords=coords,
                                          dims=['lat', 'lon'])
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

        if not os.path.exists(out): os.mkdir(out)
        if os.path.exists(result_file): os.remove(result_file)
        expanded_da = xr.concat(data_list, 'time')
        expanded_da.to_netcdf(result_file, 'w')

    def calculate_light(self):

        io = CMIP6_IO.CMIP6_IO()
        io.organize_cmip6_datasets(self.config)
        self.cmip6_models = io.models
        print("[CMIP6_light] Light calculations will involve {} CMIP6 models".format(len(self.cmip6_models)))

        for model in self.cmip6_models:
            print("[CMIP6_light] Model {}".format(model.description))

            for member_id in model.member_ids:
                print("[CMIP6_light] Preparing light calculations for member {}".format(member_id))
                model.current_member_id = member_id
                # Interpolate all variables for all time-steps in one go

                self.perform_light_calculations(model)


def main():
    light = CMIP6_light()
    light.config.setup_parameters()
    light.calculate_light()


if __name__ == '__main__':
#scheduler-address:8786)
    with Client() as client:
   # print(client)
 #   cluster = LocalCluster(host='127.0.0.1', scheduler_port=8786, dashboard_address='127.0.0.1:8787', processes=True,
 #                          local_directory='../oceanography/tmp')
 #   client = Client(cluster)

    # gateway = Gateway()
    # cluster = gateway.new_cluster()
    # cluster.adapt(minimum=1, maximum=50)
    # client = Client(cluster)

        main()
