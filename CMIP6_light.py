import datetime
import logging
import os
from typing import List, Any

# Computational modules
import cftime
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

    def cloud_opacity_factor(self, I_diff_clouds, I_dir_clouds, I_ghi_clouds, spectra):
        # First we calculate the rho fraction based on campbell_norman irradiance
        # with clouds converted to POA irradiance. In the paper these
        # values are obtained from observations. The equations used for calculating cloud opacity factor
        # to scale the clear sky spectral estimates using spectrl2. Results can be compared with sun calculator:
        # https://www2.pvlighthouse.com.au/calculators/solar%20spectrum%20calculator/solar%20spectrum%20calculator.aspx
        #
        # Ref: Marco Ernst, Hendrik Holst, Matthias Winter, Pietro P. Altermatt,
        # SunCalculator: A program to calculate the angular and spectral distribution of direct and diffuse solar radiation,
        # Solar Energy Materials and Solar Cells, Volume 157, 2016, Pages 913-922,

        rho = I_diff_clouds / I_ghi_clouds

        I_diff_s = np.trapz(y=spectra['poa_sky_diffuse'][:, 0], x=spectra['wavelength'])
        I_dir_s = np.trapz(y=spectra['poa_direct'][:, 0], x=spectra['wavelength'])
        I_glob_s = np.trapz(y=spectra['poa_global'][:, 0], x=spectra['wavelength'])

        rho_spectra = I_diff_s / I_glob_s

        N_rho = (rho - rho_spectra) / (1 - rho_spectra)

        # Direct light. Equation 6 Ernst et al. 2016
        F_diff_s = spectra['poa_sky_diffuse'][:, :]
        F_dir_s = spectra['poa_direct'][:, :]

        F_dir = (F_dir_s / I_dir_s) * I_dir_clouds

        # Diffuse light scaling factor. Equation 7 Ernst et al. 2016
        s_diff = (1 - N_rho) * (F_diff_s / I_diff_s) + N_rho * ((F_dir_s + F_diff_s) / I_glob_s)

        # Equation 8 Ernst et al. 2016
        F_diff = s_diff * I_diff_clouds

        return F_dir, F_diff

    def radiation(self, cloud_covers, latitude, ctime, system, albedo, ozone) -> np.array:
        """Returns an array of calculated diffuse and direct light for each longitude index 
        around the globe (fixed latitude). Output has the shape:  [len(wavelengths), 361, 3] and 
        the indexes refer to:
        0: len(wavelengths)
        1: longitudes global
        2: [direct light, diffuse light, zenith]
        Args:
            cloud_covers (np.array): numpy array of cloudcover for longitude band (fixed latitude)
            latitude (float): latitude
            ctime (pd.DatetimeIndex): datetime for when light will be calculated
            system (json): Return from setup_pv_system
            albedo (np.array): numpy array of albedo for longitude band (fixed latitude)
            ozone (np.array): numpy array of ozone for longitude band (fixed latitude)

        Returns:
            np.array: array containing diffuse and direct light, and zenith for each wavelength and longitude
            but remember units are W/m2/nm so you have to integrate across wavelengths to get irradiance
        """
        wavelengths = np.arange(200, 2700, 10)
        results = np.zeros((len(wavelengths), np.shape(cloud_covers)[0], 3))
        altitude = 0.0
        print(np.shape(results))
        # Some calculations are done only on Greenwhich meridian line as they are identical around the globe at the
        # same latitude. For that reason longitude is set to Greenwhich meridian and do not change. The only reason
        # to use longitude would be to have the local sun position for given time but since we calculate position at
        # the same time of the day (hour_of_day) and month (month) we can assume its the same across all longitudes,
        # and only change with latitude.

        longitude = 0.0
        # get_solar-position returns a Pandas dataframe with index so we convert the value
        # to numpy after calculating
        solpos = pvlib.solarposition.get_solarposition(ctime, latitude, longitude)

        airmass_relative = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'].to_numpy(),
                                                                 model='kasten1966')
        pressure = pvlib.atmosphere.alt2pres(altitude)
        airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_relative, pressure)

        airmass_abs_array = np.ones((np.shape(cloud_covers)))* airmass_abs
        am_rel_array = np.ones((np.shape(cloud_covers)))* airmass_relative
        apparent_zenith = np.ones((np.shape(cloud_covers)))* solpos['apparent_zenith'].to_numpy()
        zenith = np.ones((np.shape(cloud_covers)))* solpos['zenith'].to_numpy()
        azimuth = np.ones((np.shape(cloud_covers))) * solpos['azimuth'].to_numpy()
        surface_azimuth = np.ones((np.shape(cloud_covers))) * system['surface_azimuth']

        # Always we use zero tilt when working with pvlib and incoming
        # irradiance on a horizontal plane flat on earth
        surface_tilt = np.zeros((np.shape(cloud_covers)))

        # cloud cover in fraction units here. this is used in campbell_norman functions
        transmittance = (1.0 - cloud_covers) * 0.75
        aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, apparent_zenith, azimuth)

        # Fixed atmospheric components used from pvlib example
        water_vapor_content = np.ones((np.shape(cloud_covers))) * 0.5
        tau500 = np.ones((np.shape(cloud_covers))) * 0.1

        # day of year is an int64index array so access first item
        day_of_year = ctime.dayofyear
        day_of_year = np.ones((np.shape(cloud_covers))) * day_of_year[0]

        spectra = pvlib.spectrum.spectrl2(
            apparent_zenith=apparent_zenith,
            aoi=aoi,
            surface_tilt=surface_tilt,
            ground_albedo=albedo,
            surface_pressure=pressure,
            relative_airmass=airmass_relative,
            precipitable_water=water_vapor_content,
            ozone=ozone,
            aerosol_turbidity_500nm=tau500,
            dayofyear=day_of_year)

        irrads_clouds = pvlib.irradiance.campbell_norman(zenith, transmittance)

        # Convert the irradiance to a plane with tilt zero horizontal to the earth. This is done applying tilt=0 to POA
        # calculations using the output from campbell_norman. The POA calculations include calculting sky and ground
        # diffuse light where specific models can be selected (we use default)
        POA_irradiance_clouds = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            dni=irrads_clouds['dni'],
            ghi=irrads_clouds['ghi'],
            dhi=irrads_clouds['dhi'],
            solar_zenith=apparent_zenith,
            solar_azimuth=azimuth)

        # Account for cloud opacity on the spectral radiation
        F_dir, F_diff = self.cloud_opacity_factor(POA_irradiance_clouds['poa_direct'],
                                             POA_irradiance_clouds['poa_diffuse'],
                                             POA_irradiance_clouds['poa_global'],
                                             spectra)
        # Do the linear interpolation
        for lon_index in range(len(F_dir[0, :])):

            interp_fdir = np.interp(wavelengths, spectra["wavelength"], F_dir[:,lon_index])
            interp_fdiff = np.interp(wavelengths, spectra["wavelength"], F_diff[:,lon_index])


        #    print("1",lon_index, np.trapz(y=F_dir[:,lon_index], x=spectra["wavelength"]))
        #    print("2",lon_index,np.trapz(y=interp_fdir, x=wavelengths))

        #    print("3",lon_index, np.trapz(y=F_diff[:, lon_index], x=spectra["wavelength"]))
        #    print("4", lon_index,np.trapz(y=interp_fdiff, x=wavelengths))

        #    print("shape results", np.shape(results), np.shape(interp_fdir))
           
            results[:,lon_index, 0] = np.squeeze(interp_fdir)
            results[:,lon_index, 1] = np.squeeze(interp_fdiff)
            results[:,lon_index, 2] = solpos['zenith']
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
                extracted[key] = model_obj.ds_sets[model_obj.current_member_id][key].isel(time=int(t_index)).to_array()
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

            current_ds = model_obj.ds_sets[model_obj.current_member_id][key].isel(time=int(t_index)).sel(
                y=slice(int(self.config.min_lat), int(self.config.max_lat)),
                x=slice(int(self.config.min_lon), int(self.config.max_lon)))

            if key in ["uas", "vas", "clt", "tas", "chl"]:
                out_amon = re.regrid_variable(key,
                                              current_ds,
                                              ds_out_amon,
                                              interpolation_method=self.config.interp,
                                              use_esmf_v801=self.config.use_esmf_v801).to_dataset()

                out = re.regrid_variable(key, out_amon, ds_out,
                                         interpolation_method=self.config.interp,
                                         use_esmf_v801=self.config.use_esmf_v801)

            else:
                out = re.regrid_variable(key, current_ds,
                                         ds_out,
                                         interpolation_method=self.config.interp,
                                         use_esmf_v801=self.config.use_esmf_v801)
            extracted[key] = out
        return extracted

    def filter_extremes(self, df):
        return np.where(((df < -1000) | (df > 1000)), np.nan, df)

    def values_for_timestep(self, extracted_ds, selected_time):

        lat = np.squeeze(extracted_ds["uas"].lat.values)
        lon = np.squeeze(extracted_ds["uas"].lon.values)
        chl = np.squeeze(extracted_ds["chl"].values)
        sisnconc = np.squeeze(extracted_ds["sisnconc"].values)
        sisnthick = np.squeeze(extracted_ds["sisnthick"].values)
        siconc = np.squeeze(extracted_ds["siconc"].values)
        sithick = np.squeeze(extracted_ds["sithick"].values)
        uas = np.squeeze(extracted_ds["uas"].values)
        vas = np.squeeze(extracted_ds["vas"].values)
        clt = np.squeeze(extracted_ds["clt"].values)
        tas = np.squeeze(extracted_ds["tas"].values - 273.15)
        tas = np.where(tas == -273.15, np.nan, tas)

        clt = self.filter_extremes(clt)
        chl = self.filter_extremes(chl)
        uas = self.filter_extremes(uas)
        vas = self.filter_extremes(vas)
        sisnconc = self.filter_extremes(sisnconc)
        sisnthick = self.filter_extremes(sisnthick)
        siconc = self.filter_extremes(siconc)
        sithick = self.filter_extremes(sithick)
        tas = self.filter_extremes(tas)

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

        print("Max clt {} sisnc {} sic {} tas {} min tas {}".format(np.nanmax(clt),
                                  np.nanmax(sisnconc),
                                  np.nanmax(siconc),
                                  np.nanmin(tas),
                                  np.nanmax(tas)))
      #  assert np.nanmax(clt) <= 1.5, "Clouds needs to be scaled to between 0 and 1"
      #  assert np.nanmax(sisnconc) <= 2.5, "Sea-ice snow concentration needs to be scaled to between 0 and 1"
      #  assert np.nanmax(siconc) <= 2.5, "Sea-ice needs to be scaled to between 0 and 1"
      #  assert np.nanmax(tas) <= 60, "Temperature needs to be in Celsius"
      #  assert np.nanmin(tas) > -60, "Temperature wrongly converted"

        return wind, lat, lon, clt, chl, sisnconc, sisnthick, siconc, sithick, tas, m, n

    def calculate_radiation(self,
                            hour_of_day: int,
                            model_object: CMIP6_MODEL,
                            clt: np.ndarray,
                            ozone: np.ndarray,
                            direct_OSA: np.ndarray,
                            lat: np.ndarray,
                            m: int, n: int) -> np.ndarray:
        logging.info("[CMIP6_light] Running for hour {}".format(hour_of_day))

        ctime, pv_system = self.setup_pv_system(model_object.current_time.month, hour_of_day)

      #  calc_radiation = [
      #      dask.delayed(self.radiation)(clt[j, :], lat[j, 0], ctime, pv_system, direct_OSA[j, :], ozone[j, :]) for j
      #      in range(m)]
        
        wavelengths = np.arange(200, 2700, 10)
        calc_radiation = [self.radiation(clt[j, :], lat[j, 0], ctime, pv_system, direct_OSA[j, :], ozone[j, :]) for j
            in range(m)]

        # https://github.com/dask/dask/issues/5464
        rad = dask.compute(calc_radiation)
        rads = np.squeeze(np.asarray(rad).reshape((m, len(wavelengths), n, 3)))

        # Transpose to get order: wavelengths, lat, lon, elements
        return np.transpose(rads, (1, 0, 2, 3))


    def get_ozone_dataset(self) -> xr.Dataset:
        # Method that reads the total ozone column from input4MPI dataset (Micahela Heggelin)
        # and regrid to consistent 1x1 degree dataset.
        logging.info("[CMIP6_light] Regridding ozone data to standard grid")
        toz_full = xr.open_dataset(self.config.cmip6_netcdf_dir + "/ozone-absorption/TOZ.nc")
        toz_full = toz_full.sel(time=slice(self.config.start_date, self.config.end_date))\
            #.sel(
            #lat=slice(self.config.min_lat, self.config.max_lat),
            #lon=slice(self.config.min_lon, self.config.max_lon))

        re = CMIP6_regrid.CMIP6_regrid()
        ds_out = xe.util.grid_2d(self.config.min_lon,
                                 self.config.max_lon, 1,
                                 self.config.min_lat,
                                 self.config.max_lat, 1)

        toz_ds = re.regrid_variable("TOZ", toz_full,
                                    ds_out,
                                    interpolation_method=self.config.interp,
                                    use_esmf_v801=self.config.use_esmf_v801).to_dataset()
        print(toz_ds)
       # toz_ds.to_netcdf("test_toz.nc")
        return toz_ds

    def convert_dobson_units_to_atm_cm(self, ozone):
        # One Dobson Unit is the number of molecules of ozone that would be required to create a layer
        # of pure ozone 0.01 millimeters thick at a temperature of 0 degrees Celsius and a pressure of 1 atmosphere
        # (the air pressure at the surface of the Earth). Expressed another way, a column of air with an ozone
        # concentration of 1 Dobson Unit would contain about 2.69x1016 ozone molecules for every
        # square centimeter of area at the base of the column. Over the Earth’s surface, the ozone layer’s
        # average thickness is about 300 Dobson Units or a layer that is 3 millimeters thick.
        #
        # https://ozonewatch.gsfc.nasa.gov/facts/dobson_SH.html
        ozone=np.where(ozone==0,np.nan,ozone)
        assert np.nanmax(ozone) <= 700
        assert np.nanmin(ozone) > 100
        ozone = ozone / 1000.
        assert np.nanmin(ozone) <= 0.7
        assert np.nanmin(ozone) > 0
        return ozone

    def perform_light_calculations(self, model_object):

        times = model_object.ds_sets[model_object.current_member_id]["uas"].time
        data_list = []

        toz_ds = self.get_ozone_dataset()

        for selected_time in range(0, len(times.values)):
            sel_time = times.values[selected_time]
            if isinstance(sel_time, cftime._cftime.DatetimeNoLeap):
                sel_time = datetime.datetime(year=sel_time.year, month=sel_time.month, day=sel_time.day)
            if sel_time.dtype in ["datetime64[ns]"]:
                sel_time = pd.DatetimeIndex([sel_time],
                              dtype='datetime64[ns]', name='datetime', freq=None).to_pydatetime()[0]
              #  print("TIME 3: {} TYPE: {}".format(sel_time.month, sel_time.dtype))

            model_object.current_time = sel_time
            extracted_ds = self.extract_dataset_and_regrid(model_object, selected_time)
            logging.info("[CMIP6_light] Running for timestep {} model {}".format(model_object.current_time,
                                                                                 model_object.name))
            wind, lat, lon, clt, chl, sisnconc, sisnthick, siconc, sithick, tas, m, n = self.values_for_timestep(
                extracted_ds, selected_time)
            ozone = self.convert_dobson_units_to_atm_cm(toz_ds["TOZ"][selected_time, :, :].values)

            print("Ozone {} to {} mean {}".format(np.nanmin(ozone), np.nanmax(ozone), np.nanmean(ozone)))

            for hour_of_day in range(12, 13, 1):
                # Calculate zenith for each grid point
                # Not currently used...
                #  albedo_simple = self.cmip6_ccsm3.calculate_diffuse_albedo_per_grid_point(sisnconc=sisnconc,
                #                                                               siconc=siconc)

                ctime, pv_system = self.setup_pv_system(model_object.current_time.month, hour_of_day)
                calc_zenith = [dask.delayed(self.calculate_zenith)(lat[j, 0], ctime) for j in range(m)]
                zenith = dask.compute(calc_zenith)
                zeniths = np.asarray(zenith).reshape(m)

                scenarios = ["osa"]  # , "no_chl", "no_wind", "normal"]
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

                    logging.info("OSA: min {} mean {} max {}".format(np.nanmin(direct_OSA),
                                                                     np.nanmean(direct_OSA),
                                                                     np.nanmax(direct_OSA)))
                    OSA_UV = res[:, 1, :].reshape((m, n, 2))
                    OSA_VIS = res[:, 2, :].reshape((m, n, 2))

                    if scenario == "normal":
                        direct_OSA = np.where(direct_OSA < 0.08, 0.06, direct_OSA)

                    # Calculate radiation calculation uses the direct_OSA to calculate the diffuse radiation
                    rads = self.calculate_radiation(hour_of_day, model_object, clt, ozone, direct_OSA, lat, m, n)

                    # Initialize the ccsm3 object for calculating effect of snow and ice. We want to calculate the
                    # albedo for visible and for UV light in two steps.
                    albedo_druv = OSA_UV[:, :, 0]
                    albedo_dfuv = OSA_UV[:, :, 1]
                    albedo_drvis = OSA_VIS[:, :, 0]
                    albedo_dfvis = OSA_VIS[:, :, 1]

                    self.cmip6_ccsm3 = CMIP6_ccsm3.CMIP6_CCSM3()
                    direct_sw = rads[:, :, :, 0]
                    diffuse_sw = rads[:, :, :, 1]

                    # Calculate shortwave radiation entering the ocean after accounting for the effect of snow
                    # and ice to the direct and diffuse albedos and for attenutation (no scattering).
                    # The final product adds diffuse and direct
                    # light for the spectrum in question (vis or uv).
                    start_index_visible = len(np.arange(200, 400, 10))
                    end_index_visible = len(np.arange(200, 710, 10))
                    wavelengths = np.arange(200, 2700, 10)
                    direct_sw_albedo_ice_snow_corrected_vis = self.cmip6_ccsm3.compute_surface_solar_for_specific_wavelength_band(
                        albedo_drvis,
                        albedo_dfvis,
                        direct_sw[start_index_visible:end_index_visible,:,:],
                        diffuse_sw[start_index_visible:end_index_visible, :, :],
                        chl * chl_scale,
                        sisnthick,
                        sithick,
                        siconc,
                        sisnconc,
                        tas,
                        lon, lat,
                        model_object,
                        spectrum="vis")

                    start_index_uv = len(np.arange(200, 280, 10))
                    end_index_uv = len(np.arange(200, 390, 10))

                    direct_sw_albedo_ice_snow_corrected_uv = self.cmip6_ccsm3.compute_surface_solar_for_specific_wavelength_band(
                        albedo_druv,
                        albedo_dfuv,
                        direct_sw[start_index_uv:end_index_uv, :, :],
                        diffuse_sw[start_index_uv:end_index_uv, :, :],
                        chl * chl_scale,
                        sisnthick,
                        sithick,
                        siconc,
                        sisnconc,
                        tas,
                        lon, lat,
                        model_object,
                        spectrum="uv")

                    dr_vis = np.squeeze(np.trapz(y=direct_sw_albedo_ice_snow_corrected_vis,
                             x=wavelengths[start_index_visible:end_index_visible], axis=0))
                    dr_uv = np.squeeze(np.trapz(y=direct_sw_albedo_ice_snow_corrected_uv,
                             x=wavelengths[start_index_uv:end_index_uv], axis=0))


                    uvi = self.cmip6_ccsm3.calculate_uvi(direct_sw_albedo_ice_snow_corrected_uv, ozone, wavelengths[start_index_uv:end_index_uv])
                    print("UVI mean: {} range: {} to {}".format(np.nanmean(uvi), np.nanmin(uvi), np.nanmax(uvi)))

                    do_plot=True
                    if do_plot:
                        plotter = CMIP6_albedo_plot.CMIP6_albedo_plot()

                        plotter.create_plots(lon, lat, model_object,
                                                direct_sw=dr_vis,
                                                plotname_postfix="_vis_{}".format(scenario))

                        plotter.create_plots(lon, lat, model_object,
                                             uvi=uvi,
                                             plotname_postfix="_UVI_{}".format(scenario))

                        plotter.create_plots(lon, lat, model_object,
                                             siconc=siconc,
                                             plotname_postfix="_siconc_{}".format(scenario))

                        plotter.create_plots(lon, lat, model_object,
                                             sithick=sithick,
                                             plotname_postfix="_sithick_{}".format(scenario))

                        plotter.create_plots(lon, lat, model_object,
                                             sithick=sithick,
                                             plotname_postfix="_sithick_{}".format(scenario))

                        plotter.create_plots(lon, lat, model_object,
                                             chl=chl,
                                             plotname_postfix="_chl_{}".format(scenario))

                        plotter.create_plots(lon, lat, model_object,
                                             clt=clt,
                                             plotname_postfix="_clt_{}".format(scenario))

                    coords = {'lat': lat[:, 0], 'lon': lon[0, :], 'time': model_object.current_time}
                    if selected_time == 0:
                        data_array = xr.DataArray(name="irradiance_scenario_{}".format(scenario),
                                                  data=dr_vis, coords=coords,
                                                  dims=['lat', 'lon'])

                    data_array["PAR"] = (
                        ['lat', 'lon'], dr_vis)
                    data_array["UV"] = (
                        ['lat', 'lon'], dr_uv)
                    data_array["UVI"] = (
                        ['lat', 'lon'],uvi)
               #     data_array["chl"] = (
               #         ['lat', 'lon'], chl)
               #     data_array["siconc"] = (
               #         ['lat', 'lon'], siconc)
               #     data_array["sithick"] = (
               #         ['lat', 'lon'], sithick)
               #     data_array["clt"] = (
               #         ['lat', 'lon'], clt)
                #    data_array["clt"] = (
                #        ['lat', 'lon'], clt)

                    data_list.append(data_array)

        self.save_irradiance_to_netcdf(model_object.name,
                                       model_object.current_member_id,
                                       data_list, scenario)

    def save_irradiance_to_netcdf(self, model_name, member_id, data_list, scenario):
        out = self.config.outdir + "ncfiles/"
        result_file = out + "Light_{}_{}_{}-{}_scenario_{}_{}.nc".format(model_name,
                                                               member_id,
                                                               self.config.start_date,
                                                               self.config.end_date,
                                                                           scenario,
                                                                      self.config.current_experiment_id)

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

                # Save datafiles to do calculations locally
                if self.config.write_CMIP6_to_file:
                    io.extract_dataset_and_save_to_netcdf(model, self.config)
                if self.config.perform_light_calculations:
                    self.perform_light_calculations(model)


def main():
    light = CMIP6_light()
    light.config.setup_logging()
    light.config.setup_parameters()
    logging.info("[CMIP6_config] logging started")
    for light.config.current_experiment_id in light.config.experiment_ids:
        light.calculate_light()



if __name__ == '__main__':
    np.warnings.filterwarnings('ignore')
    # https://docs.dask.org/en/latest/diagnostics-distributed.html
    # https://docs.dask.org/en/latest/setup/single-distributed.html
    from dask.distributed import Client

  #  os.environ['NUMEXPR_MAX_THREADS'] = '16'
    dask.config.set(scheduler='processes')
    #dask.config.set({'array.slicing.split_large_chunks': True})

    with Client(n_workers=20, threads_per_worker=10, processes=True, memory_limit='60GB') as client:
        status = client.scheduler_info()['services']
        assert client.status == "running"
        main()
    client.close()
    assert client.status == "closed"

    logging.info("[CMIP6_light] Execution of downscaling completed")

