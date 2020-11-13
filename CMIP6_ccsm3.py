import sys
from typing import Tuple

import numpy as np
import logging
import CMIP6_albedo_plot
import CMIP6_config

# Local files and utility functions
sys.path.append("./subroutines/")


# Class for calculating albedo of sea-ice, snow and snow-ponds
# The albedo and absorbed/transmitted flux parameterizations for
# snow over ice, bare ice and ponded ice.
# Methods applied from CCSM3 online code :
# http://www.cesm.ucar.edu/models/cesm1.2/cesm/cesmBbrowser/html_code/cice/ice_shortwave.F90.html#COMPUTE_ALBEDOS
class CMIP6_CCSM3():

    def __init__(self) -> None:

        self.config = CMIP6_config.Config_albedo()
        self.config.setup_parameters()
        self.chl_abs_A, self.chl_abs_B, self.chl_abs_wavelength = self.config.setup_absorption_chl()

        # Input parameter is ocean_albedo with the same size as the global/full grid (360x180).
        # This could be the ocean albedo assuming no ice and can be the output
        # from the OSA (ocean surface albedo) calculations.
        # In addition, snow and ice parameters needed are:
        # ice_thickness, snow_thickness,sea_ice_concentration
        #
        self.shortwave = 'ccsm3'  # shortwave
        self.albedo_type = 'ccsm3'  # albedo parameterization, 'default'('ccsm3')

    # http://www.cesm.ucar.edu/models/cesm1.2/cesm/cesmBbrowser/html_code/cice/ice_shortwave.F90.html#COMPUTE_ALBEDOS
    def compute_direct_and_diffuse_albedo_from_snow_and_ice(self,
                                                            osa_direct: np.ndarray,
                                                            osa_diffuse: np.ndarray,
                                                            snow_concentration: np.ndarray,
                                                            ice_concentration: np.ndarray,
                                                            snow_thickness: np.ndarray,
                                                            ice_thickness: np.ndarray,
                                                            air_temp: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                                           np.ndarray, np.ndarray]:
        """
        Calculate albedo for each grid point taking into account snow and ice area. Also,
        accounts for changes in albedo due to melting of sea-ice and snow. Based on CICE5 shortwave
        albedo calculations:
        https://github.com/CICE-Consortium/CICE-svn-trunk/blob/7d9cff9d8dbabf6d4947e388a6d98c870c808536/cice/source/ice_shortwave.F90
        Units:
        air_temp # Celsius
        sea_ice_concentration # fraction
        ice_thickness  # meter
        snow_thickness # meter
        snow_concentration  # fraction

        Returns:
        alvdfn = albedo visual diffuse
        alidfn = albedo infrared diffuse
        alvdrn = albedo visual direct
        alidrn = albedo infrared direct
        :return: alvdfn, alidfn, alvdrn, alidrn
        """
        ahmax = 0.5
        dT_mlt = 1.0
        dalb_mlt = -0.075
        dalb_mltv = -0.1
        dalb_mlti = -0.15
        # http://www.cesm.ucar.edu/models/ccsm3.0/csim/UsersGuide/ice_usrdoc/node22.html
        albicev = 0.73  # Visible ice albedo (CCSM3)
        albicei = 0.33  # Near-infrared ice albedo (CCSM3)
        albsnowv = 0.96  # Visible snow albedo (CCSM3)
        albsnowi = 0.68  # Near-infrared snow albedo (CCSM3)
        snowpatch = 0.02  # https://journals.ametsoc.org/jcli/article/26/4/1355/34253/The-Sensitivity-of-the-Arctic-Ocean-Sea-Ice
        puny = 1.0e-11
        Timelt = 0.0,  # melting temp.ice top surface(C)

        # ice albedo - al(albedo)v/i(visual/near-infrared)dr/df(direct/diffuse)ni/ns(ice/snow)
        alvdrni = osa_direct
        alidrni = osa_direct
        alvdfni = osa_diffuse
        alidfni = osa_diffuse

        # snow albedo - al(albedo)v/i(visual/near-infrared)dr/df(direct/diffuse)ni/ns(ice/snow)
        alvdrns = osa_direct
        alidrns = osa_direct
        alvdfns = osa_diffuse
        alidfns = osa_diffuse

        # hi = ice height
        fhtan = np.arctan(ahmax * 4.0)

        # bare ice, thickness dependence
        fh = np.where(np.arctan(ice_thickness * 4.0) / fhtan > 1.0, 1, np.arctan(ice_thickness * 4.0) / fhtan)

        albo_df = osa_diffuse * (1.0 - fh)
        albo_dr = osa_direct * (1.0 - fh)
        alvdfni = albicev * fh + albo_df
        alidfni = albicei * fh + albo_df
        alvdrni = albicev * fh + albo_dr
        alidrni = albicei * fh + albo_dr

        # bare ice, temperature dependence
        dTs = Timelt - air_temp

        fT = np.where((dTs / dT_mlt) - 1.0 > 0, 0, (dTs / dT_mlt) - 1.0)
        alvdfni = alvdfni - dalb_mlt * fT
        alidfni = alidfni - dalb_mlt * fT
        alvdrni = alvdrni - dalb_mlt * fT
        alidrni = alidrni - dalb_mlt * fT

        # avoid negative albedo for thin, bare, melting ice
        alvdfni = np.where(alvdfni > 0, alvdfni, osa_diffuse)
        alidfni = np.where(alidfni > 0, alidfni, osa_diffuse)
        alvdrni = np.where(alvdfni > 0, alvdrni, osa_direct)
        alidrni = np.where(alidrni > 0, alidrni, osa_direct)

        # Effect of snow
        alvdfns = np.where(snow_thickness > puny, albsnowv, alvdfns)
        alidfns = np.where(snow_thickness > puny, albsnowi, alidfns)
        alvdrns = np.where(snow_thickness > puny, albsnowv, alvdrns)
        alidrns = np.where(snow_thickness > puny, albsnowi, alidrns)

        # snow on ice, temperature dependence
        alvdfns = np.where(snow_thickness > puny, alvdfns - dalb_mltv * fT, alvdfns)
        alidfns = np.where(snow_thickness > puny, alidfns - dalb_mlti * fT, alidfns)
        alvdrns = np.where(snow_thickness > puny, alvdrns - dalb_mltv * fT, alvdrns)
        alidrns = np.where(snow_thickness > puny, alidrns - dalb_mlti * fT, alidrns)

        # fractional area of snow cover
        asnow = np.where(snow_concentration > puny, snow_concentration, 0.0)

        # Combine snow and ice albedo. In areas with snow we only use snow albedo while in areas with
        # ice and no snow only ice. This is scaled using asnow.
        alvdfn = alvdfni * (1.0 - asnow) + alvdfns * asnow
        alidfn = alidfni * (1.0 - asnow) + alidfns * asnow
        alvdrn = alvdrni * (1.0 - asnow) + alvdrns * asnow
        alidrn = alidrni * (1.0 - asnow) + alidrns * asnow

        return alvdfn, alidfn, alvdrn, alidrn

    def calc_snow_attenuation(self, dr, snow_thickness: np.ndarray):
        attenuation_snow = 20  # unit : m-1

        total_snow = np.count_nonzero(np.where(snow_thickness > 0))
        per = (total_snow / snow_thickness.size) * 100.
        logging.info("[CMIP6_ccsm3] Number of grid points with snow {}".format(per))
        logging.info("[CMIP6_ccsm3] Mean snow thickness {:3.2f}".format(np.nanmean(snow_thickness)))

        return dr * np.exp(attenuation_snow * (-snow_thickness))

    def calc_ice_attenuation_top10cm(self, dr, ice_thickness: np.ndarray):

        attenuationIceTop10cm = 5

        total_ice = np.count_nonzero(np.where(ice_thickness > 0))
        per = (total_ice / ice_thickness.size) * 100.
        logging.info("[CMIP6_ccsm3] Number of grid points with ice {}".format(per))
        logging.info("[CMIP6_ccsm3] Mean ice thickness {:3.2f}".format(np.nanmean(ice_thickness)))

        return dr * np.exp(attenuationIceTop10cm * (-0.1))

    def calc_ice_attenuation(self, spectrum: str, dr: np.ndarray, ice_thickness: np.ndarray):
        logging.info("[CMIP6_ccsm3] calc_ice_attenuation started")

        if spectrum == "uv":
            start_index = 0
            end_index = len(np.arange(200, 400, 10))
            dr = dr * self.config.fraction_shortwave_to_uv
        elif spectrum == "vis":
            start_index = len(np.arange(200, 400, 10))
            end_index = len(np.arange(200, 700, 10))
            dr = dr * self.config.fraction_shortwave_to_vis
        else:
            raise Exception("[CMIP6_ccsm3] No valid spectrum defined ({})".format(spectrum))

        segments = len(self.config.wavelengths_ice[start_index:end_index])
        logging.info(
            "[CMIP6_ccsm3] {} segments to integrate for effect of wavelength on attenuation in ice".format(segments))

        # Split the total radiation into segments and calculate the effect for individual wavelength bands
        dr_segment = dr / float(segments)
        attenuation = self.config.absorption_ice_pg[start_index:end_index]
        dr_final = np.empty(np.shape(dr))
        for k in attenuation:
            dr_final += dr_segment * np.exp(k * (-ice_thickness))

        total_ice = np.count_nonzero(np.where(ice_thickness > 0))
        per = (total_ice / ice_thickness.size) * 100.
        logging.info("[CMIP6_ccsm3] Sea-ice attenuation ranges from {} to {}".format(np.nanmin(attenuation),
                                                                                     np.nanmax(attenuation)))
        logging.info("[CMIP6_ccsm3] Mean VIS SW {:3.2f} in ice covered cells".format(np.nanmean(dr_final)))
        logging.info("[CMIP6_ccsm3] Percentage of grid point ice cover {}".format(per))
        logging.info("[CMIP6_ccsm3] Mean ice thickness {:3.2f}".format(np.nanmean(ice_thickness)))

        return dr_final

    # absorbed_solar - shortwave radiation absorbed by ice, ocean
    # Compute solar radiation absorbed in ice and penetrating to ocean
    def compute_surface_solar_for_specific_wavelength_band(self,
                                                           osa_albedo_dr: np.ndarray,
                                                           osa_albedo_df: np.ndarray,
                                                           direct_sw: np.ndarray,
                                                           diffuse_sw: np.ndarray,
                                                           chl: np.ndarray,
                                                           snow_thickness: np.ndarray,
                                                           ice_thickness: np.ndarray,
                                                           sea_ice_concentration: np.ndarray,
                                                           snow_concentration: np.ndarray,
                                                           air_temp: np.ndarray,
                                                           spectrum: str = "uv") -> np.ndarray:

        # Before calling this method you need to initialize CMIP6_CCSM3 with the OSA albedo array from the
        # wavelength band of interest:
        # wavelength_band_name:
        # OSA_uv, OSA_vis, OSA_nir, OSA_full
        # For OSA_uv and OSA_vis we just use the output of alvdfn and alvdrn as identical  but with different fraction
        # of energy component in total energy.
        # is_ = ice-snow
        is_albedo_df_vis, is_albedo_df_nir, is_albedo_dr_vis, is_albedo_dr_nir = self.compute_direct_and_diffuse_albedo_from_snow_and_ice(
            osa_albedo_dr,
            osa_albedo_df,
            snow_concentration,
            sea_ice_concentration,
            snow_thickness,
            ice_thickness,
            air_temp)

        # Effect of snow and ice
        # Albedo from snow and ice - direct
        direct_sw_ice = direct_sw * sea_ice_concentration * (1.0 - is_albedo_dr_vis)
        direct_sw_ocean = direct_sw * (1.0 - sea_ice_concentration) * (1.0 - osa_albedo_dr)

        # Albedo from snow and ice - diffuse
        diffuse_sw_ice = diffuse_sw * sea_ice_concentration * (1.0 - is_albedo_df_vis)
        diffuse_sw_ocean = diffuse_sw * (1.0 - sea_ice_concentration) * (1.0 - osa_albedo_df)

        # The amount of shortwave irradiance reaching into the snow, water, ice after albedo corrected
        sw_albedo_corrected = np.nan_to_num(direct_sw_ice) + \
                              np.nan_to_num(direct_sw_ocean) + \
                              np.nan_to_num(diffuse_sw_ice) + \
                              np.nan_to_num(diffuse_sw_ocean)

        #  The effect of snow on attenuation
        sw_albedo_corrected_snow = np.where(snow_thickness > 0,
                                            self.calc_snow_attenuation(sw_albedo_corrected, snow_thickness),
                                            sw_albedo_corrected)

        # The wavelength dependent effect of ice on attenuation
        sw_albedo_corrected_ice = np.where(ice_thickness > 0,
                                           self.calc_ice_attenuation(spectrum, sw_albedo_corrected_snow, ice_thickness),
                                           sw_albedo_corrected_snow)

        if spectrum == "uv":
            return sw_albedo_corrected_ice

        # Account for the chlorophyll abundance and effect on attenuation of visible light
        return self.calculate_chl_attenuated_shortwave(sw_albedo_corrected_ice, chl)

    def calculate_chl_attenuated_shortwave(self, dr: np.ndarray, chl: np.ndarray, depth: float = 0.1):
        """
        Following Matsuoka et al. 2007 as defined in Table 3.

        chlorophyll values in mg/m-3 but need to be converted to mg/m-3
        """
        kg2mg = 1.e6
        print("dr",dr)
        # Divide total incoming irradiance on number of wavelength segments,
        # then iterate the absorption effect for each wavelength and calculate total
        # irradiance absorbed by chl.
        segments = len(self.chl_abs_wavelength)
        dr_chl = dr / segments
        logging.info(
            "[CMIP6_ccsm3] {} segments to integrate for effect of wavelength on attenuation by chl".format(segments))

        # Convert the units of chlorophyll to mgm-3
        chl = chl * kg2mg
        dr_chl_integrated = np.zeros(np.shape(dr))

        # Integrate over all wavelengths and calculate total absorption and
        # return the final light values
        for i_wave, x in enumerate(self.chl_abs_wavelength):
            dr_chl_integrated += np.nan_to_num(dr_chl) * np.exp(
                -depth * self.chl_abs_A[i_wave] * chl ** self.chl_abs_B[i_wave])

        return dr_chl_integrated

    def calculate_albedo_in_mixed_snow_ice_grid_cell(self,sisnconc,siconc,albicev,albsnowv):
        return (1.-siconc)*0.06+siconc * ((1.-sisnconc)*albicev+sisnconc*albsnowv)

    def calculate_diffuse_albedo_per_grid_point(self, sisnconc: np.ndarray,
                                                siconc: np.ndarray) -> np.ndarray:
        """
        Routine for  getting a crude estimate of the albedo based on ocean, snow, and ice values.
        The result is used by pvlib to calculate the  initial diffuse irradiance.
        :param sisnconc:
        :param siconc:
        :return: albedo (preliminary version used for pvlib)
        """
        albicev = 0.73  # Visible ice albedo (CCSM3)
        albsnowv = 0.96  # Visible snow albedo (CCSM3)

        albedo = np.zeros(np.shape(sisnconc)) + 0.06
        ice_alb = np.where(siconc > 0, self.calculate_albedo_in_mixed_snow_ice_grid_cell(sisnconc,siconc,albicev,albsnowv), albedo)
        albedo[~np.isnan(ice_alb)] = ice_alb[~np.isnan(ice_alb)]
        return albedo
