import unittest
import CMIP6_ccsm3
import numpy as np
import logging


class MyCMIP6CCSM3(unittest.TestCase):

    def setUp(self) -> None:
        self.lons = np.arange(10, 20, 1)
        self.lats = np.arange(75, 80, 1)
        self.ice_thickness = np.ones(
            (len(self.lats), len(self.lons)))  # np.random.random((len(self.lats), len(self.lons)))
        self.snow_thickness = self.ice_thickness / 10.0

        # Constant ocean albedo of 0.06
        self.ocean_albedo = np.ones(np.shape(self.snow_thickness)) * 0.06
        # slightly warmer than melting so we get the effect of melt ponds
        self.air_temp = np.ones(np.shape(self.snow_thickness)) - 10.0
        self.sea_ice_concentration = np.ones(np.shape(self.snow_thickness))
        self.snow_concentration = self.sea_ice_concentration

        self.direct_nir = np.ones(np.shape(self.snow_thickness)) * 1.0
        self.direct_vis = np.ones(np.shape(self.snow_thickness)) * 10.0
        self.diffuse_nir = np.ones(np.shape(self.snow_thickness)) * 0.5
        self.diffuse_vis = np.ones(np.shape(self.snow_thickness)) * 5

        self.cmip6_ccsm3 = CMIP6_ccsm3.CMIP6_CCSM3(self.ocean_albedo,
                                                   self.ocean_albedo,
                                                   self.air_temp,
                                                   self.ice_thickness,
                                                   self.snow_thickness,
                                                   self.sea_ice_concentration,
                                                   self.snow_concentration)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def test_init_correct(self):
        expected_formatted_name = "ccsm3"
        self.assertEqual(self.cmip6_ccsm3.shortwave, expected_formatted_name)

    def test_albedo_correct_when_sea_ice_concentration_is_zero(self):
        self.cmip6_ccsm3.sea_ice_concentration = np.zeros(np.shape(self.snow_thickness))
        self.cmip6_ccsm3.snow_thickness = np.zeros(np.shape(self.snow_thickness))
        self.cmip6_ccsm3.ice_thickness = np.zeros(np.shape(self.snow_thickness))
        self.cmip6_ccsm3.air_temp = np.ones(np.shape(self.snow_thickness)) * (-20.0)
        alvdfn, alidfn, alvdrn, alidrn = self.cmip6_ccsm3.compute_direct_and_diffuse_albedo_from_snow_and_ice()

        np.testing.assert_almost_equal(alvdrn, self.ocean_albedo, decimal=2)
        np.testing.assert_almost_equal(alvdfn, self.ocean_albedo, decimal=2)
        np.testing.assert_almost_equal(alidrn, self.ocean_albedo, decimal=2)
        np.testing.assert_almost_equal(alidfn, self.ocean_albedo, decimal=2)

    def test_albedo_correct_when_sea_air_temp_is_very_cold_and_full_ice_snow_cover(self):
        # Very thick ice
        self.cmip6_ccsm3.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.cmip6_ccsm3.snow_concentration = self.cmip6_ccsm3.sea_ice_concentration
        # Make sure the effect of pond melting is absolutely none
        self.cmip6_ccsm3.air_temp = np.ones(np.shape(self.snow_thickness)) - 20.0
        alvdfn, alidfn, alvdrn, alidrn = self.cmip6_ccsm3.compute_direct_and_diffuse_albedo_from_snow_and_ice()

        np.testing.assert_almost_equal(alvdfn, np.ones(np.shape(self.snow_thickness)) * 0.96, decimal=2)
        np.testing.assert_almost_equal(alvdfn, np.ones(np.shape(self.snow_thickness)) * 0.96, decimal=2)
        np.testing.assert_almost_equal(alidrn, np.ones(np.shape(self.snow_thickness)) * 0.68, decimal=2)
        np.testing.assert_almost_equal(alidfn, np.ones(np.shape(self.snow_thickness)) * 0.68, decimal=2)

    def test_albedo_correct_when_sea_air_temp_is_very_cold_and_full_ice_no_snow_cover(self):
        # Very thick ice
        self.cmip6_ccsm3.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.cmip6_ccsm3.snow_concentration = self.cmip6_ccsm3.sea_ice_concentration*0.0
        self.cmip6_ccsm3.snow_thickness = self.cmip6_ccsm3.snow_concentration

        # Make sure the effect of pond melting is absolutely none
        self.cmip6_ccsm3.air_temp = np.ones(np.shape(self.snow_thickness)) - 20.0
        alvdfn, alidfn, alvdrn, alidrn = self.cmip6_ccsm3.compute_direct_and_diffuse_albedo_from_snow_and_ice()

        np.testing.assert_almost_equal(alvdfn, np.ones(np.shape(self.snow_thickness)) * 0.73, decimal=2)
        np.testing.assert_almost_equal(alvdfn, np.ones(np.shape(self.snow_thickness)) * 0.73, decimal=2)
        np.testing.assert_almost_equal(alidrn, np.ones(np.shape(self.snow_thickness)) * 0.33, decimal=2)
        np.testing.assert_almost_equal(alidfn, np.ones(np.shape(self.snow_thickness)) * 0.33, decimal=2)

    def test_albedo_correct_when_sea_air_temp_is_very_warm_and_full_ice_no_snow_cover(self):
        # Very thick ice, but very warm weather creating melt ponds that drastically decreases
        # the albedo from sea-ice (to 0.06)
        self.cmip6_ccsm3.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.cmip6_ccsm3.snow_concentration = self.cmip6_ccsm3.sea_ice_concentration * 0.0
        self.cmip6_ccsm3.snow_thickness = self.cmip6_ccsm3.snow_concentration

        # Make sure the effect of pond melting is absolutely none
        self.cmip6_ccsm3.air_temp = np.ones(np.shape(self.snow_thickness)) + 20.0
        alvdfn, alidfn, alvdrn, alidrn = self.cmip6_ccsm3.compute_direct_and_diffuse_albedo_from_snow_and_ice()

        np.testing.assert_almost_equal(alvdfn, np.ones(np.shape(self.snow_thickness)) * 0.06, decimal=2)
        np.testing.assert_almost_equal(alvdfn, np.ones(np.shape(self.snow_thickness)) * 0.06, decimal=2)
        np.testing.assert_almost_equal(alidrn, np.ones(np.shape(self.snow_thickness)) * 0.06, decimal=2)
        np.testing.assert_almost_equal(alidfn, np.ones(np.shape(self.snow_thickness)) * 0.06, decimal=2)

    def test_calculate_diffuse_albedo_per_grid_point_all_snow(self):
        # Thick ice, full area cover, full snow cover
        self.cmip6_ccsm3.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.cmip6_ccsm3.sea_ice_concentration = np.ones(np.shape(self.sea_ice_concentration))
        self.cmip6_ccsm3.snow_concentration = self.cmip6_ccsm3.sea_ice_concentration
        self.cmip6_ccsm3.snow_thickness = self.cmip6_ccsm3.snow_concentration

        albedo = self.cmip6_ccsm3.calculate_diffuse_albedo_per_grid_point(self.cmip6_ccsm3.snow_concentration,
                                                                          self.cmip6_ccsm3.sea_ice_concentration)

        np.testing.assert_almost_equal(albedo, np.ones(np.shape(self.snow_thickness)) * 0.96, decimal=2)

    def test_calculate_diffuse_albedo_per_grid_point_all_ice_no_snow(self):
        # Thick ice, full area cover, full snow cover
        self.cmip6_ccsm3.ice_thickness = np.ones(np.shape(self.snow_thickness)) + 10
        self.cmip6_ccsm3.sea_ice_concentration = np.ones(np.shape(self.sea_ice_concentration))
        self.cmip6_ccsm3.snow_concentration = self.cmip6_ccsm3.sea_ice_concentration*0.0
        self.cmip6_ccsm3.snow_thickness = self.cmip6_ccsm3.snow_concentration

        albedo = self.cmip6_ccsm3.calculate_diffuse_albedo_per_grid_point(self.cmip6_ccsm3.snow_concentration,
                                                                          self.cmip6_ccsm3.sea_ice_concentration)

        np.testing.assert_almost_equal(albedo, np.ones(np.shape(self.snow_thickness)) * 0.73, decimal=2)

    def test_calculate_diffuse_albedo_per_grid_point_all_ice_except_one_grid_cell(self):
        # Thick ice, full area cover, full snow cover with the exception
        # of one grid cell that should return normal ocean albedo (0.06)
        self.cmip6_ccsm3.ice_thickness = np.ones(np.shape(self.snow_thickness))
        self.cmip6_ccsm3.sea_ice_concentration = np.ones(np.shape(self.sea_ice_concentration))
        self.cmip6_ccsm3.snow_concentration = self.cmip6_ccsm3.sea_ice_concentration * 0.0
        self.cmip6_ccsm3.snow_thickness = self.cmip6_ccsm3.snow_concentration

        self.cmip6_ccsm3.snow_thickness[0,0]=0
        self.cmip6_ccsm3.snow_concentration[0,0]=0
        self.cmip6_ccsm3.ice_thickness[0,0]=0
        self.cmip6_ccsm3.sea_ice_concentration[0,0]=0

        albedo = self.cmip6_ccsm3.calculate_diffuse_albedo_per_grid_point(self.cmip6_ccsm3.snow_concentration,
                                                                          self.cmip6_ccsm3.sea_ice_concentration)

        result_should_be=np.ones(np.shape(self.snow_thickness)) * 0.73
        result_should_be[0,0]=0.06
        np.testing.assert_almost_equal(albedo, result_should_be, decimal=2)

    def test_shortwave_calculations(self):
        self.cmip6_ccsm3.compute_direct_and_diffuse_albedo_from_snow_and_ice()


if __name__ == '__main__':
    unittest.main()
