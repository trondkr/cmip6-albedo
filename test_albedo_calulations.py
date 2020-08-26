
import unittest
import CMIP6_light
import CMIP6_config
import CMIP6_model
import CMIP6_IO
from datetime import datetime
import numpy as np
import xarray as xr

# Tests to ensure that the method for calulating albedo used by pvlib for calculating diffuse light
# handles correct variations of concentrations of seaice and snow.
# If the seaice concentrations is zero the open water will give albedo=0.06 regardless of what sisnconc is.

class TestCMIP6_light(unittest.TestCase):

    def setUp(self):
        self.cmip6 = CMIP6_light.CMIP6_light()
        self.cmip6_model=CMIP6_model.CMIP6_MODEL(name="test")
        self.cmip6_IO = CMIP6_IO.CMIP6_IO()
        self.query_string = "source_id=='ACCESS-ESM1-5'and table_id=='Amon' and grid_label=='gn' \
                and experiment_id=='historical' and variable_id=='uas'"
        self.sisnconc = np.array([[0, 0.5], [0, 0.5]])
        self.siconc = np.array([[0, 0.5], [0, 0.75]])


    def test_calculate_diffuse_albedo_gives_correct_result(self):

        result_should_be = np.array([[0.06, 0.1925], [0.06, 0.3525]])

        albedo = self.cmip6.calculate_diffuse_albedo_per_grid_point(sisnconc=self.sisnconc,
                                                                    siconc=self.siconc)
        np.testing.assert_almost_equal(result_should_be, albedo)

    def test_calculate_diffuse_albedo_gives_correct_result_over_seawater(self):

        albedo = self.cmip6.calculate_diffuse_albedo_per_grid_point(sisnconc=self.sisnconc,
                                                                    siconc=self.siconc)
        np.testing.assert_almost_equal(0.06, albedo[0, 0])
        np.testing.assert_almost_equal(0.06, albedo[1, 0])

    def test_calculate_diffuse_albedo_gives_correct_result_over_sea_ice(self):

        self.siconc = np.random.randint(2, size=10)
        self.sisnconc = np.random.randint(2, size=10)
        albedo = self.cmip6.calculate_diffuse_albedo_per_grid_point(sisnconc=self.sisnconc,
                                                                    siconc=self.siconc)
        np.testing.assert_almost_equal(albedo[self.siconc == 0], 0.06)

    def test_calculate_diffuse_albedo_gives_correct_result_over_sea_no_seaice(self):

        self.siconc = np.random.randint(2, size=10) * 0
        self.sisnconc = np.random.randint(2, size=10) * 0
        albedo = self.cmip6.calculate_diffuse_albedo_per_grid_point(sisnconc=self.sisnconc,
                                                                    siconc=self.siconc)
        np.testing.assert_allclose(albedo, 0.06)

