import pandas as pd
import gcsfs
import numpy as np
import logging


class Config_albedo():

    def __init__(self):
        print("[CMIP6_config] Defining the config file for the calculations")
        self.fs = gcsfs.GCSFileSystem(token="anon", access="read_only")
        self.grid_labels = ["gn"]  # Can be gr=grid rotated, or gn=grid native
        self.member_ids = ["r1i1p1f1"]  #
        self.experiment_ids = ["ssp585"]  # 'abrupt-4xCO2',
        self.source_ids = ["ACCESS-ESM1-5"]  # , "MPI-ESM1-2-LR", "MPI-ESM1-2-HR"]  # ["CanESM5"] #"MPI-ESM1-2-LR"]
        self.variable_ids = ["uas", "vas", "chl", "clt", "sithick", "siconc", "sisnthick", "sisnconc"]
        self.table_ids = ["Amon", "Amon", "Omon", "Amon", "SImon", "SImon", "SImon",
                          "SImon"]  # Amon=atmospheric variables, Omon=Ocean variables, SImon=sea-ice variables

        self.dset_dict = {}
        self.start_date = "1950-01-01"
        self.end_date = "1950-12-01"
        self.clim_start = "1961-01-01"
        self.clim_end = "1990-01-01"
        self.use_esmf_v801 = True
        self.use_local_CMIP6_files = True
        self.cmip6_netcdf_dir = "/Volumes/DATASETS/cmip6/"
        self.generate_local_CMIP6_files = False
        self.perform_light_calculations = True
        self.cmip6_outdir = "../oceanography/cmip6/"

        # Cut the region of the global data to these longitude and latitudes
        self.min_lat = 30
        self.max_lat = 90
        self.min_lon = 0
        self.max_lon = 360

        # ESMF and Dask related
        self.dask_chunk = 30
        self.interp = 'bilinear'
        self.outdir = "../oceanography/light/"
        self.selected_depth = 0
        self.models = {}
        self.regional_plot_region = np.array([[45, 49], [-126, -120]])

        self.fraction_shortwave_to_uv = None
        self.fraction_shortwave_to_vis = None
        self.fraction_shortwave_to_nir = None

    def setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def read_cmip6_repository(self):
        self.df = pd.read_csv("https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv")

    def setup_parameters(self):
        wl = pd.read_csv("data/Wavelength/Fresnels_refraction.csv", header=0, sep=";", decimal=",")
        self.wavelengths = wl["λ"].values
        self.refractive_indexes = wl["n(λ)"].values
        self.alpha_chl = wl["a_chl(λ)"].values
        self.alpha_w = wl["a_w(λ)"].values
        self.beta_w = wl["b_w(λ)"].values
        self.alpha_wc = wl["a_wc(λ)"].values
        self.solar_energy = wl["E(λ)"].values
        print("[CMIP6_config] {}".format(wl.head()))
        start_index_uv = 0
        end_index_uv = len(np.arange(200, 400, 10))
        start_index_visible = len(np.arange(200, 400, 10))
        end_index_visible = len(np.arange(200, 700, 10))
        start_index_nir = len(np.arange(200, 800, 10))
        end_index_nir = len(np.arange(200, 2500, 10))

        self.fraction_shortwave_to_uv = np.sum(self.solar_energy[start_index_uv:end_index_uv])
        self.fraction_shortwave_to_vis = np.sum(self.solar_energy[start_index_visible:end_index_visible])
        self.fraction_shortwave_to_nir = np.sum(self.solar_energy[start_index_nir:end_index_nir])

        logging.info("[CMIP6_config] Energy fraction UV ({} to {}): {:3.3f}".format(self.wavelengths[start_index_uv],
                                                                                    self.wavelengths[end_index_uv],
                                                                                    self.fraction_shortwave_to_uv))

        logging.info(
            "[CMIP6_config] Energy fraction PAR ({} to {}): {:3.3f}".format(self.wavelengths[start_index_visible],
                                                                            self.wavelengths[end_index_visible],
                                                                            self.fraction_shortwave_to_vis))

        logging.info("[CMIP6_config] Energy fraction NIR ({} to {}): {:3.3f}".format(self.wavelengths[start_index_nir],
                                                                                     self.wavelengths[end_index_nir],
                                                                                     self.fraction_shortwave_to_nir))

        # Read in the ice values for how ice absorbs irradiance as a function of wavelength
        ice_wl = pd.read_csv("ice-absorption/sea_ice_absorption_perovich_and_govoni_interpolated.csv", header=0,
                             sep=",", decimal=".")

        self.wavelengths_ice = ice_wl["wavelength"].values
        self.absorption_ice_pg = ice_wl["k_ice_pg"].values

    # import matplotlib.pyplot as plt
    # plt.plot(self.wavelengths,self.solar_energy)
    # plt.title("Energy contributions from wavelengths 200-4000")

    # plt.savefig("energy_fractions.png", dpi=150)
