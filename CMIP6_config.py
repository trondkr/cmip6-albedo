import datetime
import sys

import cftime
import pandas as pd
import gcsfs
import numpy as np
import logging


class Config_albedo():

    def __init__(self):
        logging.info("[CMIP6_config] Defining the config file for the calculations")
        self.fs = gcsfs.GCSFileSystem(token="anon", access="read_only")
        self.grid_labels = ["gn"]  # Can be gr=grid rotated, or gn=grid native
        self.member_ids = ["r10i1p1f1", "r4i1p1f1", "r10i1p2f1", "r3i1p2f1", "r2i1p1f2", "r4i1p1f2",
                           "r2i1p1f1"]  # ,"r1i1p1f1","r1i1p1f1","r1i1p1f2"]
        n = 2
        self.member_ids = ["r{}i{}p{}f{}".format(str(i + 1), str(ii + 1), str(iii + 1), str(iv + 1)) for i in range(n)
                           for ii in range(n) for iii in range(n) for iv in range(n)]

        self.experiment_ids = ["ssp585","ssp245"]
        self.source_ids = ["CMCC-ESM2","UKESM1-O-LL"] # ["CanESM5-CanOE","UKESM1-O-LL"] #["UKESM1-0-LL","MPI-ESM1-2-LR"] #["MPI-ESM1-2-HR"] #["ACCESS-ESM1-5"] #,"MPI-ESM1-2-HR"] #,"UKESM1-0-LL","MPI-ESM1-2-LR","CanESM5"] #,"MPI-ESM1-2-HR","UKESM1-0-LL"] #,"UKESM1-0-LL","CanESM5"]
        self.variable_ids = ["prw","clt", "uas", "vas", "chl", "sithick", "siconc", "sisnthick", "sisnconc", "tas"]  # ,"toz"]
        self.table_ids = ["Amon","Amon", "Amon", "Amon", "Omon", "SImon", "SImon", "SImon", "SImon","Amon"]
        # ,"AERmon"]  # Amon=atmospheric variables, Omon=Ocean variables, SImon=sea-ice variables

        self.bias_correct_ghi = True
        self.bias_correct_file = "bias_correct/ghi_deltas.nc"

        self.dset_dict = {}
        self.start_date = "1950-01-01"
        self.end_date = "2099-12-16"
        self.clim_start = "1961-01-01"
        self.clim_end = "1990-01-01"
        self.use_esmf_v801 = True
        self.use_local_CMIP6_files = False
        self.write_CMIP6_to_file = True
        self.perform_light_calculations = False

        self.cmip6_netcdf_dir = "../oceanography/cmip6/light"  # /Volumes/DATASETS/cmip6/ACCESS-ESM1-5/" #"../oceanography/cmip6/light/" #"/Volumes/DATASETS/cmip6/"
        self.cmip6_outdir = "../oceanography/cmip6/light"
     #   self.current_experiment_id = None

        # Cut the region of the global data to these longitude and latitudes
        self.min_lat = 0
        self.max_lat = 90
        self.min_lon = 0
        self.max_lon = 360

        # ESMF and Dask related
        self.dask_chunk = 10
        self.interp = 'bilinear'
        self.outdir = "../oceanography/light/"
        self.selected_depth = 0
        self.models = {}
        self.regional_plot_region = np.array([[45, 49], [-126, -120]])

        self.setup_erythema_action_spectrum()

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
      #  logging.info("[CMIP6_config] {}".format(wl.head()))
        start_index_uv = len(np.arange(200, 280, 10))
        end_index_uv = len(np.arange(200, 390, 10))
        start_index_visible = len(np.arange(200, 400, 10))
        end_index_visible = len(np.arange(200, 710, 10))
        start_index_nir = len(np.arange(200, 800, 10))
        end_index_nir = len(np.arange(200, 2500, 10))

        self.fractions_shortwave_uv = self.solar_energy[start_index_uv:end_index_uv]
        self.fractions_shortwave_vis = self.solar_energy[start_index_visible:end_index_visible]
        self.fractions_shortwave_nir = self.solar_energy[start_index_nir:end_index_nir]

        logging.info("[CMIP6_config] Energy fraction UV ({} to {}): {:3.3f}".format(self.wavelengths[start_index_uv],
                                                                                    self.wavelengths[end_index_uv],
                                                                                    np.sum(
                                                                                        self.fractions_shortwave_uv)))

        logging.info(
            "[CMIP6_config] Energy fraction PAR ({} to {}): {:3.3f}".format(self.wavelengths[start_index_visible],
                                                                            self.wavelengths[end_index_visible],
                                                                            np.sum(self.fractions_shortwave_vis)))

      #  logging.info("[CMIP6_config] Energy fraction NIR ({} to {}): {:3.3f}".format(self.wavelengths[start_index_nir],
      #                                                                               self.wavelengths[end_index_nir],
      #                                                                               np.sum(
      #                                                                                   self.fractions_shortwave_nir)))

        # Read in the ice values for how ice absorbs irradiance as a function of wavelength
        ice_wl = pd.read_csv("ice-absorption/sea_ice_absorption_perovich_and_govoni_interpolated.csv", header=0,
                             sep=",", decimal=".")

        self.wavelengths_ice = ice_wl["wavelength"].values
        self.absorption_ice_pg = ice_wl["k_ice_pg"].values

    def setup_erythema_action_spectrum(self):
        # https://www.esrl.noaa.gov/gmd/grad/antuv/docs/version2/doserates.CIE.txt
        # A = 	1		for  250 <= W <= 298
        # A = 	10^(0.094(298- W))	for 298 < W < 328
        # A = 	10^(0.015(139-W-))	for 328 < W < 400
        wavelengths = np.arange(280, 390, 10)
        self.erythema_spectrum = np.zeros(len(wavelengths))

        # https://www.nature.com/articles/s41598-018-36850-x
        for i, wavelength in enumerate(wavelengths):
            if 250 <= wavelength <= 298:
                self.erythema_spectrum[i] = 1.0
            elif 298 <= wavelength <= 328:
                self.erythema_spectrum[i] = 10.0 ** (0.094 * (298 - wavelength))
            elif 328 < wavelength < 400:
                self.erythema_spectrum[i] = 10.0 ** (0.015 * (139 - wavelength))
        logging.info("[CMIP6_config] Calculated erythema action spectrum for wavelengths 280-400 at 10 nm increment")

    def setup_ozone_uv_spectrum(self):
        # Data collected from Figure 4
        # http://html.rhhz.net/qxxb_en/html/20190207.htm#rhhz
        infile = "ozone-absorption/O3_UV_absorption_edited.csv"
        df = pd.read_csv(infile, sep="\t")

        # Get values from dataframe
        o3_wavelength = df["wavelength"].values
        o3_abs = df["o3_absorption"].values

        wavelengths = np.arange(280, 390, 10)

        # Do the linear interpolation
        o3_abs_interp = np.interp(wavelengths, o3_wavelength, o3_abs)

        logging.info("[CMIP6_config] Calculated erythema action spectrum for wavelengths 280-400 at 10 nm increment")
        return o3_abs_interp, wavelengths

    def setup_absorption_chl(self):

        # Data exported from publication Matsuoka et al. 2007 (Table. 3)
        # Data are interpolated to a fixed wavelength grid that fits with the wavelengths of
        # Seferian et al. 2018
        infile = "chl-absorption/Matsuoka2007-chla_wavelength_absorption.csv"
        df = pd.read_csv(infile, sep=" ")

        # Get values from dataframe
        chl_abs_A = df["A"].values
        chl_abs_B = df["B"].values
        chl_abs_wavelength = df["wavelength"].values

        # Interpolate to 10 nm wavelength bands - only visible
        # This is because all other wavelength calculations are done at 10 nm bands.
        # Original Matsuoka et al. 2007 operates at 5 nm bands.
        wavelengths = np.arange(400, 710, 10)

        # Do the linear interpolation
        A_chl_interp = np.interp(wavelengths, chl_abs_wavelength, chl_abs_A)
        B_chl_interp = np.interp(wavelengths, chl_abs_wavelength, chl_abs_B)

        return A_chl_interp, B_chl_interp, wavelengths

    # import matplotlib.pyplot as plt
    # plt.plot(self.wavelengths,self.solar_energy)
    # plt.title("Energy contributions from wavelengths 200-4000")

    # plt.savefig("energy_fractions.png", dpi=150)
