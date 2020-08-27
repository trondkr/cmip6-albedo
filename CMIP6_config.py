import pandas as pd
import gcsfs
import numpy as np

class Config_albedo():

    def __init__(self):
        print("[CMIP6_config] Defining the config file for the calculations")
        self.df = pd.read_csv("https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv")
        self.fs = gcsfs.GCSFileSystem(token="anon", access="read_only")
        self.grid_labels = ["gn"]  # Can be gr=grid rotated, or gn=grid native
        self.member_ids = ["r1i1p1f1"]  #
        self.experiment_ids = ["ssp585"]  # 'abrupt-4xCO2',
        self.source_ids = ["ACCESS-ESM1-5"] #["ACCESS-ESM1-5"]  # ["CanESM5"] #"MPI-ESM1-2-LR"]
        self.variable_ids = ["uas", "vas", "chl", "clt", "sithick", "siconc", "sisnthick", "sisnconc"]
        self.table_ids = ["Amon", "Amon", "Omon", "Amon", "SImon", "SImon", "SImon",
                          "SImon"]  # Amon=atmospheric variables, Omon=Ocean variables, SImon=sea-ice variables
        self.variable_ids = ["chl"]
        self.table_ids = ["Omon"]

        self.dset_dict = {}
        self.start_date = "1950-01-01"
        self.end_date = "2099-12-01"
        self.clim_start = "1961-01-01"
        self.clim_end = "1990-01-01"
        self.use_esmf_v801=False
        self.use_local_CMIP6_files = False
        self.generate_local_CMIP6_files=True
        self.perform_light_calculations=False

        # Cut the region of the global data to these longitude and latitudes
        self.min_lat=30
        self.max_lat=90
        self.min_lon=0
        self.max_lon=360

        # ESMF and Dask related
        self.dask_chunk = 30
        self.interp = 'bilinear'
        self.outdir="../oceanography/light/"
        self.selected_depth = 0
        self.models = {}
        self.regional_plot_region = np.array([[45, 49], [-126, -120]])

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
