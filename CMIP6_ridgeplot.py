from datetime import datetime
import logging
from typing import List

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import cmocean
import matplotlib
import seaborn as sns
import matplotlib as mpl
import pandas as pd
plt.style.use("mpl15")
import cartopy.crs as ccrs

import CMIP6_area_calculations

class CMIP6_ridgeplot:

    @staticmethod
    def convert_to_DataArray(
            var_name: str,
            times: np.ndarray,
            lats: np.ndarray,
            lons: np.ndarray,
            data_array: np.ndarray,
            depths: np.ndarray = None,
    ) -> xr.DataArray:

        coords = {"time": times, "lat": lats, "lon": lons}
        dims = ["time", "lat", "lon"]

        if depths is not None:
            coords["depth"] = depths
            dims.insert(0, "depth")

        return xr.DataArray(name=var_name, data=data_array, coords=coords, dims=dims)


    @classmethod
    def calculate_climatology(cls, ds: xr.Dataset, var_name: str, infile: str):
        lons = ds["lon"].values
        lats = ds["lat"].values
        times = ds["time"].values

        data_array = ds[var_name]

        # Calculate the trend stats
        trend_stats = CMIP6_area_calculations.calc_trend(data_array)

        # Remove the trend difference from the timeseries: the difference is y=ax and not y=ax+b
        y = [
            (data_array[i, :, :].values - (trend_stats.slope.values * i))
            for i in range(len(times))
        ]
        logging.info("[CMIP6_ridgeplot] Finished removing trend for {}".format(infile))

        # Organize the de_trended data into dataset
        de_trended_data = np.zeros((len(times), len(lats), len(lons)))
        for time_index, de_trended_at_timestep in enumerate(y):
            de_trended_data[time_index, :, :] = de_trended_at_timestep[:, :]

        ds_de_trended = cls.convert_to_DataArray(
            var_name, times, lats, lons, de_trended_data
        ).to_dataset()

        # Calculate the climatology and the anomalies from the de-trended dataset
        climatology = ds_de_trended.groupby("time.month").mean("time", keep_attrs=True)
        # std_climatology = ds_de_trended.groupby("time.month").std("time", keep_attrs=True)
        # climatology["std"] = (['time', 'lat', 'lon'], std_climatology[var_name])
        # climatology = climatology.rename({"time": "month"})

        logging.info(
            "[CMIP6_ridgeplot] Finished calculating climatology for {}".format(infile)
        )
        return climatology

    # Define and use a simple function to label the plot in axes coordinates
    @classmethod
    def label(cls, x, color, label):
        ax = plt.gca()
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        # months[int(label) - 1]

        ax.text(
            0,
            0.2,
            label,
            fontweight="regular",
            color="black",
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    @classmethod
    def return_df_climatology(
        cls,
        var_name: str,
        infile: str,
        start_time: datetime,
        end_time: datetime,
        depth_threshold: float,
        ds: xr.Dataset = None,
    ) -> pd.DataFrame:
        """
        Create a Pandas dataframe organized as monthly climatology for a specific period
        of time from a dataset.
         If ds is not Noe, we pass a dataset instead of file to the function. The filename is then ignored and
         we use the dataset instead.
        :param depth_threshold: If provides will filter dataset to include only depths less than depth_threshold
        :param infile: string
        :param start_time: datetime
        :param end_time: datetime
        :param ds: xr.Dataset
        :return: pd.DataFrame

        """
        if ds is None:
            ds = xr.open_dataset(infile).sel(time=slice(start_time, end_time))
        else:
            ds = ds.sel(time=slice(start_time, end_time))
        ds = ds.chunk({"time": -1})
        if depth_threshold is not None:
            if "depth" in ds.variables:
                ds = ds.where(ds.depth < depth_threshold)
            elif "depth_mean" in ds.variables:
                ds = ds.where(ds.depth_mean < depth_threshold)

        clim = cls.calculate_climatology(ds, var_name, infile)
        return clim.to_dataframe()

    @classmethod
    def create_ridgeplot(
        cls, var_name, df: pd.DataFrame, outfile: str, labels: str
    ) -> None:

        logging.info(
            "[CMIP6_ridgeplot] Starting ridgeplot creation for {}".format(var_name)
        )

        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.8)
        pal = sns.cubehelix_palette(12, rot=0.3, light=0.7)

        # Color combinations
        # https://digitalsynopsis.com/design/minimal-web-color-palettes-combination-hex-code/
        color1 = "#FAA50A"
        color2 = "#FA5A0A"
        color3 = "#DC2B50"

        g1 = sns.FacetGrid(
            df, row="month", hue="month", aspect=15, height=0.75, palette=pal
        )

        # Draw the densities in a few steps
        g1.map(
            sns.kdeplot,
            var_name,
            bw_adjust=0.5,
            clip_on=False,
            fill=True,
            alpha=0.7,
            linewidth=1.0,
            color=color1,
        )
        g1.map(sns.kdeplot, var_name, clip_on=False, color=color1, lw=1, bw_adjust=0.5)
        g1.map(
            sns.kdeplot,
            "Proj",
            bw_adjust=0.5,
            clip_on=False,
            fill=True,
            alpha=0.45,
            linewidth=1.0,
            color=color2,
        )
        g1.map(sns.kdeplot, "Proj", clip_on=False, color=color2, lw=1.0, bw_adjust=0.5)
        g1.map(
            sns.kdeplot,
            "Proj2",
            bw_adjust=0.5,
            clip_on=False,
            fill=True,
            alpha=0.45,
            linewidth=1.0,
            color=color3,
        )
        g1.map(sns.kdeplot, "Proj2", clip_on=False, color=color3, lw=1.0, bw_adjust=0.5)

        g1.map(plt.axhline, y=0, lw=1, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        g1.map(cls.label, var_name)
        # Set the subplots to overlap
        g1.fig.subplots_adjust(hspace=-0.25)

        # Remove axes details that don't play well with overlap
        g1.set_titles("")
        g1.set(yticks=[])

        g1.despine(bottom=True, left=True)
        for ax in g1.axes.ravel():
            if ax.is_first_row():
                ax.legend(
                    labels=[labels[0], labels[1], labels[2]],
                    facecolor="white",
                    framealpha=1,
                    loc="upper right",
                )

        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        if os.path.exists(outfile):
            os.remove(outfile)
        print("[CMIP6_plot] Created plot {}".format(outfile))
        plt.savefig(outfile, dpi=300)

        plt.show()

    @classmethod
    def ridgeplot(
        cls,
        var_name,
        infile,
        outfile,
        glorys=False,
        depth_threshold=None,
        ds: xr.Dataset = None,
    ):
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
        if glorys is True:
            start_time = datetime(1993, 1, 1)
            end_time = datetime(2000, 1, 1)

            start_time2 = datetime(2000, 1, 1)
            end_time2 = datetime(2010, 1, 1)

            start_time3 = datetime(2010, 1, 1)
            end_time3 = datetime(2020, 1, 1)
        else:
            start_time = datetime(2020, 1, 1)
            end_time = datetime(2030, 1, 1)

            start_time2 = datetime(2050, 1, 1)
            end_time2 = datetime(2060, 1, 1)

            start_time3 = datetime(2080, 1, 1)
            end_time3 = datetime(2090, 1, 1)

        labels = [
            "{}-{}".format(start_time.year, end_time.year),
            "{}-{}".format(start_time2.year, end_time2.year),
            "{}-{}".format(start_time3.year, end_time3.year),
        ]

        df1 = cls.return_df_climatology(
            var_name,
            infile,
            start_time=start_time,
            end_time=end_time,
            depth_threshold=depth_threshold,
            ds=ds,
        )
        df2 = cls.return_df_climatology(
            var_name,
            infile,
            start_time=start_time2,
            end_time=end_time2,
            depth_threshold=depth_threshold,
            ds=ds,
        )
        df3 = cls.return_df_climatology(
            var_name,
            infile,
            start_time=start_time3,
            end_time=end_time3,
            depth_threshold=depth_threshold,
            ds=ds,
        )

        df1 = df1.reset_index(level="month")
        df2 = df2.reset_index(level="month")
        df3 = df3.reset_index(level="month")
        df2["Proj"] = df2[var_name]
        df3["Proj"] = df3[var_name]

        df = df1
        df["Proj"] = df2["Proj"]
        df["Proj2"] = df3["Proj"]

        cls.create_ridgeplot(var_name, df, outfile, labels)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("[CMIP6_ridgeplot] Initialized logging")

    infile = "../shared/cmip6/downscaling/NewFoundland/ssp585/ensemble/thetao/thetao_ensemble_sd+ba_surface_depth_5_stats_ssp585.nc"
    outfile = "Figures/thetao_ensemble_sd+ba_surface_depth_5_stats_ssp585_300m.png"

    CMIP6_ridgeplot.ridgeplot(
        "thetao_mean", infile, outfile, glorys=False, depth_threshold=300
    )
