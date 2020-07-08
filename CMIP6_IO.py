import xarray as xr
import cftime
import numpy as np
import pandas as pd
import geopandas as gpd
from cmip6_preprocessing.preprocessing import combined_preprocessing
import CMIP6_model
import CMIP6_light

class CMIP6_IO:

    def __init__(self):
        self.models = []

    def get_and_organize_cmip6_data(self, config):

        for experiment_id in config.experiment_ids:
            for grid_label in config.grid_labels:
                for source_id in config.source_ids:
                    if source_id in config.models.keys():
                        model_object = config.models[source_id]
                    else:
                        model_object = CMIP6_model.CMIP6_MODEL(name=source_id)

                    print("MODEL object {}".format(model_object.name))

                    for member_id in config.member_ids:
                        for variable_id, table_id in zip(config.variable_ids, config.table_ids):

                            # Historical query string
                            query_string = "source_id=='{}'and table_id=='{}' and grid_label=='{}' and experiment_id=='historical' and variable_id=='{}'".format(
                                source_id,
                                table_id,
                                grid_label,
                                variable_id)

                            ds_hist = self.perform_cmip6_query(config, query_string)

                            # Future projection depending on choice in experiment_id
                            query_string = "source_id=='{}'and table_id=='{}' and member_id=='{}' and grid_label=='{}' and experiment_id=='{}' and variable_id=='{}'".format(
                                source_id,
                                table_id,
                                member_id,
                                grid_label,
                                experiment_id,
                                variable_id,
                            )

                            ds_proj = self.perform_cmip6_query(config, query_string)

                            # Make sure datasets exists not empty for both historical and projections before storing in
                            # model object

                            #if ds_hist.zstore.values.size > 0 or ds_proj.zstore.values.size > 0:
                            # Concatenate the historical and projections datasets
                            ds = xr.concat([ds_hist, ds_proj], dim="time")
                            # Remove the duplicate overlapping times (e.g. 2001-2014)
                            _, index = np.unique(ds["time"], return_index=True)
                            ds = ds.isel(time=index)

                            # Extract the time period of interest
                            ds = ds.sel(time=slice(config.start_date, config.end_date))
                            print("{} => Dates extracted range from {} to {}\n".format(source_id,
                                                                                       ds["time"].values[0],
                                                                                       ds["time"].values[-1]))

                            # pass the pre-processing directly
                            dset_processed = combined_preprocessing(ds)
                            if variable_id in ["chl"]:
                                if source_id in ["CESM2", "CESM2-FV2", "CESM2-WACCM-FV2", "CESM2-WACCM"]:
                                    dset_processed = dset_processed.isel(lev_partial=config.selected_depth)
                                else:
                                    dset_processed = dset_processed.isel(lev=config.selected_depth)

                            # Save the dataset for variable_id in the dictionary
                            # Create unique key to hold dataset in dictionary
                            key = "{}_{}_{}_{}_{}".format(variable_id, experiment_id, grid_label, source_id,
                                                                  member_id)

                            model_object.member_ids.append(member_id)
                            model_object.ocean_vars.append(variable_id)
                            model_object.ds_dict[key] = dset_processed
                    self.models.append(model_object)
                    print("Found {} datasets for model {}".format(len(self.models), key))

    def perform_cmip6_query(self, config, query_string):
        df_sub = config.df.query(query_string)
        if df_sub.zstore.values.size == 0:
            return df_sub

        mapper = config.fs.get_mapper(df_sub.zstore.values[-1])
        ds = xr.open_zarr(mapper, consolidated=True)
        # print("Time encoding: {} - {}".format(ds.indexes['time'], ds.indexes['time'].dtype))
        if not ds.indexes["time"].dtype in ["datetime64[ns]", "object"]:

            time_object = ds.indexes['time'].to_datetimeindex()  # pd.DatetimeIndex([ds["time"].values[0]])

            # Convert if necessary
            if time_object[0].year == 1:

                times = ds.indexes['time'].to_datetimeindex()  # pd.DatetimeIndex([ds["time"].values])
                times_plus_2000 = []
                for t in times:
                    times_plus_2000.append(
                        cftime.DatetimeNoLeap(t.year + 2000, t.month, t.day, t.hour)
                    )
                ds["time"].values = times_plus_2000
                ds = xr.decode_cf(ds)
        return ds
