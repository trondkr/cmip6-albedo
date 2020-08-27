import xarray as xr
import cftime
import numpy as np
from cmip6_preprocessing.preprocessing import combined_preprocessing
import CMIP6_model
import CMIP6_config
import CMIP6_regrid
import xesmf as xe
import os


class CMIP6_IO:

    def __init__(self):
        self.models = []

    # Loop over all models and scenarios listed in CMIP6_light.config
    # and store each CMIP6 variable and scenario into a CMIP6 model object
    def organize_cmip6_datasets(self, config: CMIP6_config.Config_albedo):

        for experiment_id in config.experiment_ids:
            for grid_label in config.grid_labels:
                for source_id in config.source_ids:
                    if source_id in config.models.keys():
                        model_object = config.models[source_id]
                    else:
                        model_object = CMIP6_model.CMIP6_MODEL(name=source_id)

                    print("[CMIP6_IO] Organizing CMIP6 model object {}".format(model_object.name))

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

                            # Concatenate the historical and projections datasets
                            ds = xr.concat([ds_hist, ds_proj], dim="time")
                            # Remove the duplicate overlapping times (e.g. 2001-2014)
                            _, index = np.unique(ds["time"], return_index=True)
                            ds = ds.isel(time=index)

                            # Extract the time period of interest
                            ds = ds.sel(time=slice(config.start_date, config.end_date))
                            print("[CMIP6_IO] {} => Dates extracted {} range from {} to {}".format(source_id,
                                                                                                   variable_id,
                                                                                                   ds["time"].values[0],
                                                                                                   ds["time"].values[
                                                                                                       -1]))

                            # pass the pre-processing directly
                            dset_processed = combined_preprocessing(ds)
                            if variable_id in ["chl"]:
                                if source_id in ["CESM2", "CESM2-FV2", "CESM2-WACCM-FV2", "CESM2-WACCM"]:
                                    dset_processed = dset_processed.isel(lev_partial=config.selected_depth)
                                else:
                                    dset_processed = dset_processed.isel(lev=config.selected_depth)

                            # Save the info to model object
                            if member_id in model_object.member_ids is False:
                                model_object.member_ids.append(member_id)
                            if variable_id  in model_object.ocean_vars is False:
                                model_object.ocean_vars.append(variable_id)
                            self.dataset_into_model_dictionary(member_id, variable_id, dset_processed, model_object)

                    self.models.append(model_object)
                    print("[CMIP6_IO] Stored {} variables for model {}".format(len(model_object.ocean_vars),
                                                                               model_object.name))

    def dataset_into_model_dictionary(self,
                                      member_id: str,
                                      variable_id: str,
                                      dset: xr.Dataset,
                                      model_object: CMIP6_model.CMIP6_MODEL):
        # Store each dataset for each variable as a dictionary of variables for each member_id
        try:
            existing_ds = model_object.ds_sets[member_id]
        except KeyError:
            existing_ds = {}
        existing_ds[variable_id] = dset

        model_object.ds_sets[member_id] = existing_ds

    def perform_cmip6_query(self, config: CMIP6_config.Config_albedo, query_string: str) -> xr.Dataset:
        df_sub = config.df.query(query_string)
        if df_sub.zstore.values.size == 0:
            return df_sub

        mapper = config.fs.get_mapper(df_sub.zstore.values[-1])
        ds = xr.open_zarr(mapper, consolidated=True)  # , mask_and_scale=True)
        # print("[CMIP6_IO] chunks {}".format(ds.chunks))

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

    """
        Regrid to cartesian grid and save to NetCDF:
        For any Amon related variables (wind, clouds), the resolution from CMIP6 models is less than
        1 degree longitude x latitude. To interpolate to a 1x1 degree grid we therefore first interpolate to a
        2x2 degrees grid and then subsequently to a 1x1 degree grid.
    """

    def extract_dataset_and_save_to_netcdf(self, model_obj, config: CMIP6_config.Config_albedo):
        for key in model_obj.ds_sets[model_obj.current_member_id].keys():
            outfile = "{}CMIP6_{}_{}_{}.nc".format(config.cmip6_outdir, model_obj.name, model_obj.current_member_id,
                                                   key)
            print(outfile)
        return

        if not os.path.exists(config.cmip6_outdir): os.mkdir(config.cmip6_outdir)

        ds_out_amon = xe.util.grid_2d(config.min_lon,
                                      config.max_lon, 2,
                                      config.min_lat,
                                      config.max_lat, 2)
        ds_out = xe.util.grid_2d(config.min_lon,
                                 config.max_lon, 1,
                                 config.min_lat,
                                 config.max_lat, 1)

        re = CMIP6_regrid.CMIP6_regrid()

        for key in model_obj.ds_sets[model_obj.current_member_id].keys():

            current_ds = model_obj.ds_sets[model_obj.current_member_id][key].sel(
                y=slice(config.min_lat, config.max_lat),
                x=slice(config.min_lon, config.max_lon))

            if key in ["chl", "sithick", "siconc", "sisnthick", "sisnconc"]:
                ds_trans = current_ds.chunk({'time': -1}).transpose('bnds', 'time', 'vertex', 'y', 'x')
            else:
                ds_trans = current_ds.chunk({'time': -1}).transpose('bnds', 'time', 'y', 'x')

            if key in ["uas", "vas", "clt", "chl"]:
                out_amon = re.regrid_variable(key,
                                              ds_trans,
                                              ds_out_amon,
                                              interpolation_method=config.interp,
                                              use_esmf_v801=config.use_esmf_v801).to_dataset()

                out = re.regrid_variable(key, out_amon, ds_out,
                                         interpolation_method=config.interp,
                                         use_esmf_v801=config.use_esmf_v801)
            else:
                out = re.regrid_variable(key, ds_trans,
                                         ds_out,
                                         interpolation_method=config.interp,
                                         use_esmf_v801=config.use_esmf_v801)

            outfile = "{}CMIP6_{}_{}_{}.nc".format(config.cmip6_outdir, model_obj.name, model_obj.current_member_id,
                                                   key)
            if os.path.exists(outfile): os.remove(outfile)

            # Convert to dataset before writing to netcdf file. Writing to file downlods and concatenates all
            # of the data and we therefore re-chunk to split the process into several using dask
            ds = out.to_dataset()
            ds.chunk({'time': -1, 'y': 10, 'x': 50}).to_netcdf(path=outfile, format='NETCDF4', engine='netcdf4')
            print("[CMIP6_light] wrote variable {} to file".format(key))
