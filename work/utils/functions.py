import xarray as xr

import xesmf as xe
import cftime


def regrid_data(ds_in, ds_out):
    # Regridder data
    regridder = xe.Regridder(ds_in, ds_out, "bilinear")

    # Apply regridder to data
    # the entire dataset can be processed at once
    ds_in_regrid = regridder(ds_in)

    # verify that the result is the same as regridding each variable one-by-one
    for k in ds_in.data_vars:
        print(k, ds_in_regrid[k].equals(regridder(ds_in[k])))

        if ds_in_regrid[k].equals(regridder(ds_in[k])) == True:
            ### Assign attributes from the original file to the regridded data
            #  ds_in_regrid.attrs['Conventions'] = ds_in.attrs['Conventions']
            # ds_in_regrid.attrs['history']     = ds_in.attrs['history']
            ds_in_regrid.attrs = ds_in.attrs

            ds_in_regrid[k].attrs["units"] = ds_in[k].attrs["units"]
            ds_in_regrid[k].attrs["long_name"] = ds_in[k].attrs["long_name"]
            try:
                ds_in_regrid[k].attrs["standard_name"] = ds_in[k].attrs["standard_name"]
                ds_in_regrid[k].attrs["comment"] = ds_in[k].attrs["comment"]
                ds_in_regrid[k].attrs["original_name"] = ds_in[k].attrs["original_name"]
                ds_in_regrid[k].attrs["cell_methods"] = ds_in[k].attrs["cell_methods"]
                ds_in_regrid[k].attrs["cell_measures"] = ds_in[k].attrs["cell_measures"]
            except KeyError:
                continue
    return ds_in_regrid

