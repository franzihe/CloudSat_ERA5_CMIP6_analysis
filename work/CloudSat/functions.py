from imports import xr, np


def create_xr_1D_ds(Profile_time, _lat, _lon, _Data_quality, var, unit, _var):
    _ds = xr.Dataset(
        {
            "Profile_time": xr.DataArray(
                data=Profile_time,
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": "seconds"},
            ),
            "Latitude": xr.DataArray(
                data=_lat[:, 0],
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": "degrees"},
            ),
            "Longitude": xr.DataArray(
                data=_lon[:, 0],
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": "degrees"},
            ),
            "Data_quality": xr.DataArray(
                data=_Data_quality[:, 0],
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": ""},
            ),
            var: xr.DataArray(
                data=_var[:, 0],
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": unit},
            ),
        },
        attrs={"example_attr": "this is a global attribute"},
    )

    return _ds


def create_xr_2D_ds(
    Profile_time, _lat, _lon, _Data_quality, pressure, temperature, var, unit, _var
):
    _ds = xr.Dataset(
        {
            "Profile_time": xr.DataArray(
                data=Profile_time,
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": "seconds"},
            ),
            "Latitude": xr.DataArray(
                data=_lat[:, 0],
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": "degrees"},
            ),
            "Longitude": xr.DataArray(
                data=_lon[:, 0],
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": "degrees"},
            ),
            "Data_quality": xr.DataArray(
                data=_Data_quality[:, 0],
                dims=["nray"],
                coords={
                    "nray": Profile_time,
                },
                attrs={"_FillValue": np.nan, "units": ""},
            ),
            "pressure": xr.DataArray(
                data=pressure,  # enter data here
                dims=["nray", "nbin"],
                coords={"nray": Profile_time, "nbin": np.arange(0, pressure.shape[1])},
                attrs={"_FillValue": np.nan, "units": "hPa"},
            ),
            "temperature": xr.DataArray(
                data=temperature,  # enter data here
                dims=["nray", "nbin"],
                coords={"nray": Profile_time, "nbin": np.arange(0, pressure.shape[1])},
                attrs={"_FillValue": np.nan, "units": "Kelvin"},
            ),
            var: xr.DataArray(
                data=_var,  #
                dims=["nray", "nbin"],
                coords={"nray": Profile_time, "nbin": np.arange(0, pressure.shape[1])},
                attrs={"_FillValue": np.nan, "units": unit},
            ),
        },
        attrs={"example_attr": "this is a global attribute"},
    )

    return _ds
