# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %% [markdown]
# # Regrid CloudSat data to common 2D grid
# The CloudSat data is still in as a granule, but to compare it to ERA5 and CMIP6 models the data has to be in a 2D grid with coordinates `(time, lat, lon)`.

# %%
# supress warnings
import warnings

warnings.filterwarnings("ignore")  # don't output warnings

# import packages
from imports import glob, xr, np


# %%
year = 2008
var = "snowfall_rate_sfc"

# filepath = '/tos-project2/NS9600K/franzihe/data/cloudsat/2c_snow_onemonth_onevariable'
# filepath = '/cluster/work/users/franzihe/data/cloudsat/2c_snow_onemonth_onevariable'
path = "/scratch/franzihe"
datapath = "{path}/output/cloudsat/2C-SNOW_onemonth_onevariable/{year}".format(
    path=path, year=year
)

ff_cs = sorted(
    glob("{datapath}/{var}_{year}*.nc".format(datapath=datapath, var=var, year=year))
)
ff_cs


# %%
# Read in NorESM-MM grid to get CloudSat data on same grid as NorESM2-MM
noresm = xr.open_dataset(
    "{path}/input/cmip6_hist/1deg/grid_NorESM2-MM.nc".format(path=path)
)


# %%
# define time array
time = xr.DataArray(
    data=np.full(shape=(0,), dtype="datetime64[s]", fill_value=np.nan),
    dims=dict(time=([])),
)
time = time.assign_coords(
    {"time": np.full(shape=(0,), dtype="datetime64[s]", fill_value=np.nan)}
)

# Define lat, lon grid
lat_grid = noresm.lat.values  # np.arange(-90, 100, 0.5)
lon_grid = noresm.lon.values  # np.arange(-180, 190, 0.5)

# Define lat, lon boundaries
lat_bnds = noresm.lat_bnds.values
lon_bnds = noresm.lon_bnds.values


counter = 0
for i in range(len(ff_cs)):
    # read in files
    _ds = xr.open_dataset(ff_cs[i])

    # create time array with monthly averaged time
    _time_mean = _ds.Profile_time.mean(skipna=True, keep_attrs=True).assign_coords(
        {"time": _ds.Profile_time.mean(skipna=True, keep_attrs=True)}
    )
    time = xr.concat([time, _time_mean], dim="time")


# define global variable
_var_mean_grid = xr.DataArray(
    data=np.full(shape=(len(time), len(lat_grid), len(lon_grid)), fill_value=np.nan),
    dims=dict(time=([]), lat=([]), lon=([])),
)
for i in range(len(ff_cs)):
    # assign nan where Data Quality != 0
    _ds[var] = _ds[var].where(_ds.Data_quality == 0, drop=True)
    # calculate mean of variable with lat, lon grid
    count_lon = 0
    count_lat = 0

    for _lon, _lonb in zip(lon_grid, lon_bnds):
        # print('_lon: ', _lon)
        for _lat, _latb in zip(lat_grid, lat_bnds):
            filter1 = (_ds.Longitude > _lonb[0]) & (_ds.Longitude < _lonb[1])
            filter2 = (_ds.Latitude > _latb[0]) & (_ds.Latitude < _latb[1])

            _var = _ds[var].isel(nray=(filter1 & filter2))
            _var_mean_grid[i, count_lat, count_lon] = _var.mean(
                skipna=True, keep_attrs=True
            )
            count_lat += 1

        count_lon += 1
        count_lat = 0

# assign coordinates to variable
_var_mean_grid = _var_mean_grid.assign_coords(
    {"time": time, "lat": lat_grid, "lon": lon_grid}
)

filename = "{var}_{year}.nc".format(var=var, year=year)
savepath = "{path}/output/cloudsat/2C-SNOW_lat_lon/{year}/".format(path=path, year=year)
files = glob(savepath + filename)
if savepath + filename in files:
    print(
        "{savepath}{filename} is downloaded".format(
            savepath=savepath, filename=filename
        )
    )
    counter += 1
    print("Have downloaded in total: {:} files".format(str(counter)))
else:
    _var_mean_grid.to_netcdf(
        "{savepath}{filename}".format(savepath=savepath, filename=filename)
    )
    print(
        "file saved: {savepath}{filename}".format(savepath=savepath, filename=filename)
    )


# %%

