# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# Read in the CloudSat R05 data and create the monthly mean of
# information of variables: http://www.cloudsat.cira.colostate.edu/data-products/level-2c/2c-snow-profile?term=90
# Documentation: http://www.cloudsat.cira.colostate.edu/sites/default/files/products/files/2C-SNOW-PROFILE_PDICD.P1_R05.rev0_.pdf


# 1D variables
# 'snowfall_rate_sfc'

# 2D variables
# 'Height'
# 'snowfall_rate'
# 'snow_water_content'


# necessary variables
# 'Latitude'
# 'Longitude'
# 'Vertical_binsize'
# profile times as YYYYMMDD-HH-MM-SS


# %%
# supress warnings
import warnings

warnings.filterwarnings("ignore")  # don't output warnings

# import packages
from imports import glob, pySD, pyHDF, read_var_eos, xr, np, datetime, timedelta, fct


# %%
year = 2008


one_D = False
two_D = True


available_month = {
    "1": "01",
    # "2": "02",
    # "3": "03",
    # "4": "04",
    # "5": "05",
    # "6": "06",
    # "7": "07",
    # "8": "08",
    # "9": "09",
    # "10": "10",
    # "11": "11",
    # "12": "12",
}


# %%
# datapath = '/tos-project2/NS9600K/data'
path = "/scratch/franzihe"
datapath = "{:}/input/cloudsat/2C-SNOW-PROFILE.P1_R05".format(path)
ff_cs = sorted(glob("{}/{}/*/*.hdf".format(datapath, year,)))


# %%
filepath = "{:}/input/cloudsat/ECMWF-AUX.P_R05".format(path)
ff_ec = sorted(glob("{}/{}/*/*.hdf".format(filepath, year,)))


# %%
if one_D == True:
    # 1D variables
    variables = {
        # 'DEM_elevation'               : 'm',        #Elevation in meters above Mean Sea Level. A value of -9999 indicates ocean. A value of 9999 indicates an error in calculation of the elevation.
        # 'Vertical_binsize'            : '',         #effective vertical height of the radar range bin.
        "snowfall_rate_sfc": "mm h-1",  # Surface snowfall rate in mm of liquid water per hour. The specified range is typical.
        # 'snowfall_rate_sfc_uncert'    : '',         #The estimated 1-sigma uncertainty of the surface snowfall rate in mm of liquid water per hour. The specified range is typical.
        # 'snowfall_rate_sfc_confidence': '',         #Flag indicating the relative quality of the surface snowfall rate estimate. 4: High confidence
    }

if two_D == True:
    # 2D variables
    variables = {
        # 'Height'                    : '',               #Height of the radar range bins in meters above mean sea level.
        "snowfall_rate": "mm h-1",  # Profile of snowfall rates in the precipitating column in mm of liquid water per hour. The specified range is typical.
        # 'snowfall_rate_uncert'      : '',               #The estimated 1-sigma uncertainties of the snowfall rates in the precipitating column. The specified range is typical.
        # 'snow_water_content'        : 'g kg-1',          #Profile of snow water content in the precipitating column in grams per m^3. The specified range is typical.
        # 'snow_water_content_uncert' : '',                #The estimated 1-sigma uncertainties of the snow water contents in the precipitating column in grams per m^3. The specified range is typical.
    }


# %%
pressure_grid = np.array(
    [
        24.0,
        25.0,
        26.0,
        27.0,
        28.0,
        29.0,
        30.0,
        32.0,
        33.0,
        34.0,
        35.0,
        37.0,
        38.0,
        40.0,
        41.0,
        43.0,
        44.0,
        45.0,
        48.0,
        50.0,
        52.0,
        54.0,
        55.0,
        58.0,
        60.0,
        63.0,
        65.0,
        68.0,
        70.0,
        73.0,
        75.0,
        80.0,
        83.0,
        85.0,
        90.0,
        93.0,
        98.0,
        100.0,
        105.0,
        110.0,
        113.0,
        115.0,
        120.0,
        125.0,
        130.0,
        135.0,
        140.0,
        145.0,
        155.0,
        160.0,
        165.0,
        170.0,
        180.0,
        185.0,
        190.0,
        200.0,
        210.0,
        215.0,
        225.0,
        230.0,
        240.0,
        250.0,
        260.0,
        270.0,
        280.0,
        290.0,
        300.0,
        310.0,
        320.0,
        330.0,
        345.0,
        360.0,
        370.0,
        380.0,
        395.0,
        400.0,
        425.0,
        440.0,
        450.0,
        470.0,
        480.0,
        500.0,
        515.0,
        530.0,
        550.0,
        570.0,
        585.0,
        600.0,
        625.0,
        645.0,
        665.0,
        685.0,
        700.0,
        725.0,
        750.0,
        770.0,
        800.0,
        825.0,
        850.0,
        870.0,
        900.0,
        925.0,
        950.0,
        988.0,
        1000.0,
        1010,
        1015,
        1020,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
)


# %%
counter = 0
for month, mm in available_month.items():
    if one_D == True:
        ds = xr.Dataset(
            data_vars=dict(
                Profile_time=(["nray"], np.empty(shape=(0,), dtype="datetime64[s]")),
                Latitude=(["nray"], np.empty(shape=(0,),)),
                Longitude=(["nray"], np.empty(shape=(0,),)),
                Data_quality=(["nray"], np.empty(shape=(0,),)),
            ),
            coords=dict(nray=([]), nbin=([])),
            attrs=None,
        )
    if two_D == True:
        ds = xr.Dataset(
            data_vars=dict(
                Profile_time=(["nray"], np.empty(shape=(0,), dtype="datetime64[s]")),
                Latitude=(["nray"], np.empty(shape=(0,),)),
                Longitude=(["nray"], np.empty(shape=(0,),)),
                Data_quality=(["nray"], np.empty(shape=(0,),)),
                pressure=(["nray", "nbin"], np.empty(shape=(0, 0),)),
                temperature=(["nray", "nbin"], np.empty(shape=(0, 0),)),
            ),
            coords=dict(nray=([]), nbin=([])),
            attrs=None,
        )
    for var, unit in variables.items():
        # create new variable
        if one_D == True:
            ds[var] = xr.DataArray(
                data=np.full(shape=(0,), fill_value=np.nan),
                dims=["nray"],
                attrs={"units": unit},
            )
        if two_D == True:
            ds[var] = xr.DataArray(
                data=np.full(shape=(0, 0), fill_value=np.nan),
                dims=["nray", "nbin"],
                attrs={"units": unit},
            )
        filename = "{var}_{year}{month}.nc".format(var=var, year=year, month=mm)
        savepath = "{path}/output/cloudsat/2C-SNOW_onemonth_onevariable/{year}/".format(
            path=path, year=year
        )
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
            for i in range(2):  # len(ff_cs)):
                # for i in range(6): # read in one file and bring 2D Variables on a common pressure grid
                year = int(ff_cs[i].split("/")[-3])
                doy = int(ff_cs[i].split("/")[-2])  # day of the year
                _t = datetime(year, 1, 1) + timedelta(doy - 1)  # create date

                if _t.month != int(month):
                    continue
                elif _t.month == int(month):

                    # Read in CloudSat
                    f_SD_ptr = pySD.SD(ff_cs[i], pySD.SDC.READ)
                    f_VD_ptr = pyHDF.HDF(ff_cs[i], pyHDF.HC.READ)

                    # get profile times from file
                    Profile_time = read_var_eos.get_profile_times(f_VD_ptr)

                    # get geolocation
                    _lat = read_var_eos.get_1D_var(
                        f_VD_ptr, "Latitude"
                    )  # Spacecraft Geodetic Latitude.
                    _lon = read_var_eos.get_1D_var(
                        f_VD_ptr, "Longitude"
                    )  # Spacecraft geodetic longitude

                    # get data quality
                    _Data_quality = read_var_eos.get_1D_var(
                        f_VD_ptr, "Data_quality"
                    )  # Flags indicating data quality. If 0, then data is of good quality.

                    # get variable
                    if one_D == True:
                        _var = read_var_eos.get_1D_var(f_VD_ptr, var)
                    if two_D == True:
                        _var = read_var_eos.get_2D_var(f_SD_ptr, f_VD_ptr, var)

                    # assign np.nan where missing vallues
                    _var[np.where(_var == -999.0)] = np.nan

                    f_VD_ptr.close()
                    f_SD_ptr.end()

                    # create dataset
                    if one_D == True:
                        _ds = fct.create_xr_1D_ds(
                            Profile_time, _lat, _lon, _Data_quality, var, unit, _var
                        )

                    if two_D == True:
                        # Read in ECMWF-Aux files for pressure averaging for 2D files
                        f_SD_ptr = pySD.SD(ff_ec[i], pySD.SDC.READ)
                        f_VD_ptr = pyHDF.HDF(ff_ec[i], pyHDF.HC.READ)

                        # # Sometimes different data products don’t have the same dimensions, e.g. 2007 granule 3853
                        # if lwc.shape != iwc.shape:
                        #     dimension_failure += 1
                        #     print(‘Skipping granule (dimension failure)...’)
                        #     continue

                        # get 2D variable
                        pressure = read_var_eos.get_2D_var(
                            f_SD_ptr, f_VD_ptr, "Pressure"
                        )
                        temperature = read_var_eos.get_2D_var(
                            f_SD_ptr, f_VD_ptr, "Temperature"
                        )

                        # convert pressure into hPa
                        pressure[np.where(pressure == -999.0)] = np.nan
                        pressure = pressure / 100.0

                        # assign np.nan where missing vallues
                        temperature[np.where(temperature == -999.0)] = np.nan

                        f_VD_ptr.close()
                        f_SD_ptr.end()
                        # create dataset
                        _ds = fct.create_xr_2D_ds(
                            Profile_time,
                            _lat,
                            _lon,
                            _Data_quality,
                            pressure,
                            temperature,
                            var,
                            unit,
                            _var,
                        )

                        # assign pressure grid coordinate
                        _ds = _ds.assign_coords(pressure_grid=pressure_grid)

                        # define new variable to be on the pressure grid
                        _ds[var + "_regrid"] = (
                            ["nray", "pressure_grid",],
                            np.full(
                                shape=(len(_ds.nray), len(_ds.pressure_grid)),
                                fill_value=np.nan,
                            ),
                        )

                        # put the 2D variable on equal pressure grid
                        for t in range(len(_ds.nray)):
                            for k in range(len(_ds.pressure_grid)):
                                # First, find the index of the grid point nearest a specific pressure level
                                abs_pressure = np.abs(
                                    _ds.pressure.isel(nray=t)
                                    - _ds.pressure_grid.isel(pressure_grid=k)
                                )
                                c = abs_pressure

                                try:
                                    ([xloc,]) = np.where(c == np.nanmin(c))
                                    # Now I can use that index location to get the values at the x/y diminsion
                                    _ds[var + "_regrid"][
                                        t, xloc
                                    ] = _ds.snowfall_rate.isel(nray=t).sel(nbin=xloc)

                                except:
                                    print("c values", np.nanmin(c))
                                    _ds[var + "_regrid"][t, xloc] = np.nan

                        _ds = _ds.drop_vars(var)

                    ds = xr.concat([ds, _ds], dim="nray")

            ds.to_netcdf(
                path="{savepath}{filename}".format(savepath=savepath, filename=filename)
            )
            print(
                "file saved: {savepath}{filename}".format(
                    savepath=savepath, filename=filename
                )
            )



# %%

# %%
