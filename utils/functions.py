from imports import (
    xr,
    xe,
    cftime,
    plt,
    ccrs,
    cm,
    cy,
    np,
    linregress,
    pd,
    da,
    datetime,
    timedelta,
    BoundaryNorm,
    Patch,
    Line2D,
    r2_score,
    LinearSegmentedColormap, 
    Normalize,
    FuncFormatter,
    FormatStrFormatter,
    Rectangle
    
)

import warnings

# fig_label = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)']
fig_label = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 'u)', 'v)', 'w)', 'x)', 'y)', 'z)',
             'aa)', 'bb)', 'cc)', 'dd)', 'ee)', 'ff)', 'gg)', 'hh)', 'ii)', 'jj)', 'kk)', 'll)', 'mm)', 'nn)', 'oo)', 'pp)', 'qq)', 'rr)', 'ss)', 'tt)', 'uu)', 'vv)', 'ww)', 'xx)', 'yy)', 'zz)']

class MidpointNormalize(Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def rename_coords_lon_lat(ds):
    for k in ds.indexes.keys():
        if k == "longitude":
            ds["longitude"].attrs = {
                "standard_name": "longitude",
                "units": "degrees_east",
            }
            ds = ds.rename({"longitude": "lon"})
        if k == "latitude":
            ds["latitude"].attrs = {
                "standard_name": "latitude",
                "units": "degrees_north",
            }
            ds = ds.rename({"latitude": "lat"})

    return ds


def regrid_data(ds_in, ds_out):
    ds_in = rename_coords_lon_lat(ds_in)
    ds_out = rename_coords_lon_lat(ds_out)

    # Regridder data
    regridder = xe.Regridder(ds_in, ds_out, "conservative")  # "bilinear")  #
    # regridder.clean_weight_file()

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


def dataset_IWC_LWC_level(ds, iwc_stat, var1, var2, data_name):
    dset_dict = dict()
    dset_dict[data_name] = ds
    for i in iwc_stat.items():
        dset_dict["{}_{}".format(data_name, i[0])] = find_IWC_LWC_level(
            ds, var1=var1, var2=var2, value=i[1], coordinate="level"
        )

    ## Connect all statistics into one Dataset with new coordinate 'statistic'
    _ds = list(dset_dict.values())
    _coord = list(dset_dict.keys())
    ds_new = xr.concat(objs=_ds, dim=_coord, coords="all").rename(
        {"concat_dim": "statistic"}
    )

    return ds_new


def find_IWC_LWC_level(ds, var1, var2, value, coordinate):
    # 1.1. IWC + LWC = 100%
    iwc_lwc = ds[var1] + ds[var2]
    # 1.2. IWC/(IWC + LWC)  = fraction_iwc
    fraction = ds[var1] / iwc_lwc
    # 2. level where fraction_iwc == 0.5 and fraction_lwc == 0.5 or given value
    # use the closest layer as it might not be exactly 0.5
    filter_mc = find_nearest(fraction, value, coordinate)

    # 3. get values where level given value

    # p_5050 = ds["pressure"].where(filter_mc)  # e.g. P@50/50
    # filter_pres = p_5050 == p_5050.max(dim=coordinate)

    ds_out = xr.Dataset()
    for var_id in list(ds.keys()):
        var = ds[var_id].where(filter_mc)
        ds_out[var_id] = var
        # ds_out[var_id] = var.sum(dim=coordinate, skipna=True)
    return ds_out


def find_nearest(array, value, coordinate):
    filter1 = array[coordinate] == abs(array - value).idxmin(
        dim=coordinate, skipna=True
    )
    return filter1


def seasonal_mean_std(
    ds,
    var,
):
    ds[var + "_mean"] = (
        ds[var].groupby("time.season").mean("time", keep_attrs=True, skipna=True)
    )
    ds[var + "_std"] = (
        ds[var].groupby("time.season").std("time", keep_attrs=True, skipna=True)
    )

    return ds


def plt_spatial_seasonal_mean(variable, global_mean, title, var_id, extend):

    fg = variable.plot(
        col="season",
        col_wrap=2,
        transform=ccrs.PlateCarree(),  # remember to provide this!
        subplot_kws={
            "projection": ccrs.PlateCarree()
        },
        cbar_kwargs={"orientation": "vertical", "shrink": 0.8, "aspect": 40},
        cmap=cm.devon_r, 
        figsize=[10, 7],
        robust=True,
        extend=extend,
        add_colorbar=False,
        vmin=plt_dict[var_id][plt_dict['header'].index('vmin')],
        vmax=plt_dict[var_id][plt_dict['header'].index('vmax')],
        levels=plt_dict[var_id][plt_dict['header'].index('levels')],
    )
    for ax, i in zip(fg.axes.flat, variable.season.values):
        ax.set_title('season: {}, global mean: {:.3f}'.format(i, global_mean.sel(season=i).values))
    # lets add a coastline to each axis
    # great reason to use FacetGrid.map
    fg.map(lambda: plt.gca().coastlines())
    fg.fig.suptitle(title, fontsize=16, fontweight="bold")


    fg.add_colorbar(fraction=0.05, pad=0.04)

    fg.cbar.set_label(label='{}'.format(plt_dict[var_id][plt_dict['header'].index('label')], weight='bold'))
    


plt_dict = {
    "header": ["label", "vmin", "vmax", "levels", "vmin_std", "vmax_std"],
    "sf": ["Snowfall (mm$\,$day$^{-1}$)", 0, 2.5, 26, 0, 0.6],
    "tp": ["Total precipitation (mm$\,$day$^{-1}$)", 0, 10, 25, 0, 2.4],
    "tciw": ["Ice Water Path (g$\,$m$^{-2}$)", 0, 100, 25, 0, 20],
    "tclw": ["Liquid Water Path (g$\,$m$^{-2}$)", 0, 100, 25, 0, 20],
    "2t": ["2-m temperature (K)", 235, 310, 25, 0, 6],
    "t": ["Air temperature (K)", 230, 260, 30, 0, 6 ],
    "msr": ["Mean snowfall rate (mm$\,$day$^{-1}$)", 0, 2.5, 26, 0, 1],
    "clic": ["Specific cloud ice water content (g kg$^{-1}$)", 0, 0.01, 11, 0, 1],
    "clwc": ["Specific cloud liquid water content (g kg$^{-1}$)", 0, 0.01, 11, 0, 1],
    "cswc": ["Specific snow water content (g kg$^{-1}$)", 0.05, 0.08, 11, 0, 1],
    "pressure": ["Pressure", 400, 1000, 26, 0, 50],
}

to_era_variable = {
    "tas": "2t",
    "cli": "clic",
    "clw": "clwc",
    "prsn": "sf",
    "ta": "t",
    "clivi": "tciw",
    "lwp": "tclw",
    "pr": "tp",
}


def plt_diff_seasonal(
    true_var, estimated_var, cbar_label, vmin=None, vmax=None, levels=None, title=None
):

    fig, axsm = plt.subplots(
        2, 2, figsize=[10, 7], subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axs = axsm.flatten()
    for ax, i in zip(axs, true_var.season):
        im = (
            (true_var - estimated_var)
            .sel(season=i)
            .plot.contourf(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cm.bam,
                robust=True,
                add_colorbar=False,
                extend="both",
                vmin=vmin,
                vmax=vmax,
                levels=levels,
            )
        )
        # Plot cosmetics
        ax.coastlines()
        gl = ax.gridlines()
        ax.add_feature(cy.feature.BORDERS)
        gl.top_labels = False
        ax.set_title("season: {}".format(i.values))

    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1, 0.15, 0.025, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="vertical", fraction=0.046, pad=0.04)
    # set cbar label
    cb.set_label(label=cbar_label, weight="bold")

    plt.tight_layout()
    fig.subplots_adjust(top=1)

    return axs


def plt_zonal_seasonal(variable_model, title=None, label=None):
    fig, axsm = plt.subplots(
        2,
        2,
        figsize=[10, 7],
        sharex=True,
        sharey=True,
    )

    fig.suptitle(title, fontsize=16, fontweight="bold")

    axs = axsm.flatten()
    for ax, i in zip(axs, variable_model.season):
        for k, c in zip(
            variable_model.model.values,
            cm.romaO(range(0, 256, int(256 / len(variable_model.model.values)))),
        ):
            variable_model.sel(season=i, model=k).plot(
                ax=ax,
                label=k,
                color=c,
            )

        ax.set_ylabel(label, fontweight="bold")
        ax.grid()
        ax.set_title("season: {}".format(i.values))

    axs[1].legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1),
        title=label,
        fontsize="small",
        fancybox=True,
    )

    return axs


def plt_bar_area_mean(
    ax,
    var_model,
    var_obs,
    loc,
    bar_width=None,
    hatch=None,
    alpha=None,
    label=None,
    ylabel=None,
):

    for k, c, pos in zip(
        var_model.model.values,
        cm.romaO(range(0, 256, int(256 / len(var_model.model.values)))),
        range(len(var_model.model.values)),
    ):
        ax.bar(
            pos + loc * bar_width,
            var_model.sel(model=k).values,
            color=c,
            width=bar_width,
            edgecolor="black",
            hatch=hatch,
            alpha=alpha,
        )

    ax.bar(
        len(var_model.model.values) + loc * bar_width,
        var_obs.values,
        color="k",
        width=bar_width,
        edgecolor="white",
        hatch=hatch,
        alpha=alpha,
        label=label,
    )

    ax.set_xticks(range(len(np.append((var_model.model.values), "ERA5").tolist())))
    ax.set_xticklabels(
        np.append((var_model.model.values), "ERA5").tolist(), fontsize=12, rotation=90
    )
    ax.set_ylabel(ylabel, fontweight="bold")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="MEAN",
        fontsize="small",
        fancybox=True,
    )

    plt.tight_layout()


def calc_regression(ds, ds_result, lat, step, season, model=None):

    x = ds["iwp_{}_{}".format(lat, lat + step)].sel(season=season).values.flatten()
    y = ds["sf_{}_{}".format(lat, lat + step)].sel(season=season).values.flatten()

    mask = ~np.isnan(y) & ~np.isnan(x)

    if x[mask].size == 0 or y[mask].size == 0:
        # print(
        #     "Sorry, nothing exists for {} in {}. ({}, {})".format(
        #         model, season, lat, lat + step
        #     )
        # )
        ds_result[
            "slope_{}_{}".format(
                lat,
                lat + step,
            )
        ] = np.nan
        ds_result[
            "intercept_{}_{}".format(
                lat,
                lat + step,
            )
        ] = np.nan
        ds_result["rvalue_{}_{}".format(lat, lat + step)] = np.nan
    else:
        _res = linregress(x[mask], y[mask])

        ds_result[
            "slope_{}_{}".format(
                lat,
                lat + step,
            )
        ] = _res.slope
        ds_result[
            "intercept_{}_{}".format(
                lat,
                lat + step,
            )
        ] = _res.intercept
        ds_result["rvalue_{}_{}".format(lat, lat + step)] = _res.rvalue

    return ds_result


def plt_scatter_iwp_sf_seasonal(
    ds, linreg, iteration, step, title=None, xlim=None, ylim=None
):

    fig, axsm = plt.subplots(
        2,
        2,
        figsize=[10, 7],
        sharex=True,
        sharey=True,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axs = axsm.flatten()
    for ax, i in zip(axs, ds.season):
        ax.grid()
        for _lat, c in zip(iteration, cm.romaO(range(0, 256, int(256 / 4)))):
            # plot scatter
            ax.scatter(
                ds["iwp_{}_{}".format(_lat, _lat + step)].sel(season=i),
                ds["sf_{}_{}".format(_lat, _lat + step)].sel(season=i),
                label="{}, {}".format(_lat, _lat + step),
                color=c,
                alpha=0.5,
            )

            # plot regression line
            y = (
                np.linspace(0, 350)
                * linreg["slope_{}_{}".format(_lat, _lat + step)].sel(season=i).values
                + linreg["intercept_{}_{}".format(_lat, _lat + step)]
                .sel(season=i)
                .values
            )

            ax.plot(np.linspace(0, 350), y, color=c, linewidth="2")

        ax.set_ylabel("Snowfall (mm$\,$day$^{-1}$)", fontweight="bold")
        ax.set_xlabel("Ice Water Path (g$\,$m$^{-2}$)", fontweight="bold")
        ax.set_title(
            "season: {}; lat: ({}, {})".format(
                i.values, iteration[0], iteration[-1] + step
            )
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    axs[1].legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize="small",
        fancybox=True,
    )

    plt.tight_layout()


def return_ds_regression(ds, iteration, step):
    ds_result = dict()
    for season in ds.season.values:
        ds_res = xr.Dataset()
        for _lat in iteration:

            ds_res = calc_regression(ds, ds_res, _lat, step, season)

        ds_result[season] = ds_res

    ds_linear_reg = xr.concat(
        objs=list(ds_result.values()), dim=list(ds_result.keys()), coords="all"
    ).rename({"concat_dim": "season"})

    return ds_linear_reg


def return_pd_dataFrame_corr_coeff(ds, season, iteration, step):

    rvalue = []
    slope = []
    intercept = []
    index_season = []
    index_lat = []

    for lat in iteration:
        index_season.append("{}".format(season))
        index_lat.append("[{},{}]".format(lat, lat + step))
        rvalue.append(
            ds["rvalue_{}_{}".format(lat, lat + step)].sel(season=season).values.item()
        )
        slope.append(
            ds["slope_{}_{}".format(lat, lat + step)].sel(season=season).values.item()
        )
        intercept.append(
            ds["intercept_{}_{}".format(lat, lat + step)]
            .sel(season=season)
            .values.item()
        )

    da = pd.concat(
        [
            pd.Series(index_season),
            pd.Series(rvalue),
            pd.Series(slope),
            pd.Series(intercept),
        ],
        axis=1,
    )
    da.columns = ["season", "R**2", "a", "b"]
    return (da.T, pd.Series(index_lat))


# log_interpolate_1d_V2 is taken from the Metpy package https://github.com/Unidata/MetPy/blob/3dc05c8ae40a784956327907955fc09a3e46e0e1/src/metpy/interpolate/one_dimension.py#L53
def log_interpolate_1d_V2(x, xp, *args, axis=0, fill_value=np.nan):
    r"""Interpolates data with logarithmic x-scale over a specified axis.
    Interpolation on a logarithmic x-scale for interpolation values in pressure coordinates.
    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.
    xp : array-like
        The x-coordinates of the data points.
    args : array-like
        The data to be interpolated. Can be multiple arguments, all must be the same shape as
        xp.
    axis : int, optional
        The axis to interpolate over. Defaults to 0.
    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If None, will return
        ValueError if points are out of bounds. Defaults to nan.
    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in ascending order.
    Examples
    --------
     >>> x_log = np.array([1e3, 1e4, 1e5, 1e6])
     >>> y_log = np.log(x_log) * 2 + 3
     >>> x_interp = np.array([5e3, 5e4, 5e5])
     >>> metpy.interpolate.log_interpolate_1d(x_interp, x_log, y_log)
     array([20.03438638, 24.63955657, 29.24472675])
    Notes
    -----
    xp and args must be the same shape.
    """
    # # Handle units
    # x, xp = _strip_matching_units(x, xp)

    # Log x and xp
    log_x = np.log(x)
    log_xp = np.log(xp)
    return interpolate_1d(log_x, log_xp, *args, axis=axis, fill_value=fill_value)


def interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan, return_list_always=False):
    r"""Interpolates data with any shape over a specified axis.
    Interpolation over a specified axis for arrays of any shape.
    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.
    xp : array-like
        The x-coordinates of the data points.
    args : array-like
        The data to be interpolated. Can be multiple arguments, all must be the same shape as
        xp.
    axis : int, optional
        The axis to interpolate over. Defaults to 0.
    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If None, will return
        ValueError if points are out of bounds. Defaults to nan.
    return_list_always: bool, optional
        Whether to always return a list of interpolated arrays, even when only a single
        array is passed to `args`. Defaults to ``False``.
    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in ascending order.
    Examples
    --------
     >>> x = np.array([1., 2., 3., 4.])
     >>> y = np.array([1., 2., 3., 4.])
     >>> x_interp = np.array([2.5, 3.5])
     >>> metpy.interpolate.interpolate_1d(x_interp, x, y)
     array([2.5, 3.5])
    Notes
    -----
    xp and args must be the same shape.
    """
    # # Handle units
    # x, xp = _strip_matching_units(x, xp)

    # Make x an array
    x = np.asanyarray(x).reshape(-1)

    # Save number of dimensions in xp
    ndim = xp.ndim

    # Sort input data
    sort_args = np.argsort(xp, axis=axis)
    sort_x = np.argsort(x)

    # indices for sorting
    sorter = broadcast_indices(xp, sort_args, ndim, axis)

    # sort xp
    xp = xp[sorter]
    # Ensure pressure in increasing order
    variables = [arr[sorter] for arr in args]

    # Make x broadcast with xp
    x_array = x[sort_x]
    expand = [np.newaxis] * ndim
    expand[axis] = slice(None)
    x_array = x_array[tuple(expand)]

    # Calculate value above interpolated value
    minv = np.apply_along_axis(np.searchsorted, axis, xp, x[sort_x])
    minv2 = np.copy(minv)

    # If fill_value is none and data is out of bounds, raise value error
    if ((np.max(minv) == xp.shape[axis]) or (np.min(minv) == 0)) and fill_value is None:
        raise ValueError("Interpolation point out of data bounds encountered")

    # Warn if interpolated values are outside data bounds, will make these the values
    # at end of data range.
    if np.max(minv) == xp.shape[axis]:
        warnings.warn("Interpolation point out of data bounds encountered")
        minv2[minv == xp.shape[axis]] = xp.shape[axis] - 1
    if np.min(minv) == 0:
        minv2[minv == 0] = 1

    # Get indices for broadcasting arrays
    above = broadcast_indices(xp, minv2, ndim, axis)
    below = broadcast_indices(xp, minv2 - 1, ndim, axis)

    if np.any(x_array < xp[below]):
        warnings.warn("Interpolation point out of data bounds encountered")

    # Create empty output list
    ret = []

    # Calculate interpolation for each variable
    for var in variables:
        # Var needs to be on the *left* of the multiply to ensure that if it's a pint
        # Quantity, it gets to control the operation--at least until we make sure
        # masked arrays and pint play together better. See https://github.com/hgrecco/pint#633
        var_interp = var[below] + (var[above] - var[below]) * (
            (x_array - xp[below]) / (xp[above] - xp[below])
        )

        # Set points out of bounds to fill value.
        var_interp[minv == xp.shape[axis]] = fill_value
        var_interp[x_array < xp[below]] = fill_value

        # Check for input points in decreasing order and return output to match.
        if x[0] > x[-1]:
            var_interp = np.swapaxes(np.swapaxes(var_interp, 0, axis)[::-1], 0, axis)
        # Output to list
        ret.append(var_interp)

    if return_list_always or len(ret) > 1:
        return ret
    else:
        return ret[0]


def broadcast_indices(x, minv, ndim, axis):
    """Calculate index values to properly broadcast index array within data array.
    See usage in interp.
    """
    ret = []
    for dim in range(ndim):
        if dim == axis:
            ret.append(minv)
        else:
            broadcast_slice = [np.newaxis] * ndim
            broadcast_slice[dim] = slice(None)
            dim_inds = np.arange(x.shape[dim])
            ret.append(dim_inds[tuple(broadcast_slice)])
    return tuple(ret)


def get_profile_times(h5file):
    time_offset_seconds = get_geoloc_var(h5file, "Profile_time")
    UTC_start_time_seconds = get_geoloc_var(h5file, "UTC_start")[0]
    start_time_string = h5file["Swath Attributes"]["start_time"][0][0].decode("UTF-8")
    YYYYmmdd = start_time_string[0:8]
    base_time = datetime.strptime(YYYYmmdd, "%Y%m%d")
    UTC_start_offset = timedelta(seconds=UTC_start_time_seconds)
    profile_times = np.array(
        [
            base_time + UTC_start_offset + timedelta(seconds=x)
            for x in time_offset_seconds
        ]
    )

    da = xr.DataArray(
        data=profile_times,
        dims=["nray"],
        coords=dict(
            nray=range(len(profile_times)),
        ),
        attrs=dict(
            description="time",
        ),
    )
    return da


def get_geoloc_var(
    h5file,
    varname,
):
    var_value = h5file["Geolocation Fields"][varname][:]
    var_value = var_value.astype(float)
    factor = h5file["Swath Attributes"][varname + ".factor"][0][0]
    offset = h5file["Swath Attributes"][varname + ".offset"][0][0]
    var_value = (var_value - offset) / factor

    if varname == "DEM_elevation":
        var_value[var_value < 0.0] = 0.0
    if (
        varname == "Height"
        or varname == "DEM_elevation"
        or varname == "Vertical_binsize"
    ):
        # mask missing values
        var_value[
            var_value == h5file["Swath Attributes"][varname + ".missing"][0][0]
        ] = np.nan

    if (
        varname == "Latitude"
        or varname == "Longitude"
        or varname == "Height"
        or varname == "DEM_elevation"
        or varname == "Vertical_binsize"
    ):
        if var_value.ndim == 1:
            da = xr.DataArray(
                data=var_value,
                dims=["nray"],
                coords=dict(
                    nray=range(len(var_value)),
                ),
                attrs=dict(
                    longname=h5file["Swath Attributes"][varname + ".long_name"][0][
                        0
                    ].decode("UTF-8"),
                    units=h5file["Swath Attributes"][varname + ".units"][0][0].decode(
                        "UTF-8"
                    ),
                ),
            )
        if var_value.ndim == 2:
            da = xr.DataArray(
                data=var_value,
                dims=["nray", "nbin"],
                coords=dict(
                    nray=range(var_value.shape[0]), nbin=range(var_value.shape[1])
                ),
                attrs=dict(
                    longname=h5file["Swath Attributes"][varname + ".long_name"][0][
                        0
                    ].decode("UTF-8"),
                    units=h5file["Swath Attributes"][varname + ".units"][0][0].decode(
                        "UTF-8"
                    ),
                ),
            )
        return da

    else:
        return var_value


def get_data_var(h5file, varname):
    var_value = h5file["Data Fields"][varname][:]
    var_value = var_value.astype(float)
    factor = h5file["Swath Attributes"][varname + ".factor"][0][0]
    offset = h5file["Swath Attributes"][varname + ".offset"][0][0]
    var_value = (var_value - offset) / factor

    var_value[
        var_value == h5file["Swath Attributes"][varname + ".missing"][0][0]
    ] = np.nan

    if var_value.ndim == 1:
        da = xr.DataArray(
            data=var_value,
            dims=["nray"],
            coords=dict(
                nray=range(len(var_value)),
            ),
            attrs=dict(
                longname=h5file["Swath Attributes"][varname + ".long_name"][0][
                    0
                ].decode("UTF-8"),
                units=h5file["Swath Attributes"][varname + ".units"][0][0].decode(
                    "UTF-8"
                ),
            ),
        )
    if var_value.ndim == 2:
        da = xr.DataArray(
            data=var_value,
            dims=["nray", "nbin"],
            coords=dict(nray=range(var_value.shape[0]), nbin=range(var_value.shape[1])),
            attrs=dict(
                longname=h5file["Swath Attributes"][varname + ".long_name"][0][
                    0
                ].decode("UTF-8"),
                units=h5file["Swath Attributes"][varname + ".units"][0][0].decode(
                    "UTF-8"
                ),
            ),
        )
    return da


def exponential_fit(x, a, b, c):
    return a * np.exp(b * x) + c

def set4D_latitude_values_nan(array, upper_lat, lower_lat):
    array[:,:,array['latitude'].where(array['latitude']==upper_lat).argmin('latitude').values+1:\
              array['latitude'].where(array['latitude']==lower_lat).argmin('latitude').values,:] = xr.DataArray(data=da.full(shape =(array[:,:,array['latitude'].where(array['latitude']==upper_lat).argmin('latitude').values+1:\
                                                                                                                               array['latitude'].where(array['latitude']==lower_lat).argmin('latitude').values,:]).shape,
                                                                                                             fill_value=np.nan),
                                                                                                dims=(array[:,:,array['latitude'].where(array['latitude']==upper_lat).argmin('latitude').values+1:\
                                                                                                                array['latitude'].where(array['latitude']==lower_lat).argmin('latitude').values,:]).dims, 
                                                                                                coords=(array[:,:,array['latitude'].where(array['latitude']==upper_lat).argmin('latitude').values+1:\
                                                                                                                  array['latitude'].where(array['latitude']==lower_lat).argmin('latitude').values,:]).coords)

    return array

def set3D_latitude_values_nan(array, upper_lat, lower_lat):
    array[:,array['latitude'].where(array['latitude']==upper_lat).argmin('latitude').values+1:\
            array['latitude'].where(array['latitude']==lower_lat).argmin('latitude').values,:] = xr.DataArray(data=da.full(shape =(array[:,array['latitude'].where(array['latitude']==upper_lat).argmin('latitude').values+1:\
                                                                                                                                           array['latitude'].where(array['latitude']==lower_lat).argmin('latitude').values,:]).shape,
                                                                                                             fill_value=np.nan),
                                                                                                dims=(array[:,array['latitude'].where(array['latitude']==upper_lat).argmin('latitude').values+1:\
                                                                                                              array['latitude'].where(array['latitude']==lower_lat).argmin('latitude').values,:]).dims, 
                                                                                                coords=(array[:,array['latitude'].where(array['latitude']==upper_lat).argmin('latitude').values+1:\
                                                                                                                array['latitude'].where(array['latitude']==lower_lat).argmin('latitude').values,:]).coords)

    return array


def plt_seasonal_NH_SH(variable,levels,cbar_label,plt_title, extend):

    f, axsm = plt.subplots(nrows=2,ncols=4,figsize =[10,7], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0,globe=None)})

    coast = cy.feature.NaturalEarthFeature(category='physical', scale='110m',
                            facecolor='none', name='coastline')
    
    for ax, k, season in zip(axsm.flatten()[:4], range(len(fig_label)), variable.season):
        # ax.add_feature(cy.feature.COASTLINE, alpha=0.5)
        ax.add_feature(coast,alpha=0.5)
        ax.set_extent([-180, 180, 90, 45], ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels   = False
        gl.right_labels = False
        variable.sel(season=season, lat=slice(45,90)).plot(ax=ax, transform=ccrs.PlateCarree(), extend=extend, add_colorbar=False,
                            cmap=cm.hawaii_r, levels=levels)
        ax.set(title ='season = {}'.format(season.values))
        
        
        ax.text(0.05, 0.95,
                        fig_label[k],
                        fontweight='bold',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = ax.transAxes)

    for ax, i, k, season in zip(axsm.flatten()[4:], np.arange(5,9), range(len(fig_label))[4:], variable.season):
        ax.remove()
        ax = f.add_subplot(2,4,i, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None))
        # ax.add_feature(cy.feature.COASTLINE, alpha=0.5)
        ax.add_feature(coast,alpha=0.5)
        ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels   = False
        gl.right_labels = False
        cf = variable.sel(season=season, lat=slice(-90,-45)).plot(ax=ax, transform=ccrs.PlateCarree(), extend=extend, add_colorbar=False,
                            cmap=cm.hawaii_r, levels=levels)
        ax.set(title ='season = {}'.format(season.values))
        ax.text(0.05, 0.95,
                        fig_label[k],
                        fontweight='bold',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = ax.transAxes)

    cbaxes = f.add_axes([1.0125, 0.025, 0.025, 0.9])
    cbar = plt.colorbar(cf, cax=cbaxes, shrink=0.5,extend=extend, orientation='vertical', label=cbar_label)
    f.suptitle(plt_title, fontweight="bold");
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
    

def plt_seasonal_diff(variable, levels, cbar_label, plt_title):
    f, axsm = plt.subplots(nrows=2,ncols=4,figsize =[10,7], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0,globe=None)})

    coast = cy.feature.NaturalEarthFeature(category='physical', scale='110m',
                            facecolor='none', name='coastline')
    
    for ax,k , season in zip(axsm.flatten()[:4], range(len(fig_label)),variable.season):
        # ax.add_feature(cy.feature.COASTLINE, alpha=0.5)
        ax.add_feature(coast,alpha=0.5)
        ax.set_extent([-180, 180, 90, 45], ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels   = False
        gl.right_labels = False
        variable.sel(season=season, lat=slice(45,90)).plot(ax=ax, transform=ccrs.PlateCarree(), extend='both', add_colorbar=False,
                            cmap=cm.vik, levels=levels)
        ax.set(title ='season = {}'.format(season.values))
        ax.text(0.05, 0.95,
                        fig_label[k],
                        fontweight='bold',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = ax.transAxes)

    for ax, i, k, season in zip(axsm.flatten()[4:], np.arange(5,9), range(len(fig_label))[4:], variable.season):
        ax.remove()
        ax = f.add_subplot(2,4,i, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None))
        # ax.add_feature(cy.feature.COASTLINE, alpha=0.5)
        ax.add_feature(coast,alpha=0.5)
        ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels   = False
        gl.right_labels = False
        cf = variable.sel(season=season, lat=slice(-90,-45)).plot(ax=ax, transform=ccrs.PlateCarree(), extend='both', add_colorbar=False,
                            cmap=cm.vik,levels=levels)
        ax.set(title ='season = {}'.format(season.values))
        ax.text(0.05, 0.95,
                        fig_label[k],
                        fontweight='bold',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = ax.transAxes)

    cbaxes = f.add_axes([1.0125, 0.025, 0.025, 0.9])
    cbar = plt.colorbar(cf, cax=cbaxes, shrink=0.5,extend='both', orientation='vertical', label=cbar_label)
    f.suptitle(plt_title, fontweight="bold");
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
    
    
# Radius and area of earth equations
def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or lats in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of lat in degrees
    lon: vector of lon in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda


def plt_annual_cycle(NH_mean, SH_mean, NH_std, SH_std, y_label,plt_title):
    f, axsm = plt.subplots(nrows=1,ncols=2,figsize =[10,3.5], sharex=True, sharey=True)

    axs = axsm.flatten()
    # for c in cm.hawaii_r(range(0,256, int(256/8))):
    NH_mean.plot.line(ax=axs[0], x='month', add_legend=False, ylim=[0,1], xlim=[1,12], color=[0.703779, 0.948977, 0.993775, 1.      ])
    SH_mean.plot.line(ax=axs[1], x='month', add_legend=False, ylim=[0,1], xlim=[1,12], color=[0.703779, 0.948977, 0.993775, 1.      ])

    # for year in np.unique(ds_era['time.year']):
    #     axs[0].scatter(NH_mean['lcc_wo_snow_month_'+str(year)].month.values, NH_mean['lcc_wo_snow_month_'+str(year)].values, marker='x')
    #     axs[1].scatter(SH_mean['lcc_wo_snow_month_'+str(year)].month.values, SH_mean['lcc_wo_snow_month_'+str(year)].values, marker='x')

    axs[0].fill_between(NH_mean.month,
                    NH_mean - NH_std,
                    NH_mean + NH_std,
                    alpha=0.3, color=[0.703779, 0.948977, 0.993775, 1.      ])

    axs[1].fill_between(SH_mean.month,
                    SH_mean - SH_std,
                    SH_mean + SH_std,
                    alpha=0.3, color=[0.703779, 0.948977, 0.993775, 1.      ])
    for i in range(len(axsm)):
        axs[i].text(0.05, 0.95,
                        fig_label[i],
                        fontweight='bold',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = axs[i].transAxes)
        axs[i].set_xticks(np.arange(1,13))
        axs[i].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axs[i].grid(alpha=0.3)
        axs[i].set_xlabel('')

    axs[0].set_ylabel(y_label)
    axs[1].set_ylabel('')
    axs[0].set(title ='Northern Hemisphere')
    axs[1].set(title ='Southern Hemisphere')


    f.suptitle(plt_title, fontweight="bold");
    axs[1].legend(['ERA5'], bbox_to_anchor=(.71, 1.01, -1., .102), loc='lower left', ncol = 1)
    plt.tight_layout(pad=0.15, w_pad=0.15, h_pad=0.15)
    
    
def to_ERA5_date(ds, model):
    # remove leap day from dataset  
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 30)))
    
    # if ds.time.dtype == 'datetime64[ns]':
    #     print(model,ds.time[0].values)
    if ds.time.dtype == 'object':
        # print(model, ds.time[0].values)
        ds['time'] = ds.indexes['time'].to_datetimeindex()
    
   
    
    dates = ds.time.values
    years = dates.astype('datetime64[Y]').astype(int)+1971 # add a year to be similar to ERA5
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1
    days = (dates.astype('datetime64[D]') - dates.astype('datetime64[M]')).astype(int) + 1

    
    data = np.array([list(years),list(months), list(days),[0] * len(list(years)) ])
    # Transpose the data so that columns become rows.
    data = data.T
    data = data.tolist()
    # A simple list comprehension does the trick, '*' making sure
    # the values are unpacked for 'datetime.datetime'.
    new_data = [datetime(*x) for x in data]
    # We assign the new time coordinate
    ds = ds.assign_coords({'time':new_data})
    
    return ds

def is_season(month, lower_val, upper_val):
    return (month>=lower_val) & (month <= upper_val)

def calculate_interannual_stats(dataset, var_name):
    _data = xr.concat([dataset[var_name + '_month_years_mean'],
                       dataset[var_name + '_year_years_mean'].assign_coords(coords={'month': 13})], dim='month')
    quantiles = _data.quantile([0.25, 0.5, 0.75], dim='year', skipna=True, keep_attrs=False)
    iqr = quantiles.sel(quantile=0.75) - quantiles.sel(quantile=0.25)
    max_val = (quantiles.sel(quantile=0.75) + 1.5 * iqr).assign_coords({'quantile': 'max'})
    min_val = (quantiles.sel(quantile=0.25) - 1.5 * iqr).assign_coords({'quantile': 'min'})
    return xr.concat([max_val, quantiles, min_val], dim='quantile')

#### functions for plt_seasonal_mean.py
def calculate_stats(data, weights, lat_slice):
        weighted_data = data.sel(lat=lat_slice).weighted(weights)
        mean = weighted_data.mean(('lat', 'lon'), skipna=True, keep_attrs=False)
        std = weighted_data.std(('lat', 'lon'), skipna=True, keep_attrs=False)
        if 'quantile' not in data.coords:
                
                quantiles = weighted_data.quantile([0.25, 0.5, 0.75], dim=('lat', 'lon'), skipna=True, keep_attrs=False)
                iqr = quantiles.sel(quantile=0.75) - quantiles.sel(quantile=0.25)
                max_val = (quantiles.sel(quantile=0.75) + 1.5 * iqr).assign_coords({'quantile': 'max'})
                min_val = (quantiles.sel(quantile=0.25) - 1.5 * iqr).assign_coords({'quantile': 'min'})
                stats = xr.concat([max_val, quantiles, min_val], dim='quantile')
                
        else:
                stats = xr.DataArray()
        
        return mean, std, stats
    
def weighted_average(data, weights):

    lat_north = slice(45, 90)
    lat_south = slice(-90, -45)
    
    NH_mean, NH_std, NH_stats = calculate_stats(data, weights, lat_north)
    SH_mean, SH_std, SH_stats = calculate_stats(data, weights, lat_south)
    
    mean = xr.concat([NH_mean, SH_mean], pd.Index(['NH', 'SH'], name="hemisphere"))
    std = xr.concat([NH_std, SH_std], pd.Index(['NH', 'SH'], name="hemisphere"))
    stats = xr.concat([NH_stats, SH_stats], pd.Index(['NH', 'SH'], name='hemisphere'))
    
    return mean, std, stats


def get_ratios_season_month(var1, var2, stats, out_var,seasons ):
    
    ratios = xr.Dataset()
    years = [2007, 2008, 2009, 2010]
    months = np.arange(1, 13)
    
    season_mapping = {
        'normal': {'JJA': (6, 7, 8), 'MAM': (3, 4, 5), 'SON': (9, 10, 11)},
        'alternative': {'MJJ': (5, 6, 7), 'FMA': (2, 3, 4), 'ASO': (8, 9, 10)}
        }

    selected_seasons = season_mapping[seasons]
    fseason = {season: var1.sel(time=is_season(var1['time.month'], months[0], months[2])) for season, months in selected_seasons.items()}
    if seasons == 'normal':
        fseason['DJF'] = xr.concat([var1.sel(time=is_season(var1['time.month'], 12, 12)),
                                    var1.sel(time=is_season(var1['time.month'], 1, 2))], dim='time')
    else:
        fseason['NDJ'] = xr.concat([var1.sel(time=is_season(var1['time.month'], 11, 12)),
                                    var1.sel(time=is_season(var1['time.month'], 1, 1))], dim='time')
    
    myKeys = list(fseason.keys())
    myKeys.sort()
    sorted_dict = {i: fseason[i] for i in myKeys}
    fseason = xr.concat(list(sorted_dict.values()), pd.Index(list(sorted_dict.keys()), name='season'))
        
    # fmonth = var1.groupby('time.month')  
    fmonth = xr.concat([var1.sel(time=is_season(var1['time.month'], m, m)) for m in months], pd.Index(months, name='month'))
    
    if 'time' in var2.dims:
        
            # nseason = var2.groupby('time.season')
        nseason = {season: var2.sel(time=is_season(var2['time.month'], months[0], months[2])) for season, months in selected_seasons.items()}
        if seasons == 'normal':
            nseason['DJF'] = xr.concat([var2.sel(time=is_season(var1['time.month'], 12, 12)),
                                        var2.sel(time=is_season(var1['time.month'], 1, 2))], dim='time')   
        else:
            nseason['NDJ'] = xr.concat([var2.sel(time=is_season(var1['time.month'], 11, 12)),
                                        var2.sel(time=is_season(var1['time.month'], 1, 1))], dim='time') 
        myKeys = list(nseason.keys())
        myKeys.sort()
        sorted_dict = {i: nseason[i] for i in myKeys}
        nseason = xr.concat(list(sorted_dict.values()), pd.Index(list(sorted_dict.keys()), name='season'))
        # nmonth = var2.groupby('time.month')
        nmonth = xr.concat([var2.sel(time=is_season(var1['time.month'], m, m)) for m in months], pd.Index(months, name='month'))

    if stats == 'count':
        # per season
        fseason_year = xr.concat([fseason.sel(time=fseason.time.dt.year.isin(year)).squeeze().count(dim='time', keep_attrs=False) for year in years],
                                 pd.Index(years, name='year'))
        nseason_year = xr.concat([nseason.sel(time=nseason.time.dt.year.isin(year)).squeeze().count(dim='time', keep_attrs=False) for year in years],
                                 pd.Index(years, name='year'))

        ratios[out_var+'_season'] = np.true_divide(fseason.count(dim='time', keep_attrs=False), nseason.count(dim='time', keep_attrs=False))
        ratios[out_var+'_season_years'] = np.true_divide(fseason_year, nseason_year)
    

        # per month
        fmonth_year = xr.concat([fmonth.sel(time=fmonth.time.dt.year.isin(year)).squeeze().count(dim='time', keep_attrs=False) for year in years], 
                                pd.Index(years, name='year'))
        nmonth_year = xr.concat([nmonth.sel(time=nmonth.time.dt.year.isin(year)).squeeze().count(dim='time', keep_attrs=False) for year in years], 
                                pd.Index(years, name='year'))

        ratios[out_var+'_month'] = np.true_divide(fmonth.count(dim='time', keep_attrs=False), nmonth.count(dim='time', keep_attrs=False))
        ratios[out_var+'_month_years'] = np.true_divide(fmonth_year, nmonth_year)
                
        # all years
        ratios[out_var+'_year'] = np.true_divide(var1.count(dim='time', keep_attrs=False), var2.count(dim='time', keep_attrs=False))
        ratios[out_var+'_year_years'] = np.true_divide(var1.groupby('time.year').count(dim='time', keep_attrs=False), 
                                                       var2.groupby('time.year').count(dim='time', keep_attrs=False))
        #     ratios[out_var+'_year'] = var1.count(dim='time', keep_attrs=False) / var3.count(dim='time', keep_attrs=False)
    
    elif stats == 'sum':
        # per season
        fseason_year = xr.concat([fseason.sel(time=fseason.time.dt.year.isin(year)).squeeze().sum(dim='time', skipna=True, keep_attrs=False) for year in years],
                                 pd.Index(years, name='year'))
        nseason_year = xr.concat([nseason.sel(time=nseason.time.dt.year.isin(year)).squeeze().sum(dim='time', skipna=True, keep_attrs=False) for year in years],
                                 pd.Index(years, name='year'))
        
        ratios[out_var +'_season'] = np.true_divide(fseason.sum(dim='time', skipna=True, keep_attrs=False), nseason.sum(dim='time', skipna=True, keep_attrs=False))
        ratios[out_var+'_season_years'] = np.true_divide(fseason_year, nseason_year)
        
        # per month
        fmonth_year = xr.concat([fmonth.sel(time=fmonth.time.dt.year.isin(year)).squeeze().sum(dim='time', skipna=True, keep_attrs=False) for year in years], 
                                pd.Index(years, name='year'))
        nmonth_year = xr.concat([nmonth.sel(time=nmonth.time.dt.year.isin(year)).squeeze().sum(dim='time', skipna=True, keep_attrs=False) for year in years], 
                                pd.Index(years, name='year'))
        ratios[out_var +'_month']  = np.true_divide(fmonth.sum(dim='time', skipna=True, keep_attrs=False), nmonth.sum(dim='time', skipna=True, keep_attrs=False))
        ratios[out_var+'_month_years'] = np.true_divide(fmonth_year, nmonth_year)
        
        # all years
        ratios[out_var +'_year'] = np.true_divide(var1.sum(dim='time', skipna=True, keep_attrs=False), (var2.sum(dim='time', skipna=True, keep_attrs=False)))
        ratios[out_var+'_year_years'] = np.true_divide(var1.groupby('time.year').sum(dim='time', skipna=True, keep_attrs=False),
                                                       var2.groupby('time.year').sum(dim='time', skipna=True, keep_attrs=False))

    elif stats == 'mean':
        # per season
        fseason_year = xr.concat([fseason.sel(time=fseason.time.dt.year.isin(year)).squeeze() for year in years],
                                 pd.Index(years, name='year'))
        nseason_year = xr.concat([nseason.sel(time=nseason.time.dt.year.isin(year)).squeeze() for year in years],
                                 pd.Index(years, name='year'))
        
        ratios[out_var+'_season'] = (np.true_divide(fseason,nseason)).mean(dim='time',skipna=True, keep_attrs=False)
        ratios[out_var+'_season_years'] = np.true_divide(fseason_year, nseason_year).mean(dim='time',skipna=True, keep_attrs=False)
        
        # per month
        fmonth_year = xr.concat([fmonth.sel(time=fmonth.time.dt.year.isin(year)).squeeze() for year in years],
                                 pd.Index(years, name='year'))
        nmonth_year = xr.concat([nmonth.sel(time=nmonth.time.dt.year.isin(year)).squeeze() for year in years],
                                 pd.Index(years, name='year'))

        ratios[out_var+'_month'] = (np.true_divide(fmonth,nmonth)).mean(dim='time', skipna=True, keep_attrs=False)
        ratios[out_var+'_month_years'] = (np.true_divide(fmonth_year, nmonth_year)).mean(dim='time', skipna=True, keep_attrs=False)
        # ratios[out_var+'_month'] = (np.true_divide(var1,var2)).groupby('time.month').mean(dim='time', skipna=True, keep_attrs=False)
        
        # all years
        fyear = xr.concat([var1.sel(time=var1.time.dt.year.isin(year)).squeeze() for year in years],
                          pd.Index(years, name='year'))
        nyear = xr.concat([var2.sel(time=var2.time.dt.year.isin(year)).squeeze() for year in years],
                          pd.Index(years, name='year'))
        
        ratios[out_var+'_year'] = (np.true_divide(var1,var2)).mean(dim='time', skipna=True, keep_attrs=False)
        ratios[out_var+'_year_years'] = (np.true_divide(fyear, nyear)).mean(dim='time', skipna=True, keep_attrs=False)
        
    # ratios[out_var +'_season'] = ratios[out_var +'_season'].where(ratios[out_var +'_season'] > 0., other = np.nan)
    # ratios[out_var +'_month'] = ratios[out_var +'_month'].where(ratios[out_var +'_month'] > 0., other = np.nan)
    # ratios[out_var +'_year'] = ratios[out_var +'_year'].where(ratios[out_var +'_year'] > 0., other = np.nan)
    
    if out_var != 'sf_eff':
        ratios[out_var +'_season'] = ratios[out_var +'_season']*100
        ratios[out_var +'_season_years'] = ratios[out_var +'_season_years']*100
        ratios[out_var +'_month'] = ratios[out_var +'_month']*100
        ratios[out_var +'_month_years'] = ratios[out_var +'_month_years']*100
        ratios[out_var +'_year'] = ratios[out_var +'_year']*100
        ratios[out_var +'_year_years'] = ratios[out_var +'_year_years']*100
    # for vars in ratios.keys():
    #     ratios[vars+'_mean'], ratios[vars+'_std'], ratios[vars+'_stats'] = weighted_average(ratios[vars], weights)
    ratios[out_var+'_season'] = ratios[out_var+'_season'].where(~np.isinf(ratios[out_var+'_season']), other=0.)
    ratios[out_var +'_season_years'] = ratios[out_var +'_season_years'].where(~np.isinf(ratios[out_var +'_season_years']), other=0.)
                
    ratios[out_var+'_month'] = ratios[out_var+'_month'].where(~np.isinf(ratios[out_var+'_month']), other=0.)
    ratios[out_var +'_month_years'] = ratios[out_var +'_month_years'].where(~np.isinf(ratios[out_var +'_month_years']), other=0.)
    # all years
    ratios[out_var +'_year'] = ratios[out_var +'_year'].where(~np.isinf(ratios[out_var +'_year']), other=0.)
    ratios[out_var +'_year_years'] = ratios[out_var +'_year_years'].where(~np.isinf(ratios[out_var +'_year_years']), other=0.)
    return(ratios)

def get_ratios_dict(list_models, ds,seasons):
    # if seasons == 'normal':
    #     seasons = seasons
    # else:
    #     seasons = None
    ratios = {}
    for model in list_models:
        if model == 'cloudsat_250' or model == 'cloudsat_500':
            ds['lcc_2t_days'][model]['lwp'] = xr.DataArray(np.nan, coords=ds['lcc_2t_days'][model]['sf_avg_lcc_snow'].coords, dims=ds['lcc_2t_days'][model]['sf_avg_lcc_snow'].dims)
            ratios[model] = xr.merge(objects = [
                # get_ratios_season_month(var1 = ds['lcc_2t'][model]['n_lcc'], var2 = ds['2t'][model]['n_cld'], stats = 'sum', out_var = 'lcc_wo_snow', weights = ds['2t'][model]['areacella']),
                get_ratios_season_month(var1 = ds['orig'][model]['n_lcc'], var2 = ds['orig'][model]['n_obs'],  
                                        stats ='sum',   
                                        out_var = 'FLCC', 
                                        seasons=seasons),
                get_ratios_season_month(var1 = ds['lcc_2t'][model]['n_lcc'], var2 = ds['orig'][model]['n_obs'],  
                                        stats = 'sum',  
                                        out_var = 'FsLCC', 
                                        seasons=seasons), #out_var = 'lcc_wo_snow', weights = ds['2t'][model]['areacella']),
                # # get_ratios_season_month(var1 = ds['lcc_sf'][model]['n_sf_lcc_snow'], var2 = ds['orig'][model]['n_lcc'], 
                # #                         stats = 'sum', 
                # #                         out_var = 'FoP', 
                # #                         seasons=seasons),
                # get_ratios_season_month(var1 = ds['lcc_2t_sf'][model]['n_sf_lcc_snow'], var2 = ds['lcc_2t'][model]['n_lcc'], 
                #                         stats='sum', 
                #                         out_var = 'FoS', 
                #                         seasons=seasons), #out_var='lcc_w_snow', weights=ds['lcc_2t'][model]['areacella']),
                # get_ratios_season_month(var1 = ds['lcc_2t_days'][model]['sf_avg_lcc_snow'], var2 = ds['lcc_2t_days'][model]['lwp'], 
                #                         stats = 'mean', 
                #                         out_var = 'sf_eff', 
                #                         seasons=seasons),
                # # get_ratios_season_month(var1 = ds['lcc_2t_days'][model]['sf_avg_lcc_snow'], var2 = ds['lcc_2t_days'][model]['lwp'], 
                # #                         stats = 'mean', 
                # #                         out_var = 'pr_eff', 
                # #                         seasons=seasons),
                get_ratios_season_month(var1 = ds['orig'][model]['n_lcc']-ds['lcc_2t'][model]['n_lcc'],
                                        var2 = ds['orig'][model]['n_obs'], 
                                        stats = 'sum', 
                                        out_var = 'FLCC-FsLCC', 
                                        seasons=seasons)

            ])
        else:
            ratios[model] = xr.merge(objects=[
                # get_ratios_season_month(var1=ds['lcc_2t'][model]['lwp'], var2=ds['2t'][model]['twp'].where(ds['2t'][model]['twp']>0.), stats='count', out_var='lcc_wo_snow', weights=ds['2t'][model]['areacella']),              # relative frequency of liquid containing clouds in relation to when there is a cloud
                # get_ratios_season_month(var1=ds['lcc_2t'][model]['lwp'], var2=ds['2t'][model]['tas'], stats='count', out_var='lcc_wo_snow', weights=ds['2t'][model]['areacella']), # sLCC frequency compared to all observations when T<0C
                ## use of 'tas' in var2 as this has values everywhere where data is valid, while 'lwp' or 'prsn' might not have values
                get_ratios_season_month(var1=ds['lcc'][model]['lwp'], var2=ds['orig'][model]['tas'], 
                                        stats='count', 
                                        out_var='FLCC', 
                                        seasons=seasons),
                get_ratios_season_month(var1=ds['lcc_2t'][model]['lwp'], var2=ds['orig'][model]['tas'], 
                                        stats='count', 
                                        out_var='FsLCC', 
                                        seasons=seasons),#out_var='lcc_wo_snow', weights=ds['2t'][model]['areacella']), # sLCC frequency compared to all observations when T<0C
                # # get_ratios_season_month(var1=ds['lcc'][model]['pr'].where(ds['lcc'][model]['pr']>=0.01, other=np.nan), 
                # #                         var2=ds['orig'][model]['tas'], 
                # #                         stats='count', 
                # #                         out_var='FoP', 
                # #                         seasons=seasons),
                # get_ratios_season_month(var1=ds['lcc_2t_sf'][model]['prsn'], var2=ds['lcc_2t'][model]['lwp'], 
                #                         stats='count', 
                #                         out_var='FoS', 
                #                         seasons=seasons), #out_var='lcc_w_snow', weights=ds['lcc_2t'][model]['areacella']),   # relative frequency of snowfall from liquid containing clouds
                # get_ratios_season_month(var1=ds['lcc_2t_days'][model]['prsn'], var2=ds['lcc_2t_days'][model]['lwp'], 
                #                         stats='mean', 
                #                         out_var='sf_eff', 
                #                         seasons=seasons),      # relative snowfall (precipitation) efficency
                # # get_ratios_season_month(var1=ds['lcc_2t_days'][model]['pr'], var2=ds['lcc_2t_days'][model]['lwp'], 
                # #                         stats='mean', 
                # #                         out_var='pr_eff', 
                # #                         seasons=seasons),      # relative snowfall (precipitation) efficency
                get_ratios_season_month(var1=ds['lcc'][model]['lwp']-ds['lcc_2t'][model]['lwp'],
                                        var2=ds['orig'][model]['tas'],
                                        stats='count',
                                        out_var='FLCC-FsLCC',
                                        seasons=seasons)
            
            ])
    return(ratios)

def get_only_valid_values(ratios, res, out_var):
        # for time in ['season', 'month', 'year']:
        # for out_var, time in product(d.keys(), times):
        #     for cs_key, era_key, cmip_key in zip(cloudsat_keys, era_keys, cmip_keys):
        cs_key = f'cloudsat_{res}'
        era_key = f'era_{res}'
        cmip_key = f'cmip_{res}'
        
        # v1_250 = ratios[cs_key][f'{out_var}_{time}']
        v1_250 = ratios[cs_key][f'{out_var}']
        # v1_250 = v1_250.where(v1_250 != 0., other = np.nan)
        
        v2_250 = ratios[era_key][f'{out_var}']
        # v2_250 = v2_250.where(v2_250 != 0., other = np.nan)

        
        v3_250 = ratios[cmip_key][f'{out_var}']
        # v3_250 = v3_250.where(v3_250 != 0., other = np.nan)

        # v1_era_250 = v1_250.copy()
        # # v1_era_250 = v1_era_250.where(~np.isnan(v2_250))  
        # ratios[cs_key][f'{out_var}_{time}_era'] = v1_era_250
        

        # v1_cmip_250 = v1_250.copy()
        # # v1_cmip_250 = v1_cmip_250.where(~np.isnan(v3_250))
        # # v1_cmip_250 = v1_cmip_250.mean('model',skipna=True)
        # ratios[cs_key][f'{out_var}_{time}_cmip'] = v1_cmip_250
        v1_250_cs = v1_250.copy()
        v1_250_cs = v1_250_cs.where(np.logical_and(v1_250_cs.lat <= 82., v1_250_cs.lat >= -82), other=np.nan)
        ratios[cs_key][f'{out_var}_cs'] = v1_250_cs

        # v2_250_cs = v2_250.copy()
        # v2_250_cs = v2_250_cs.where(~np.isnan(v1_250))
        # ratios[era_key][f'{out_var}_{time}_cs'] = v2_250_cs
        v2_250_cs = v2_250.copy()
        v2_250_cs = v2_250_cs.where(np.logical_and(v2_250_cs.lat <= 82., v2_250_cs.lat >= -82), other=np.nan)
        ratios[era_key][f'{out_var}_cs'] = v2_250_cs

        # v3_250_cs = v3_250.copy()
        # v3_250_cs = v3_250_cs.where(~np.isnan(v1_250))
        # ratios[cmip_key][f'{out_var}_{time}_cs'] = v3_250_cs
        v3_250_cs = v3_250.copy()
        v3_250_cs = v3_250_cs.where(np.logical_and(v3_250_cs.lat <= 82., v3_250_cs.lat >= -82.), other=np.nan)
        ratios[cmip_key][f'{out_var}_cs'] = v3_250_cs
        
        
        
            
        # if out_var == 'sf_eff' or out_var == 'pr_eff':
        #     v1_cmip_250 = v2_250.copy()
        #     v1_cmip_250 = v1_cmip_250.where(~np.isnan(v3_250))
        #     v1_cmip_250 = v1_cmip_250.mean('model', skipna=True)
        #     ratios[era_key][f'{out_var}_{time}_cmip'] = v1_cmip_250
            
        #     v2_250_era = v2_250.copy()
        #     v2_250_era = v2_250_era.where(~np.isnan(v2_250))
        #     ratios[era_key][f'{out_var}_{time}_era'] = v2_250_era
            
        #     v2_250_cmip = v3_250.copy()
        #     v2_250_cmip = v2_250_cmip.where(~np.isnan(v3_250))
        #     ratios[cmip_key][f'{out_var}_{time}_cmip'] = v2_250_cmip
            
        #     v3_250_era = v3_250.copy()
        #     v3_250_era = v3_250_era.where(~np.isnan(v2_250))
        #     ratios[cmip_key][f'{out_var}_{time}_era'] = v3_250_era
            

        return (ratios)

def create_projection(hemisphere):
    projections = {'NH': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None),
                   'SH': ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)}

    return(projections[hemisphere])

def create_colorbar_axes(fig, var_name):
    if var_name == 'sf_eff' or var_name == 'pr_eff':
        return fig.add_axes([0.92, 0.4, 0.0125, 0.45])
    else:
        return fig.add_axes([0.92, 0.65, 0.0125, 0.225])
    
def plot_difference_significance(ax, hemisphere, diff, season, CI, density):
    diff_slice = diff.sel(lat=slice(45, 90)) if hemisphere == 'NH' else diff.sel(lat=slice(-90, -45))
    significance = np.abs(diff_slice) < CI
    ax.contourf(
            diff_slice.lon, diff_slice.lat, diff_slice.where(significance).sel(season=season), transform=ccrs.PlateCarree(),
            colors='none', hatches=[density * '/', density * '/'], add_colorbar=False
        )
    ax.set_title('')

def add_text_box(ax, value, var_name):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    val = value.values.round(2)
    if var_name == 'sf_eff' or var_name == 'pr_eff':
        text = f'{val:.1f} h$^{-1}$'
    else:
        text = f'{val:.0f}%'
    ax.text(0.05, 0.125, text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    
def setup_axes(ax, hemisphere, lat_extent):
    if hemisphere == 'NH':
        ax.set_extent([-180, 180, 90, lat_extent], ccrs.PlateCarree())
    elif hemisphere == 'SH':
        ax.set_extent([-180, 180, -90, -1 * lat_extent], ccrs.PlateCarree())
    coast = cy.feature.NaturalEarthFeature(category='physical', scale='110m', facecolor='none', name='coastline')
    ax.add_feature(coast, alpha=0.5)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    

 
def plt_monthly_model_variation(ds_dict, var_name, dict_label,fig_dir, lwp_threshold):
    colors = cm.hawaii(range(0, 256, int(256 / 3) + 1))

    
    
    f, axsm = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=[12, 5])
    f.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=1.005)
    ax = axsm.flat

    bp= [[],]
    for i, hemisphere in enumerate(['NH', 'SH']):
        # ax[i].hlines(0., 0.25, 12.75, colors='k')
        ax[i].grid(True)
        cs_data = xr.concat([ds_dict[var_name + '_month_cs_mean'].sel(hemisphere=hemisphere, model='CloudSat'),
                             ds_dict[var_name + '_year_cs_mean'].sel(hemisphere=hemisphere, model='CloudSat').assign_coords(coords={'month':13})], dim='month')
        ax[i].scatter(x=np.arange(1,14), y=cs_data, color='k', marker='o',s=50)
        
        # era_data = ds_dict['era_30'][var_name + '_month_mean'].sel(hemisphere=hemisphere,)
        # cmip_key = 'cmip_500'
        
        if var_name == 'sf_eff' or var_name == 'pr_eff':
            era_data = xr.concat([ds_dict[var_name + '_month_mean'].sel(hemisphere=hemisphere, model='ERA5'),
                                  ds_dict[var_name + '_year_mean'].sel(hemisphere=hemisphere, model='ERA5').assign_coords(coords={'month':13})],dim='month')
            cmip_data = xr.concat([ds_dict[var_name + '_month_mean'].sel(hemisphere=hemisphere, model= ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                                                        'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                                                        'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']),
                                   ds_dict[var_name + '_year_mean'].sel(hemisphere=hemisphere, model = ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                                                        'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                                                        'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']).assign_coords(coords={'month':'years'})], 
                                  dim='month')
        else:
            era_data = xr.concat([ds_dict[var_name + '_month_cs_mean'].sel(hemisphere=hemisphere, model='ERA5'),
                                  ds_dict[var_name + '_year_cs_mean'].sel(hemisphere=hemisphere, model='ERA5').assign_coords(coords={'month':13})], dim='month')
            cmip_data = xr.concat([ds_dict[var_name + '_month_cs_mean'].sel(hemisphere=hemisphere, model= ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                                                           'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                                                           'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']),
                                   ds_dict[var_name + '_year_cs_mean'].sel(hemisphere=hemisphere, model= ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                                                                    'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                                                                    'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']).assign_coords(coords={'month':'years'})], 
                                  dim='month')

        ax[i].scatter(x=np.arange(1.25,14.25), y=era_data, color=colors[0], marker="h", s=50)  
        
        
        j = 0
        
        color = colors[2]
        
        quantiles = cmip_data.quantile([0.25, 0.5, 0.75], dim=('model'), skipna=True, keep_attrs=False)
        iqr = quantiles.sel(quantile=0.75) - quantiles.sel(quantile=0.25)
        max_val = (quantiles.sel(quantile=0.75) + 1.5 * iqr).assign_coords({'quantile': 'max'})
        min_val = (quantiles.sel(quantile=0.25) - 1.5 * iqr).assign_coords({'quantile': 'min'})
        # means = ds_dict[cmip_key][var_name + '_month_mean'].mean(dim='model', skipna=True).assign_coords({'quantile': 'mean'})
        means = cmip_data.mean(dim='model', skipna=True).assign_coords({'quantile':'mean'})
        stats = xr.concat([max_val, quantiles, min_val, means], dim='quantile')
           
        if j == 0:
                positions = np.arange(j + 0.75, j + 13.75, 1)
        else:
                positions = np.arange(j + 0.25, j + 13.25, 1)
        boxplot_data = stats.transpose('quantile', 'month')
        bp[j] = ax[i].boxplot(boxplot_data,  positions=positions, widths=0.4, 
                            boxprops=dict(color=color, lw=1.5),
                            medianprops=dict(color=color, lw=1.5),
                            whiskerprops=dict(color=color, lw=1.5),
                            capprops=dict(color=color, lw=1.5),
                            flierprops=dict(marker='+',markeredgecolor=color, markersize=10),
                            showmeans=True, meanprops=dict(marker='D',markerfacecolor=color, markersize=4),
                            patch_artist=True,)
            
        for patch in bp[j]['boxes']:
                patch.set(facecolor=color, alpha=0.5)
            
            
          
            
        ax[i].set_title('Northern Hemisphere' if hemisphere == 'NH' else 'Southern Hemisphere')   
        ax[i].text(0.05, 0.95, f'{fig_label[i]}', fontweight='bold', horizontalalignment='left', verticalalignment='top', transform=ax[i].transAxes)
        ax[i].set_xticks(np.arange(1,14)) 
        ax[i].set_xlim([0, 13.5])


        ax[i].set_xticklabels(np.append(np.arange(1,13), 'years'), fontsize=12)
        ax[i].set_xlabel('Month')
        
        if var_name == 'sf_eff':# or var_name == 'pr_eff':
            ax[i].set_ylim([0,8.])
            ax[i].set_yticks(np.arange(0,8.50,.50))
        elif var_name == 'pr_eff':
            ax[i].set_ylim([0,800])
            ax[i].set_yticks(np.arange(0,850,50))
        else:
            ax[i].set_ylim([dict_label['vmin'],dict_label['vmax']])
            ax[i].set_yticks(np.arange(0,110,10))
        
        ax[i].set_ylabel(dict_label['cb_label'] if i==0 else '') 
        
    s = f.subplotpars
    bb = [s.left, s.top - 0.92, (s.right - s.left), 0.05]

    ax[1].legend([
            # Line2D([0], [0], color=colors[0], lw=1.5, label='ERA5 (30 km)', linestyle=(0, (1, 1))),
            Line2D([0], [0], marker='o', color='w', label='CloudSat (500km)', markersize=10, markerfacecolor='k'),
            Line2D([0], [0], marker='h', color='w', label='ERA5 (500 km)',markersize=10, markerfacecolor=colors[0], ),
            bp[0]["boxes"][0],
            # bp[1]["boxes"][0],
        ],
        ['CloudSat (500 km)',
            'ERA5 (500 km)', #'CMIP6 (500 km)', 
         'CMIP6 (500 km)'],
        bbox_to_anchor=bb,loc=8,ncol=3,mode='expand',borderaxespad=0,fancybox=True,bbox_transform=f.transFigure,
    )
    
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.)  ;
    
    figname = f'{var_name}_monthly_model_variation_2007_2010.png'
    plt.savefig(fig_dir + figname, format='png', bbox_inches='tight', transparent=True)  
    
def plt_monthly_interannual_variation(dataset, var_name, lwp_threshold, fig_dir, dict_label):
    # calculate statistics
    stats_cloudsat = calculate_interannual_stats(dataset.sel(model='CloudSat'), var_name)
    stats_era = calculate_interannual_stats(dataset.sel(model='ERA5'), var_name)
    stats_cmip = calculate_interannual_stats(dataset.sel(model = ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                  'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                  'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']), 
                                             var_name)
    stats_cmip = stats_cmip.mean('model', skipna=True)
    
    # add cmip shading for max, min model all years
    cmip_data = xr.concat([dataset[var_name+'_month_cs_mean'].sel(model = ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                           'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                           'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']),
                           dataset[var_name + '_year_cs_mean'].sel(model = ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                            'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                            'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']).assign_coords(coords={'month':'years'})], dim='month')
    quantiles = cmip_data.quantile([0.25, 0.5, 0.75], dim=('model'), skipna=True, keep_attrs=False)
    iqr = quantiles.sel(quantile=0.75) - quantiles.sel(quantile=0.25)
    max_val = (quantiles.sel(quantile=0.75) + 1.5 * iqr).assign_coords({'quantile': 'max'})
    min_val = (quantiles.sel(quantile=0.25) - 1.5 * iqr).assign_coords({'quantile': 'min'})
    #     # means = ds_dict[cmip_key][var_name + '_month_mean'].mean(dim='model', skipna=True).assign_coords({'quantile': 'mean'})
    #     means = cmip_data.mean(dim='model', skipna=True).assign_coords({'quantile':'mean'})
    #     stats = xr.concat([max_val, quantiles, min_val, means], dim='quantile')
    
    
    colors = cm.hawaii(range(0, 256, int(256 / 3) + 1))

    f, axsm = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=[12, 5])
    f.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=1.005)
    ax = axsm.flat

    bp = [[], [], []]
    for (j, data), x_loc, color in zip(enumerate([stats_cloudsat, stats_era, stats_cmip]), [np.arange(1,14), np.arange(0.75, 13.75), np.arange(1.25, 14.25)], ['k', colors[0], colors[2]]):
    # for (j, data), x_loc, color in zip(enumerate([stats_cloudsat, ]), [np.arange(1,14), np.arange(0.75, 13.75), np.arange(1.25, 14.25)], ['k', ]):
        for (i, hemisphere) in enumerate(['NH', 'SH']):
            ax[i].grid(True)
            boxplot_data = (data.sel(hemisphere=hemisphere)).transpose('quantile', 'month')
            
            bp[j] = ax[i].boxplot(boxplot_data,  positions=x_loc, widths=0.3, 
                                        boxprops=dict(color=color, lw=1.5),
                                        medianprops=dict(color=color, lw=1.5),
                                        whiskerprops=dict(color=color, lw=1.5),
                                        capprops=dict(color=color, lw=1.5),
                                        flierprops=dict(marker='+',markeredgecolor=color, markersize=10),
                                        showmeans=True, meanprops=dict(marker='D',markerfacecolor=color, alpha=0.5, markeredgecolor=color,markersize=4),
                                        patch_artist=True,)
            for patch in bp[j]['boxes']:
                    patch.set(facecolor=color, alpha=0.5)
            
            
                    
            ax[i].set_title('Northern Hemisphere' if hemisphere == 'NH' else 'Southern Hemisphere')   
            ax[i].text(0.05, 0.95, f'{fig_label[i]}', fontweight='bold', horizontalalignment='left', verticalalignment='top', transform=ax[i].transAxes)
            ax[i].set_xticks(np.arange(1,14)) 
            ax[i].set_xlim([0, 13.5])
            
            ax[i].set_xticklabels(np.append(np.arange(1,13), 'years'), fontsize=12)
            ax[i].set_xlabel('Month')
            
            if var_name == 'sf_eff':# or var_name == 'pr_eff':
                ax[i].set_ylim([0,8.])
                ax[i].set_yticks(np.arange(0,8.50,.50))
            elif var_name == 'pr_eff':
                ax[i].set_ylim([0,800])
                ax[i].set_yticks(np.arange(0,850,50))
            else:
                ax[i].set_ylim([dict_label['vmin'],dict_label['vmax']])
                ax[i].set_yticks(np.arange(0,110,10))
            
            ax[i].set_ylabel(dict_label['cb_label'] if i==0 else '') 
    # add max, min model
    xloc = [0.75]
    xloc.extend(np.arange(2.25, 14.25))
    for (i, hemisphere) in enumerate(['NH', 'SH']):
        # ax[i].fill_between(x=xloc , y1=min_val.sel(hemisphere=hemisphere,), y2=max_val.sel(hemisphere=hemisphere,), alpha=0.25, color=colors[2])
        ax[i].scatter(np.arange(1.25, 14.25), min_val.sel(hemisphere=hemisphere,), marker=".", s=75, color=colors[2], alpha=0.45)
        ax[i].scatter(np.arange(1.25, 14.25), max_val.sel(hemisphere=hemisphere,), marker=".", s=75, color=colors[2], alpha=0.45)
            
    s = f.subplotpars
    bb = [s.left, s.top - 0.92, (s.right - s.left), 0.05]

    ax[0].legend([bp[0]["boxes"][0], bp[1]["boxes"][0], bp[2]["boxes"][0]], ['CloudSat', 'ERA5', 'CMIP6$_{mean}$'], bbox_to_anchor=bb,loc=8,ncol=3,borderaxespad=0,fancybox=True,bbox_transform=f.transFigure, mode='expand')
    # ax[0].legend([bp[0]["boxes"][0], ], ['CloudSat', ], bbox_to_anchor=bb,loc=8,ncol=3,borderaxespad=0,fancybox=True,bbox_transform=f.transFigure, mode='expand')
    
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.)  ;
    
    figname = f'{var_name}_monthly_interannual_variation_2007_2010.png'
    plt.savefig(fig_dir + figname, format='png', bbox_inches='tight', transparent=True)    
 
# def calc_linear_regression(df, model, ):
#     # To do this we use the polyfit function from Numpy. Polyfit does a least squares polynomial fit over the data that it is given. 
#     # We want a linear regression over the data in columns cloudsat and MIROC6 so we pass these as parameters. The final parameter is the 
#     # degree of the polynomial. For linear regression the degree is 1.
#     d = np.polyfit(df['reference'], df[model],1) # These are the a and b values we were looking for in the linear function formula.
#     # We then use the convenience function poly1d to provide us with a function that will do the fitting.
#     f = np.poly1d(d) #predict the estimated results
#     # We now use the function f to produce our linear regression data and inserting that into a new column called Treg.
#     df.insert(2, 'Treg', f(df['reference']))
#     # the R-squared value is a number between 0 and 1. And the closer it is to 1 the more accurate your linear regression model is.
#     R2 = r2_score(df[model],f(df['reference']))
#     return(R2,d)

# def calc_linear_regression_hemisphere(ratios_x, ratios_y, season, model, lat_slice, var_name):

#     df_NH = pd.DataFrame()
#     df_NH['reference'] = ratios_x[f'{var_name}_season'].sel(season=season, lat=lat_slice).to_dataframe()[f'{var_name}_season']
    
#     df_NH[model] = ratios_y[f'{var_name}_season'].sel(season=season, model=model, lat=lat_slice).to_dataframe()[f'{var_name}_season']
#     df_NH.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df_NH.dropna(inplace=True)
#     R2_NH, d_NH = calc_linear_regression(df_NH, model)    
#     return(df_NH, R2_NH, d_NH)
def calc_linear_regression_hemisphere(ratios, hemisphere, var_name):
    lat_slice = slice(45,90) if hemisphere == 'NH' else slice(-90,-45)

    slopes = dict()
    intercepts = dict()
    rvalues = dict()
    for season in ratios.season.values:
        slopes[season] = []
        intercepts[season] = []
        rvalues[season] = []
        
        if var_name == 'sf_eff':
            # Extract the data for ERA5 for the current season
            ratios_x = ratios[f'{var_name}_season'].sel(lat=lat_slice, season=season, model='ERA5').squeeze().values.flatten()
            models = ratios.model.values[2:]
        else:
            # Extract the data for CloudSat for the current season
            ratios_x = ratios[f'{var_name}_season'].sel(lat=lat_slice, season=season, model='CloudSat').squeeze().values.flatten()
            models = ratios.model.values[1:]
        
        # Loop through other models
        for model in models:
            ratios_y = ratios[f'{var_name}_season'].sel(lat=lat_slice, season=season, model=model).squeeze().values.flatten()
            mask = ~np.isnan(ratios_x) & ~np.isnan(ratios_y)
            
            result = linregress(ratios_x[mask],ratios_y[mask])
            slope = result.slope
            slopes[season].append(slope)
            
            intercept = result.intercept
            intercepts[season].append(intercept)
            
            rvalue = result.rvalue
            rvalues[season].append(rvalue)
            
        # Convert the list of slopes to a NumPy array
        slopes[season] = np.array(slopes[season])
        intercepts[season] = np.array(intercepts[season])
        rvalues[season] = np.array(rvalues[season])
    da = xr.Dataset()
    da['linregress_slope'] = xr.DataArray(data = list(slopes.values()),
                                          dims=['season', 'model'],
                                          coords = dict(season=list(slopes.keys()), 
                                                        model=models))
    da['linregress_intercept'] = xr.DataArray(data = list(intercepts.values()),
                                          dims=['season', 'model'],
                                          coords = dict(season=list(intercepts.keys()), 
                                                        model=models))
    da['linregress_rvalues'] = xr.DataArray(data = list(rvalues.values()),
                                          dims=['season', 'model'],
                                          coords = dict(season=list(rvalues.keys()), 
                                                        model=models))
    return da#slopes, intercepts, rvalues

def get_linear_regression_hemisphere(ratios,var_name):
    results = dict()
    for hemisphere in ['NH', 'SH']:
        results[hemisphere] = calc_linear_regression_hemisphere(ratios, hemisphere, var_name)
        
    _ds = list(results.values())
    _coord = list(results.keys())
    linregress = xr.concat(objs=_ds, dim=_coord).rename({"concat_dim":"hemisphere"})
    
    return(linregress)
# def calc_scatter_obs_model(ratios, var_name):
#     lat_north = slice(45, 90)
#     lat_south = slice(-90, -45)

#     df_NH = dict()
#     R2_NH = dict()
#     d_NH = dict()

#     df_SH = dict()
#     R2_SH = dict()
#     d_SH = dict()
#     for model in ratios['500'].model.values:
#         df_NH[model] = dict()
#         R2_NH[model] = dict()
#         d_NH[model] = dict()
        
#         df_SH[model] = dict()
#         R2_SH[model] = dict()
#         d_SH[model] = dict()
#         for season in ratios['500'].season.values:
#             # print(season)
            
#             if var_name == 'sf_eff' or var_name == 'pr_eff':
#                 df_NH[model][season], R2_NH[model][season], d_NH[model][season] = calc_linear_regression_hemisphere(ratios['era_500'], 
#                                                                                                                                             ratios['500'], 
#                                                                                                                                             season, 
#                                                                                                                                             model, 
#                                                                                                                                             lat_north, var_name)
                
#                 df_SH[model][season], R2_SH[model][season], d_SH[model][season] = calc_linear_regression_hemisphere(ratios['era_500'], 
#                                                                                                                                             ratios['500'], 
#                                                                                                                                             season, 
#                                                                                                                                             model, 
#                                                                                                                                             lat_south, var_name)
#             else:
#                 df_NH[model][season], R2_NH[model][season], d_NH[model][season] = calc_linear_regression_hemisphere(ratios['cloudsat_500'], 
#                                                                                                                                             ratios['500'], 
#                                                                                                                                             season, 
#                                                                                                                                             model, 
#                                                                                                                                             lat_north, var_name)
                
#                 df_SH[model][season], R2_SH[model][season], d_SH[model][season] = calc_linear_regression_hemisphere(ratios['cloudsat_500'], 
#                                                                                                                                             ratios['500'], 
#                                                                                                                                             season, 
#                                                                                                                                             model, 
#                                                                                                                                             lat_south, var_name)
                
#     return(df_NH, R2_NH, d_NH, df_SH, R2_SH, d_SH)



def plt_scatter_obs_model(ratios, regression, var_name, dict_label, fig_dir, lwp_threshold):
    
    # df_NH, R2_NH, d_NH, df_SH, R2_SH, d_SH = calc_scatter_obs_model(ratios, var_name)
    
    # if var_name == 'sf_eff' or var_name == 'pr_eff':
    #     df_NH.pop('ERA5')
    #     # df_SH.pop('ERA5')
    if var_name == 'sf_eff':
        models = regression.model.values[1:]
    else:
        models = regression.model.values 
    
    figsize=[12,3*len(models)]
    colors = cm.hawaii(range(0, 256, int(256/3)+1))

    f, axsm = plt.subplots(nrows=len(models), ncols=len(regression.season), sharex=True, sharey=True, figsize=figsize)
    f.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=1.005)
    for ax, k in zip(axsm.flatten(), range(len(fig_label))):
        ax.text(0.05, 0.95, f'{fig_label[k]}', fontweight='bold', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        

    # for i, model in zip(range(len(df_NH)), df_NH.keys()):
    #     for ax, season in zip(axsm.flatten()[i*4: (i+1)*4+1], df_NH[model].keys()):
                
                
    #             if i == 0:
    #                 ax.set_title(f'season = {season}')
                    
    #             ax.axline((0,0), slope=1, color='black',linestyle='--', )
                
    #             if var_name == 'sf_eff':# or var_name == 'pr_eff':
    #                 ax.set_xlim([0,10])
    #                 ax.set_ylim([0,10])
    #             else:
    #                 ax.set_xlim([dict_label['vmin'],dict_label['vmax']])
    #                 ax.set_ylim([dict_label['vmin'],dict_label['vmax']])
                
    #             df_NH[model][season].plot.scatter(ax = ax, x = 'reference',y=model, label = 'NH', color = colors[0].reshape(1,-1), alpha=0.1, legend=False)
    #             df_NH[model][season].plot(x='reference', y='Treg',color=colors[0].reshape(1,-1),ax=ax, label= f"y$_N$ = {d_NH[model][season][0].round(2)} x + {d_NH[model][season][1].round(2)}" 
    #                 "\n" 
    #                 f"r$^2$ = {R2_NH[model][season].round(2)}")
                
    #             df_SH[model][season].plot.scatter(ax = ax, x = 'reference',y=model, label = 'SH', color = colors[2].reshape(1,-1), alpha=0.1, legend=False)
    #             df_SH[model][season].plot(x='reference', y='Treg',color=colors[2].reshape(1,-1),ax=ax, label= f"y$_S$ = {d_SH[model][season][0].round(2)} x + {d_SH[model][season][1].round(2)}" 
    #                 "\n" 
    #                 f"r$^2$ = {R2_SH[model][season].round(2)}")
                
    #             ax.legend(loc='lower right')
    #             ax.grid(True)
    #             ax.set_ylabel(model)
    #             if var_name != 'sf_eff':
    #                 ax.set_xlabel('CloudSat')
    #             if var_name == 'sf_eff':
    #                 ax.set_xlabel('ERA5')
            
    for (j, model) in enumerate(models):
        for (i, season)  in enumerate(regression.season.values): 
            ax = axsm[j, i]
            ax.grid(True)
            ax.axline((0,0), slope=1, color='darkgrey', linestyle='--')
            if i == 0:
                ax.set_ylabel(f"{model}, {dict_label['cb_label']}")
            if j == 0 :
                ax.set_title(f"season = {season}")
            if j == len(regression.model)-1:
                if var_name == 'sf_eff':
                    ax.set_xlabel(f"ERA5, {dict_label['cb_label']}")
                else:
                    ax.set_xlabel(f"CloudSat, {dict_label['cb_label']}")
            for hemisphere in (['NH','SH']):
                color = colors[0] if hemisphere == 'NH' else colors[2]
                lat_slice = slice(45,90) if hemisphere == 'NH' else slice(-90,-45)
                if var_name == 'sf_eff':
                    ax.scatter(x= ratios[f"{var_name}_season"].sel(model='ERA5', lat=lat_slice, season=season),
                               y= ratios[f"{var_name}_season"].sel(model=model, lat=lat_slice, season=season), 
                               color = color, alpha=0.1, label = '_nolegend_' )
                ax.scatter(x= ratios[f"{var_name}_season"].sel(model='CloudSat', lat=lat_slice, season=season),
                           y= ratios[f"{var_name}_season"].sel(model=model, lat=lat_slice, season=season), 
                           color = color, alpha=0.1, label = '_nolegend_' )
                a = regression['linregress_intercept'].sel(variable=var_name, model=model, season=season, hemisphere=hemisphere)
                b = regression['linregress_slope'].sel(variable=var_name, model=model, season=season, hemisphere=hemisphere)
                rvalue = regression['linregress_rvalues'].sel(variable=var_name, model=model, season=season, hemisphere=hemisphere)
                # Create sequence of 100 numbers from 0 to 100
                xseq = np.linspace(0,100, num=9)
                # Plot regression line
                ax.plot(xseq, a.values + b.values * xseq, color=color, lw=2.5, label=f"{hemisphere}$_y$ = {b.values:.1f} x + {a.values:.1f}" "\n" f"R$^2$ = {rvalue.values:.1f}");
            
            ax.legend(loc='lower right')
            
            ax.set_xlim([dict_label['vmin'], dict_label['vmax']])
            ax.set_ylim([dict_label['vmin'], dict_label['vmax']])
                
    f.tight_layout(pad=0., w_pad=0., h_pad=0.)  
    # f.subplots_adjust(top=0.96)
    
    figname = f'{var_name}_season_scatter_2007_2010.png'
    plt.savefig(fig_dir + figname, format='png', bbox_inches='tight', transparent=True)

def plt_R2_heatmap_season(regression, dict_label, fig_dir, lwp_threshold):
    # def plt_R2_heatmap_season(df_NH, df_SH, dict_label, fig_dir, lwp_threshold):
    # # define heatmap colors
    # cmap = cm.hawaii_r  # define the colormap
    # # extract all colors from the hawaii map
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # # create the new map
    # cmap = LinearSegmentedColormap.from_list(
    #     'Custom cmap', cmaplist, cmap.N)

    # # define the bins and normalize
    # bounds = np.linspace(0, 1, 11)
    # norm = BoundaryNorm(bounds, cmap.N)
    
    
    # f, axsm = plt.subplots(nrows=1, ncols=len(df_NH)*2, sharex=True, sharey=True, figsize=[15, 5])
    # f.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=0.93)
    # ax = axsm.flatten()

    # for ax, var_name, k in zip(axsm.flatten()[::2],dict_label.keys(), fig_label[::2]): 
    #     im = ax.imshow(df_NH[var_name], cmap=cmap, norm=norm)
    #     ax.set_title(f'{k} NH', )#fontsize=10.)
        
    #     if k == 'a)':
    #         ax.set(yticks=range(len(df_NH[var_name].index)), yticklabels=df_NH[var_name].index)
    #     if var_name == 'FLCC':
    #         x_position = 0.75  
    #     elif var_name == 'FsLCC':
    #         x_position = 0.75
    #     elif var_name == 'FoP' or var_name == 'FoS' or var_name == 'sf_eff' or var_name == 'pr_eff' or var_name == 'FLCC-FsLCC':
    #         x_position = 0.55
        
    #     plt.figtext(x_position,-0.11, dict_label[var_name]['cb_label'], fontweight='bold', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    #     # elif var_name == 'sf_eff':
    #     #     plt.figtext(0.25,-0.09, dict_label[var_name]['cb_label'], fontweight='bold', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    #     # elif var_name == 'lcc_w_snow':
    #     #     plt.figtext(0.35,-0.09, dict_label[var_name]['cb_label'], fontweight='bold', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    #     ax.set(xticks=range(len(df_NH[var_name].columns)), xticklabels=df_NH[var_name].columns,)
    # for ax, var_name, label, k in zip(axsm.flatten()[1::2],dict_label.keys(), dict_label.values(), fig_label[1::2]):     
    #     im = ax.imshow(df_SH[var_name], cmap=cmap, norm=norm)
    #     ax.set_title(f'{k} SH')
    #     ax.set(xticks=range(len(df_SH[var_name].columns)), xticklabels=df_SH[var_name].columns,)

    # # add space for colour bar
    # f.subplots_adjust(right=0.85)
    # cbar_ax = f.add_axes([1.01, 0.15, 0.0125, 0.7])
    
    # f.colorbar(im, cax=cbar_ax, cmap=cmap, norm=norm,
    #     spacing='proportional', ticks=bounds, boundaries=bounds, label=f'R$^2$-values',shrink=0.5)

    

    # f.tight_layout(pad=0., w_pad=0.4, h_pad=0.) 
    
    figsize = [9,9]
    cmap = cm.hawaii_r
    cmaplist = [cmap(i) for i in range(cmap.N)]
    bounds = np.linspace(0, 1 , 11)
    hemispheres = ['NH', 'SH']
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    norm = BoundaryNorm(bounds, 10)

    # Create a formatting function for colorbar ticks
    # qrates = list(np.arange(0.05 , 1.05 , 0.1 ))
    qrates = list(np.arange(0.05 , 1.05 , 0.1 ))
    fmt = FuncFormatter(lambda x, pos: qrates[::][norm(x)])
    fmt = FormatStrFormatter("%.2f")


    fig, axsm = plt.subplots(nrows=2, ncols=len(regression.variable.values), sharex=True, sharey=True, figsize=figsize)
    fig.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=1.01)

    
    for j, var_name in enumerate(regression.variable.values):
                da = regression['linregress_rvalues'].sel(variable=var_name, )
                da = da.transpose()
                da = da.where(~np.isnan(da),other=np.nan)
                for i, hemisphere in enumerate(['NH', 'SH']):
                        ax = axsm[i,j]
                        # ax.set_title(f"{dict_label['cb_label']}", fontsize=8)
                        im = ax.imshow(da.sel(hemisphere=hemisphere), cmap=cmap.resampled(10), norm=norm)
                        
                        # Show all ticks and label them with the respective list entries
                        ax.set_yticks(np.arange(len(da.model.values)), labels=da.model.values)
                        ax.set_xticks(np.arange(len(da.season.values)), labels=da.season.values)
                        
                        # Rotate the tick labels and set their alignment.
                        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va="center", rotation_mode="anchor")

                        for k in range(len(da.model.values)):
                                for l in range(len(da.season.values)):
                                        if var_name == 'sf_eff' and k == 0:
                                                if l == 0 or l ==1 or l ==2 or l==3:
                                                        color = 'white'
                                        else:
                                                color = 'k'
                                        text = ax.text(l, k, f"{(abs(da.sel(hemisphere=hemisphere)).values[k, l]):.1f}",
                                                        ha="center", va="center", color=color,)
                                        
                        if hemisphere == 'SH':
                                if var_name == 'FLCC-FsLCC':
                                        var_text = 'FLCC-FsLCC (%)'
                                else:
                                        var_text = dict_label[var_name]['cb_label']
                                plt.figtext(0.5, -0.15, f"{var_text}", fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    for ax, row in zip(axsm[:,0], ['Northern Hemisphere', 'Southern Hemisphere']):
                ax.text(-1.03, 0.55, row, va='bottom', ha='center', rotation='vertical',
                        rotation_mode='anchor', transform=ax.transAxes, fontweight='bold')
    # Add subplot label
    for ax, k, var_name in zip(axsm.flatten(), range(len(fig_label)), list(dict_label.keys()) + list(dict_label.keys())):
                # ax.text(0.05, 1.035, f"{fct.fig_label[k]} {dict_label['cb_label']}", fontweight='bold', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
                ax.text(0.05, 1.035, f"{fig_label[k]}", fontweight='bold', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

    # Create colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([1.015, 0.275, 0.0125, 0.5])
    cbar_kw = dict(ticks=np.arange(0.05,1.05,0.1), format=fmt)
    fig.colorbar(im, cax=cbar_ax, **cbar_kw, label=f' R$^2$', shrink=0.5)#cmap=cmap, norm=norm, )
    plt.tight_layout(pad=0., w_pad=0., h_pad=0)  ; 
    figname = f'R2_season_2007_2010.png'
    plt.savefig(fig_dir + figname, format='png', bbox_inches='tight', transparent=True)


def plot_spatial_season(difference, val1, val2, val3, val1_mean, val2_mean, val3_mean,
                        hemisphere, ds, var_name, dict_label, fig_dir, lat_extent, lwp_threshold):
  
    if difference != None:
        if var_name == 'sf_eff' or var_name == 'pr_eff':
            _val3 = val3.copy()
            diff_2 = val2 - val3
            diff_2_mean = val2_mean - val3_mean 
        if var_name == 'FLCC' or var_name == 'FsLCC' or var_name == 'FoS' or var_name == 'FLCC-FsLCC':    
            _val1 = val1.fillna(0.)
            _val1 = _val1.where(np.logical_and(_val1.lat <= 82., _val1.lat >= -82.), other=np.nan)
            _val2 = val2.fillna(0.)
            _val2 = _val2.where(np.logical_and(_val2.lat <= 82., _val2.lat >= -82.), other=np.nan)
            _val3 = val3.fillna(0.)
            _val3 = _val3.where(np.logical_and(_val3.lat <= 82., _val3.lat >= -82.), other=np.nan)
        
            # diff_1 = _val1.where(~np.isnan(_val2)) - _val2.where(~np.isnan(_val1))
            diff_1 = _val1 - _val2
            diff_1_mean = val1_mean - val2_mean
            
            # diff_2 = _val1.where(~np.isnan(_val3)) - _val3.where(~np.isnan(_val1))
            diff_2 = _val1 - _val3
            diff_2_mean = val1_mean - val3_mean
        
        if 'model' in diff_2.coords:
            len_model = len(diff_2['model'])
            diff_2 = diff_2.mean('model', skipna=True, keep_attrs=False)
            # diff_2 = diff_2.where(~np.isnan(diff_2))
            diff_2_mean = diff_2_mean.mean('model',skipna=True, keep_attrs=False)
            # create model std 
            std_cmip = _val3.std('model', skipna=True, keep_attrs=False)
            # std_cmip = std_cmip.where(~np.isnan(std_cmip))
            # calculate statistic significance dependend on the model spread https://towardsdatascience.com/confidence-intervals-explained-simply-for-data-scientists-8354a6e2266b
            SE = std_cmip/np.sqrt(len_model) # standard error
            CI = SE * 1.96   # for 95% confidence level, or CI=2.58*SE for 99% confidence level
        
        if 'model' in _val3.coords:
            _val3 = _val3.mean('model', skipna=True, keep_attrs=False)
            val3_mean = val3_mean.mean('model', skipna=True, keep_attrs=False)    
        if var_name == 'FLCC' or var_name == 'FsLCC' or var_name == 'FoS':
    #     else:
            var1= val1.copy() #where(val1>=0., other=np.nan)
            var1_mean = val1_mean.copy()
            
            var2 = diff_1.copy()    
            var2_mean = diff_1_mean.copy()
            
            var3= diff_2.copy()            
            var3_mean = diff_2_mean.copy()
        if var_name == 'FLCC-FsLCC':
            var1 = val1.copy()
            var1_mean = val1_mean.copy()
            
            var2 = val2.copy()
            var2_mean = val2_mean.copy()
            
            # var3 = val3.copy()
            # var3_mean = val3_mean.copy()
            var3 = diff_1.copy()
            var3_mean = diff_1_mean.copy()
        if var_name == 'sf_eff' or var_name == 'pr_eff':
            var1 = val2.copy()
            var1_mean = val2_mean.copy()
            
            var2 = _val3.copy()
            var2_mean = val3_mean.copy()
            
            var3 = diff_2.copy()
            var3_mean = diff_2_mean.copy()
            

        
                
        
    
    if difference == None:
        if var_name == 'FLCC' or var_name == 'FsLCC' or var_name == 'FoS' or var_name == 'FLCC-FsLCC':
            var1 = val1.copy()
            var1_mean = val1_mean.copy()
            
            var2 = val2.copy()
            var2_mean = val2_mean.copy()
            
            if 'model' in val3.coords:
                var3 = val3.mean('model', skipna=True, keep_attrs=False)
                var3_mean = val3_mean.mean('model', skipna=True, keep_attrs=False)
            else:
                var3 = val3.copy()
                var3_mean = val3_mean.copy()
        if var_name == 'FLCC-FsLCC':
            var1 = val1.copy()
            var1_mean = val1_mean.copy()
            
            var2 = val2.copy()
            var2_mean = val2_mean.copy()
            
            var3 = val3.where(np.isnan(val3), other=np.nan)
            var3_mean = val3_mean.where(np.isnan(val3_mean), other=np.nan)
        
        if var_name == 'sf_eff' or var_name == 'pr_eff':
            var1 = val2.copy()
            var1_mean = val2_mean.copy()
            
            if 'model' in val3.coords:
                var2 = val3.mean('model', skipna=True, keep_attrs=False)
                var2_mean = val3_mean.mean('model', skipna=True, keep_attrs=False)
            else:
                var2 = val3.copy()
                var2_mean = val3_mean.copy()
                
            var3 = val3.where(np.isnan(val3), other=np.nan)
            var3_mean = val3_mean.where(np.isnan(val3_mean), other=np.nan)
            

            
    
    
    projection = create_projection(hemisphere)
    density = 4
    if ~np.isnan(var3).all() != False:
        f, axsm = plt.subplots(nrows=3, 
                            ncols=4, 
                            subplot_kw={'projection': projection}, 
                            figsize=[12, 9], sharex=True, sharey=True)
        if difference != None:
            f.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=0.925)
            if var_name == 'FLCC-FsLCC':
                f.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=1.005)
        else:
            f.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=1.005)
        
    else:
        f, axsm = plt.subplots(nrows=2, 
                           ncols=4, 
                           subplot_kw={'projection': projection}, 
                           figsize=[12, 6], sharex=True, sharey=True)
        f.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=1.005)
    if difference != None:
        if var_name == 'sf_eff' or var_name == 'pr_eff':
            model_labels = ['ERA5', 'CMIP6$_{mean}$', 'ERA5 - CMIP6$_{mean}$']
        elif var_name == 'FLCC-FsLCC':
            model_labels = ['FLCC', 'FsLCC', 'FLCC - FsLCC']
        else: 
            model_labels = ['CloudSat', 'CloudSat - ERA5', 'CloudSat - CMIP6$_{mean}$']
    if difference == None:
        if var_name == 'sf_eff' or var_name == 'pr_eff':
            model_labels = ['ERA5', 'CMIP6$_{mean}$', ]
        else:
            model_labels = ['CloudSat', 'ERA5', 'CMIP6$_{mean}$']
        
    
    for ax, row in zip(axsm[:,0], model_labels):
        ax.text(-0.07, 0.55, row, 
                va='bottom', 
                ha='center', 
                rotation='vertical', 
                rotation_mode='anchor', 
                transform=ax.transAxes, 
                fontweight='bold')
    for ax, k in zip(axsm.flatten(), range(len(fig_label))):
        setup_axes(ax, hemisphere, lat_extent)
        ax.text(0.05, 0.95, 
                f'{fig_label[k]}', 
                fontweight='bold', 
                horizontalalignment='left', 
                verticalalignment='top', 
                transform=ax.transAxes)
    
    
    vmin = dict_label['vmin']
    vmax = dict_label['vmax']
    
    list1 = [var1, var2, var3]
    list_glob = [var1_mean, var2_mean, var3_mean]

    
    for i, (value,hemi_glob) in enumerate(zip(list1,list_glob)):
        
        if i == 0:
            cmap = cm.hawaii_r 
            levels = dict_label['levels']
        if difference!=None: 
            if i == 1:
                if var_name == 'sf_eff' or var_name == 'pr_eff' or var_name == 'FLCC-FsLCC':
                    cmap = cm.hawaii_r
                    levels = dict_label['levels']
                    # boundaries = dict_label['levels']
                    # colors = list(cmap(np.arange(len(boundaries))))
                    # # set over-color to last color of list 
                    # cmap.set_over(colors[0])
                    # norm = BoundaryNorm(boundaries, ncolors=len(boundaries)-1, clip=False)
                else:
                    cmap=cm.bam
                    levels = dict_label['diff_levels']
            if i == 2:
                cmap=cm.bam
                levels = dict_label['diff_levels']
                
        if var_name == 'FLCC-FsLCC' and i == 2:
            cmaplist = [cmap(i) for i in range(cm.bam.N)]
            cmap = LinearSegmentedColormap.from_list('new', cmaplist[int(len(cmaplist)/2):], int(cm.bam.N/2))
            norm = BoundaryNorm(levels[int(len(levels)/2):], ncolors=cmap.N, clip=True)
        else:
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
            
                    
        
        sub_title = ""

        for ax, season in zip(axsm.flatten()[i * 4: (i + 1) * 4 + 1], val1.season):
            if i == 0:
                sub_title = f'season = {season.values}'
            
            val = value.sel(lat=slice(45,90)) if hemisphere == 'NH' else value.sel(lat=slice(-90,-45))
            cf = ax.pcolormesh(val.lon, val.lat, (val.where(~np.isnan(val))).sel(season=season), 
                            transform=ccrs.PlateCarree(), 
                            cmap=cmap, 
                            norm=norm)
            
            add_text_box(ax, hemi_glob.sel(hemisphere=hemisphere, season=season), var_name)
            ax.set_title(sub_title)
            
            if difference != None and i == 0:
                if var_name == 'sf_eff' or var_name == 'pr_eff':
                    cbaxes = f.add_axes([0.92, 0.4, 0.0125, 0.45])
                    cb_label = dict_label['cb_label']
                    extend = 'max'
                elif var_name == 'FLCC-FsLCC':
                    cbaxes = f.add_axes([1.0, 0.4, 0.0125, 0.45])
                    cb_label = 'Fraction (%)'
                    extend = None
                else:
                    cbaxes = f.add_axes([0.92, 0.65, 0.0125, 0.225])
                    cb_label = dict_label['cb_label']
                    extend = None
                plt.colorbar(cf, cax=cbaxes, shrink=0.5,extend=extend, orientation='vertical', label=cb_label)
            if difference != None and i == 2:
                if var_name == 'sf_eff' or var_name == 'sf_eff':
                    cbaxes = f.add_axes([0.92, 0.13, 0.0125, 0.2])
                    cb_label = 'ERA5 - Model (h$^{-1}$)'
                    extend = 'both'
                elif var_name == 'FLCC-FsLCC':
                    cbaxes = f.add_axes([1.0, 0.024, 0.0125, 0.3])
                    cb_label = 'FLCC-FsLCC (%)'
                    extend = None
                else:
                    cbaxes = f.add_axes([0.92, 0.13, 0.0125, 0.45])
                    cb_label = 'CloudSat - Model (%)'
                    extend = None
                plt.colorbar(cf, cax=cbaxes, shrink=0.5,extend=extend, orientation='vertical', label=cb_label)
            if difference == None and i == 1:
                cbaxes = f.add_axes([1, 0.25, 0.0125, 0.45])
                cb_label = dict_label['cb_label']
                if var_name == 'sf_eff':
                    extend = 'max'
                else:
                    extend = None
                plt.colorbar(cf, cax=cbaxes, shrink=0.5,extend=extend, orientation='vertical', label=cb_label)
            
    
            if difference != None and i == 2 and var_name != 'FLCC-FsLCC':
                plot_difference_significance(ax, hemisphere, diff_2, season, CI, density)
                bb = [0.95, 0.09, 0.0125, 0.05]
                axsm.flatten()[i].legend(
                    [Patch(facecolor='none', edgecolor='k', hatch=density * '/', label='CI < 95%')],
                    ['CI < 95%'], bbox_to_anchor=bb, loc=8, ncol=1, borderaxespad=0,
                    fancybox=True, bbox_transform=f.transFigure
                )
                
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.,)
    if difference != None:
        figname = f'{var_name}_season_{hemisphere}_2007_2010.png'
    if difference == None:
        figname = f'{var_name}_CS_ERA5_CMIP6_season_{hemisphere}_2007_2010.png'
    plt.savefig(fig_dir + figname, format='png', bbox_inches='tight', transparent=True)


def plt_spatial_season_var(ds, var_name, dict_label, fig_dir, lat_extent, lwp_threshold):
    if var_name == 'FLCC-FsLCC':
        val1= ds['FLCC' + '_season'].sel(model='CloudSat').squeeze()
        val1_mean= ds['FLCC' + '_season_mean'].sel(model='CloudSat').squeeze()
        

        val2= ds['FsLCC' + '_season'].sel(model='CloudSat').squeeze()
        val2_mean= ds['FsLCC' + '' + '_season_mean'].sel(model='CloudSat').squeeze()
        
        
        val3= ds['FLCC-FsLCC' + '_season'].sel(model='CloudSat').squeeze()
        val3_mean = ds['FLCC-FsLCC' + '' +'_season_mean'].sel(model='CloudSat').squeeze()
        val3 = val3.drop('model')
        val3_mean = val3_mean.drop('model')
    else:
        val1= ds[var_name + '_season'].sel(model='CloudSat').squeeze()
        val2= ds[var_name + '_season'].sel(model='ERA5').squeeze()
        val3= ds[var_name + '_season'].sel(model = ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                    'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                    'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA'])
        
        if var_name == 'sf_eff':
            val1_mean = ds[var_name + '_season_mean'].sel(model='CloudSat').squeeze()
            val2_mean = ds[var_name + '_season_mean'].sel(model='ERA5').squeeze()
            val3_mean = ds[var_name + '_season_mean'].sel(model=['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                 'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA'])
        else:
            val1_mean= ds[var_name + '_season_cs_mean'].sel(model='CloudSat').squeeze()
            val2_mean= ds[var_name + '_season_cs_mean'].sel(model='ERA5').squeeze()
            val3_mean= ds[var_name + '_season_cs_mean'].sel(model = ['MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                     'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                     'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA'])
    
    val1 = val1.drop('model')
    val1_mean = val1_mean.drop('model')
    val2 = val2.drop('model')
    val2_mean = val2_mean.drop('model')


        
    
    for hemisphere in ['NH', 'SH']:
        plot_spatial_season(None, val1, val2, val3, val1_mean, val2_mean, val3_mean,
                        hemisphere, ds, var_name, dict_label[var_name], fig_dir, lat_extent, lwp_threshold) 
        plot_spatial_season('yes', val1, val2, val3, val1_mean, val2_mean, val3_mean,
                        hemisphere, ds, var_name, dict_label[var_name], fig_dir, lat_extent, lwp_threshold)  
        
        
        


# def plt_spatial_season_FLCC_FsLCC(ds, dict_label, fig_dir, lat_extent, lwp_threshold):
#     for hemisphere in ['NH', 'SH']:
#         val1= ds['FLCC' + '_season']
#         val1_mean= ds['FLCC' + '_season_mean']

#         val2= ds['FsLCC' + '_season']
#         val2_mean= ds['FsLCC' + '' + '_season_mean']

#         val3= ds['FLCC-FsLCC' + '_season']
#         val3_mean = ds['FLCC-FsLCC' + '' +'_season_mean']
        
#         plot_spatial_season(None, val1, val2, val3, val1_mean, val2_mean, val3_mean,
#                         hemisphere, ds, 'FLCC-FsLCC', dict_label['FLCC-FsLCC'], fig_dir, lat_extent,lwp_threshold)    
#         plot_spatial_season('yes', val1, val2, val3, val1_mean, val2_mean, val3_mean,
#                         hemisphere, ds, 'FLCC-FsLCC', dict_label['FLCC-FsLCC'], fig_dir, lat_extent,lwp_threshold)
    

    
def plot_spatial_season_model(val1, val2, val3, val1_mean, val2_mean, val3_mean,
                        hemisphere, var_name, dict_label, fig_dir, lat_extent, model):
    
    
    # val1 = val1.where(val1>0., other=np.nan)

    # if 'model' in val2.coords:
    #     len_model = len(val2['model'])
    #     # create model std 
    #     std_cmip = val2.std('model', skipna=True, keep_attrs=False)
    #     std_cmip = std_cmip.where(~np.isnan(std_cmip))
    #     # calculate statistic significance dependend on the model spread https://towardsdatascience.com/confidence-intervals-explained-simply-for-data-scientists-8354a6e2266b
    #     SE = std_cmip/np.sqrt(len_model) # standard error
    #     CI = SE * 1.96   # for 95% confidence level, or CI=2.58*SE for 99% confidence level
    #     val2 = val2.sel(model=model)
    #     val2 = val2.where(val2>0., other=np.nan)
        
    #     val2_mean = val2_mean.sel(model=model)
        
    # if 'model' in val3.coords:
    #     val3 = val3.sel(model=model)
        
    #     val3_mean = val3_mean.sel(model=model)  
        

    
    projection = create_projection(hemisphere)
    density = 4
    f, axsm = plt.subplots(nrows=3, 
                            ncols=4, 
                            subplot_kw={'projection': projection}, 
                            figsize=[12, 9], sharex=True, sharey=True)
        
    if var_name == 'sf_eff' or var_name == 'pr_eff':
        # model_labels = [f'ERA5 ({res}km)', f'{model} ({res}km)', f'ERA5 - {model}']
        model_labels = [f'ERA5', f'{model}', f'ERA5 - {model}']
    else: 
        model_labels = [f'CloudSat', f'{model}', f'CloudSat - {model}']
    
    for ax, row in zip(axsm[:,0], model_labels):
        ax.text(-0.07, 0.55, row, 
                va='bottom', 
                ha='center', 
                rotation='vertical', 
                rotation_mode='anchor', 
                transform=ax.transAxes, 
                fontweight='bold')
    for ax, k in zip(axsm.flatten(), range(len(fig_label))):
        setup_axes(ax, hemisphere, lat_extent)
        ax.text(0.05, 0.95, 
                f'{fig_label[k]}', 
                fontweight='bold', 
                horizontalalignment='left', 
                verticalalignment='top', 
                transform=ax.transAxes)
    
    
    vmin = dict_label['vmin']
    vmax = dict_label['vmax']
    
    list1 = [val1, val2, val3]
    list_glob = [val1_mean, val2_mean, val3_mean]

    
    for i, (value,hemi_glob) in enumerate(zip(list1,list_glob)):
        
        if i == 0 or i == 1:
            cmap = cm.hawaii_r 
            levels = dict_label['levels']
            cbaxes = f.add_axes([1., 0.4, 0.0125, 0.45])
            cb_label = dict_label['cb_label']
            if var_name == 'sf_eff':
                extend = 'max'
            else:
                extend = None

        if i == 2:
            cmap=cm.bam
            levels = dict_label['diff_levels']
            # cbaxes = f.add_axes([1., 0.13, 0.0125, 0.2])
            cbaxes = f.add_axes([1.0, 0.025, 0.0125, 0.3])
            if var_name == 'sf_eff' or var_name == 'pr_eff':
                cb_label = f"Diff. {dict_label['cb_label']}"
                extend = 'both'
            else:
                cb_label = f"Diff. {dict_label['cb_label']}  (%)"
                extend = None

        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        sub_title = ""

        for ax, season in zip(axsm.flatten()[i * 4: (i + 1) * 4 + 1], val1.season):
            if i == 0:
                sub_title = f'season = {season.values}'

            val = value.sel(lat=slice(45,90)) if hemisphere == 'NH' else value.sel(lat=slice(-90,-45))
            cf = ax.pcolormesh(val.lon, val.lat, (val.where(~np.isnan(val))).sel(season=season), 
                            transform=ccrs.PlateCarree(), 
                            cmap=cmap, 
                            norm=norm)
            add_text_box(ax, hemi_glob.sel(hemisphere=hemisphere, season=season), var_name)
            ax.set_title(sub_title)
               
            plt.colorbar(cf, cax=cbaxes, shrink=0.5,extend=extend, orientation='vertical', label=cb_label)

    
            # if i == 2:
            #     plot_difference_significance(ax, hemisphere, val3, season, CI, density)
            #     bb = [0.95, 0.09, 0.0125, 0.05]
            #     axsm.flatten()[i].legend(
            #         [Patch(facecolor='none', edgecolor='k', hatch=density * '/', label='CI < 95%')],
            #         ['CI < 95%'], bbox_to_anchor=bb, loc=8, ncol=1, borderaxespad=0,
            #         fancybox=True, bbox_transform=f.transFigure
            #     )
                
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.,)
    figname = f'{model}_{var_name}_season_{hemisphere}_2007_2010.png'
    plt.savefig(fig_dir + figname, format='png', bbox_inches='tight', transparent=True)


def plt_spatial_season_all_models(ds, var_name, dict_label, fig_dir, lat_extent):
    
    # for model in ds['cmip_500']['model'].values:
        # if model == 'CanESM5' or model == 'IPSL-CM5A2-INCA':
        #     res = 500
        # else:
        #     res = 250
                
    if var_name == 'sf_eff':
            # val1 = ds[f'era_{res}'][var_name + '_season']
            # val1_mean = ds[f'era_{res}'][var_name + '_season_mean']
            val1 = ds[var_name + '_season'].sel(model='ERA5').squeeze()
            val1_mean = ds[var_name +'_season_cs_mean'].sel(model='ERA5').squeeze()
            models = list(ds.model.values)[2:]
    else:
            # val1 = ds[f'cloudsat_{res}'][var_name + '_season']
            # val1_mean = ds[f'cloudsat_{res}'][var_name + '_season_mean']
            val1 = ds[var_name + '_season'].sel(model='CloudSat').squeeze()
            val1_mean = ds[var_name + '_season_cs_mean'].sel(model='CloudSat').squeeze()
            models = list(ds.model.values)[1:]
    val1 = val1.drop('model')
    val1_mean = val1_mean.drop('model')
    
    for model in models:                
        # val2 = ds[f'cmip_{res}'][var_name + '_season']
        # val2_mean = ds[f'cmip_{res}'][var_name + '_season_mean']
        val2 = ds[var_name + '_season'].sel(model=model ).squeeze()
        if var_name == 'sf_eff':
            val2_mean = ds[var_name + '_season_mean'].sel(model=model).squeeze()
        else:
            val2_mean = ds[var_name + '_season_cs_mean'].sel(model=model).squeeze()
        
        val2 = val2.drop('model')
        val2_mean = val2_mean.drop('model')
        

        val3 = val1-val2
        val3_mean = val1_mean-val2_mean
        for hemisphere in ['NH', 'SH']:
            plot_spatial_season_model(val1, val2, val3, val1_mean, val2_mean, val3_mean,
                                    hemisphere, var_name, dict_label[var_name], fig_dir, lat_extent, model)
            
            
            
def plt_percent_difference_season(dataset, var_name, lwp_threshold, dict_label,  fig_dir):
    if var_name == 'sf_eff' or var_name == 'pr_eff':
        da = ((dataset[var_name+'_season_cs_mean'].sel(model='ERA5') - dataset[var_name+'_season_cs_mean'].sel(model = ['ERA5','MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                                                                                        'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                                                                                        'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA']))/dataset[var_name+'_season_cs_mean'].sel(model='ERA5'))*100
    
        da2 = dataset[var_name+'_season_cs_mean'].sel(model = ['ERA5','MIROC6', 'CanESM5', 'AWI-ESM-1-1-LR',
                                                               'MPI-ESM1-2-LR', 'UKESM1-0-LL', 'HadGEM3-GC31-LL', 'CNRM-CM6-1',
                                                               'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'IPSL-CM5A2-INCA'])
        
    else:
        da = ((dataset[var_name+'_season_cs_mean'].sel(model='CloudSat') - dataset[var_name+'_season_cs_mean'])/dataset[var_name+'_season_cs_mean'].sel(model='CloudSat'))*100
        da2 = dataset[var_name+'_season_cs_mean']
        
    da = da.where(da != 0., other=np.nan)
    

    # find minimum and maximum model in each season
    row_max = abs(da).argmax('model')
    row_max2 = abs(da.round(0)).argmax('model')
    row_min = abs(da).argmin('model')
    row_min2 = abs(da.round(0)).argmin('model')


    figsize = [12,5]
    cmap = cm.hawaii_r
    cmaplist = [cmap(i) for i in range(cmap.N)]
    bounds = np.linspace(0, 1 * 100, 11)
    hemispheres = ['NH', 'SH']

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    norm = BoundaryNorm(bounds, 10)

    # Create a formatting function for colorbar ticks
    qrates = list(np.arange(0.05 * 100, 1.05 * 100, 0.1 * 100))
    fmt = FuncFormatter(lambda x, pos: qrates[::][norm(x)])
    fmt = FormatStrFormatter("%.0f")


    # Create the figure and subplots
    fig, axsm = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=figsize)
    fig.suptitle(f'LWP threshold: {lwp_threshold} g m$^{-2}$', y=1.005)

    # Iterate over hemispheres and subplots
    for i, hemisphere in enumerate(['NH', 'SH']):
        ax = axsm[i]
        # Plot your data, add labels, and set ticks
        im = ax.imshow(abs(da.sel(hemisphere=hemisphere)), cmap=cmap.resampled(10), norm=norm)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(da.model.values)), labels=da.model.values)
        ax.set_yticks(np.arange(len(da.season.values)), labels=da.season.values)
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va="center", rotation_mode="anchor")
        
        # Turn spines on and create white grid.
        ax.spines[:].set_visible(True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
        # ax.set_xticks(np.arange(da.shape[1]+1)-.5, minor=True)
        # ax.set_yticks(np.arange(da.shape[0]+1)-.5, minor=True)
        
        
        # Add subplot label
        ax.text(-0.09, 0.99, f'{fig_label[i]}', fontweight='bold', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        
        # Loop over data dimensions and create text annotations.
        for j in range(len(da2.season.values)):
            for k in range(len(da2.model.values)):
                text = ax.text(k, j, f"{(abs(da2.sel(hemisphere=hemisphere)).values[j, k]):.0f}",
                            ha="center", va="center", color="k",)

        # Add max and min Rectangles
        for (row, position), (row2, position2) in zip(enumerate(row_max.sel(hemisphere=hemisphere).values), enumerate(row_max2.sel(hemisphere=hemisphere).values)):
            ax.add_patch(Rectangle((position-0.49,row-0.49),0.98,0.98, fill=False, edgecolor='white', lw=1.5))
            ax.add_patch(Rectangle((position2-0.49,row2-0.49),0.98,0.98, fill=False, edgecolor='white', lw=1.5))
        
        for (row, position), (row2, position2) in zip(enumerate(row_min.sel(hemisphere=hemisphere).values), enumerate(row_min2.sel(hemisphere=hemisphere).values)):
            ax.add_patch(Rectangle((position-0.49,row-0.49),0.98,0.98, fill=False, edgecolor='blue', lw=1.5))
            ax.add_patch(Rectangle((position2-0.49,row2-0.49),0.98,0.98, fill=False, edgecolor='blue', lw=1.5))

    # Create colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([.715, 0.375, 0.0125, 0.5])
    cbar_kw = dict(ticks=np.arange(0.05*100,1.05*100,0.1*100), format=fmt)
    fig.colorbar(im, cax=cbar_ax, **cbar_kw, label=f"% Difference {dict_label['cb_label']}", shrink=0.5)#cmap=cmap, norm=norm, )

    plt.tight_layout(pad=0., w_pad=0., h_pad=.5)  ;
    # save figure
    figname = f'{var_name}_perc_diff_2007_2010.png'
    plt.savefig(fig_dir + figname, format='png', bbox_inches='tight', transparent=True)   