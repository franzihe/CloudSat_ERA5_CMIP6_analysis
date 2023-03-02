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
)

import warnings

fig_label = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)']

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