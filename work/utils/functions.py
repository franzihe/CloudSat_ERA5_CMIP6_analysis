from imports import xr, xe, cftime, plt, ccrs, cm, cy, np


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
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="vertical", fraction=0.046, pad=0.04)
    # set cbar label
    cb.set_label(label=cbar_label, weight="bold")

    plt.tight_layout()
    fig.subplots_adjust(top=1)

    return axs


def plt_zonal_seasonal(variable_model, title=None, label=None):
    fig, axsm = plt.subplots(2, 2, figsize=[10, 7], sharex=True, sharey=True,)

    fig.suptitle(title, fontsize=16, fontweight="bold")

    axs = axsm.flatten()
    for ax, i in zip(axs, variable_model.season):
        for k, c in zip(
            variable_model.model.values,
            cm.romaO(range(0, 256, int(256 / len(variable_model.model.values)))),
        ):
            variable_model.sel(season=i, model=k).plot(
                ax=ax, label=k, color=c,
            )

        ax.set_ylabel(label, fontweight="bold")
        ax.grid()
        ax.set_title("season: {}".format(i.values))

    axs[1].legend(
        loc="center left",
        bbox_to_anchor=(1, -0.5),
        title=label,
        fontsize="small",
        fancybox=True,
    )

    return axs


def plt_bar_global_mean(var_model, var_obs, label=None):
    fig, ax = plt.subplots(1, 1, figsize=[10, 7], sharex=True, sharey=True,)

    fig.suptitle("Global mean (1985 - 2014)", fontsize=16, fontweight="bold")
    ax.grid()

    for k, c, pos in zip(
        var_model.model.values,
        cm.romaO(range(0, 256, int(256 / len(var_model.model.values)))),
        range(len(var_model.model.values)),
    ):
        ax.bar(pos, var_model.sel(model=k).values, color=c, width=0.5)

    ax.bar(len(var_model.model.values), var_obs.values, color="k", width=0.5)
    ax.set_xticks(range(len(np.append((var_model.model.values), "ERA5").tolist())))
    ax.set_xticklabels(
        np.append((var_model.model.values), "ERA5").tolist(), fontsize=12, rotation=90
    )
    ax.set_ylabel(label, fontweight="bold")

    plt.tight_layout()
