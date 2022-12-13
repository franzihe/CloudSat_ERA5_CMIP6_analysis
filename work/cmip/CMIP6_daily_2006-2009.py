# %% [markdown]
# # Example with CMIP6 models (100 - 500 km)
# 
# 
# # Table of Contents
# <ul>
# <li><a href="#introduction">1. Introduction</a></li>
# <li><a href="#data_wrangling">2. Data Wrangling</a></li>
# <li><a href="#exploratory">3. Exploratory Data Analysis</a></li>
# <li><a href="#conclusion">4. Conclusion</a></li>
# <li><a href="#references">5. References</a></li>
# </ul>
# 
# 

# %% [markdown]
# # 1. Introduction <a id='introduction'></a>
# Cloud feedbacks are a major contributor to the spread of climate sensitivity in global climate models (GCMs) [Zelinka et al. (2020)](https://doi-org.ezproxy.uio.no/10.1029/2019GL085782). Among the most poorly understood cloud feedbacks is the one associated with the cloud phase, which is expected to be modified with climate change [Bjordal et al. (2020)](https://doi-org.ezproxy.uio.no/10.1038/s41561-020-00649-1). Cloud phase bias, in addition, has significant implications for the simulation of radiative properties and glacier and ice sheet mass balances in climate models. 
# 
# In this context, this work aims to expand our knowledge on how the representation of the cloud phase affects snow formation in GCMs. Better understanding this aspect is necessary to develop climate models further and improve future climate predictions. 
# 
# * Retrieve CMIP6 data through [ESGF](https://esgf-node.llnl.gov/search/cmip6/)
# * Hybrid sigma-pressure coordinates to isobaric pressure levels of the European Centre for Medium-Range Weather Forecast Re-Analysis 5 (ERA5) with [GeoCAT-comb](https://geocat-comp.readthedocs.io/en/latest/index.html)
# * Regridd the CMIP6 variables to the exact horizontal resolution with [`xesmf`](https://xesmf.readthedocs.io/en/latest/)
# * Calculate an ensemble mean of all used models
# * Calculate and plot the seasonal mean of the ensemble mean
# 
# **Questions**
# * How is the cloud phase and snowfall varying between 2007 and 2010?
# 
# > **_NOTE:_** We answer questions related to the comparison of CMIP models to ERA5 in another [Jupyter Notebook](../CMIP6_ERA5_CloudSat/plt_seasonal_mean.ipynb).

# %% [markdown]
# # 2. Data Wrangling <a id='data_wrangling'></a>
# 
# This study will compare surface snowfall, ice, and liquid water content from the Coupled Model Intercomparison Project Phase 6 ([CMIP6](https://esgf-node.llnl.gov/projects/cmip6/)) climate models to the European Centre for Medium-Range Weather Forecast Re-Analysis 5 ([ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)) data from **2006 to 2009**. We conduct statistical analysis at the annual and seasonal timescales to determine the biases in cloud phase and precipitation (liquid and solid) in the CMIP6 models and their potential connection between them. 
# 
# - Time period: 2006 to 2009
# - horizonal resolution: ~100km
# - time resolution: monthly atmospheric data (Amon, AERmon)
# - Variables:
#   
# | shortname     |             Long name                   |      Units    |  levels |
# | ------------- |:---------------------------------------:| -------------:|--------:|
# |  prsn         |    Snowfall Flux                        | [kg m-2 s-1]  | surface |
# | clw           |    Mass Fraction of Cloud Liquid Water  |  [kg kg-1]    |    ml   | 
# |               |                                         | to calculate lwp use integral clw -dp/dg | |
# | tas           |    Near-Surface Air Temperature         |   [K]         | surface |
# | clivi         |    Ice Water Path                       | [kg m-2]      |         |
# | lwp           |    Liquid Water Path                    | [kg m-2]      |         |
# 
# - CMIP6 models:
# 
# | Institution                                            |     Model name    | Reference                                                     |
# | ------------------------------------------------------ |:-----------------:|--------------------------------------------------------------:|
# | [MIROC]() | MIROC6           | [Tatebe et al. (2019)]() |
# | [NCAR]()  | CESM2            | [Danabasoglu et al. (2020)]()  |
# | [CCCma]() | CanESM5          | [Swart et al. (2019)]()     |
# | [AWI]()   | AWI-ESM-1-1-LR   | []() |
# | [MOHC]()  | UKESM1-0-LL      | []() |
# | [MOHC]()  | HadGem3-GC31-LL  | [Roberts et al. (2019)]() |
# | [CNRM-CERFACS]() | CNRM-CM6-1 | [Voldoire et al. (2019)]() |
# | [CNRM-CERFACS]() | CNRM-ESM2-1| [Seferian et al. (2019)]() |
# | [IPSL]() | IPSL-CM6A-LR | [Boucher et al. (2020)]() |
# | [IPSL]() | IPSL-CM5A2-INCA | []()|

# %% [markdown]
# ## Organize my data
# 
# - Define a prefix for my project (you may need to adjust it for your own usage on your infrastructure).
#     - input folder where all the data used as input to my Jupyter Notebook is stored (and eventually shared)
#     - output folder where all the results to keep are stored
#     - tool folder where all the tools
# 
# The ERA5 0.25deg data is located in the folder `/input/cmip6_hist/daily_means`.
# 

# %%
import os
import pathlib
import sys
import socket
hostname = socket.gethostname()

abs_path = str(pathlib.Path(hostname).parent.absolute())
WORKDIR = abs_path[:- (len(abs_path.split('/')[-2] + abs_path.split('/')[-1])+1)]


if "mimi" in hostname:
    print(hostname)
    DATA_DIR = "/scratch/franzihe/"
    FIG_DIR = "/uio/kant/geo-metos-u1/franzihe/Documents/Figures/CMIP6/"
elif "glefsekaldt" in hostname: 
    DATA_DIR = "/home/franzihe/Data/"
    FIG_DIR = "/home/franzihe/Documents/Figures/CMIP6/"

INPUT_DATA_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DATA_DIR = os.path.join(DATA_DIR, 'output')
UTILS_DIR = os.path.join(WORKDIR, 'utils')

sys.path.append(UTILS_DIR)
# make figure directory
try:
    os.mkdir(FIG_DIR)
except OSError:
    pass

# %% [markdown]
# ## Import python packages
# - `Python` environment requirements: file [requirements_globalsnow.txt](../../requirements_globalsnow.txt) 
# - load `python` packages from [imports.py](../../utils/imports.py)
# - load `functions` from [functions.py](../../utils/functions.py)
# 

# %%
# supress warnings
import warnings
warnings.filterwarnings('ignore') # don't output warnings

# import packages
from imports import (xr, intake, cftime, xe, glob, np, cm, pd, fct,ccrs, cy, plt, da, gc, datetime, LogNorm)
xr.set_options(display_style="html")



# %% [markdown]
# ## Open CMIP6 variables
# Get the data required for the analysis. Beforehand we downloaded the daily averaged data on single levels and model levels via.

# %%
cmip_in = os.path.join(INPUT_DATA_DIR, 'cmip6_hist/daily_means')
cmip_out = os.path.join(OUTPUT_DATA_DIR, 'cmip6_hist/daily_means/common_grid')

# make output data directory
try:
    os.mkdir(cmip_out)
except OSError:
    pass

# %%
variable_id = ['clw', 'cli', 'clivi', 'tas', 'prsn']

# %% [markdown]
# At the moment we have downloaded the end of the historical simulations for CMIP6 models. We define start and end year to ensure to only extract the 4-year period between 2007 and 2010.
# 
# $\rightarrow$ Define a start and end year
# 
# We will load all available models into one dictonary, which includes an xarray dataset with `xarray.open_mfdataset(file)` and select the time range [by name](https://xarray.pydata.org/en/stable/user-guide/indexing.html).

# %%
# source_id
list_models = [
            #    'MIROC6', 
               # 'CESM2', 
            #    'CanESM5', 
            #    'AWI-ESM-1-1-LR', 
            #    'MPI-ESM1-2-LR', 
            # #    'UKESM1-0-LL', 
            # #    'HadGEM3-GC31-LL',
               'CNRM-CM6-1',
            #    'CNRM-ESM2-1',
            #    'IPSL-CM6A-LR',
            #    'IPSL-CM5A2-INCA'
            ]

## experiment
experiment_id = ['historical']

## time resolution
t_res = ['day',]

# %% [markdown]
# ## Search corresponding data
# Get the data required for the analysis. Define variables, models, experiment, and time resolution as defined in <a href="#data_wrangling">2. Data Wrangling</a>
# . 

# %%
starty = 2006; endy = 2006
year_range = range(starty, endy+1)

dset_dict = dict()
for model in list_models:
    cmip_file_in = glob('{}/*{}_{}_{}*'.format(cmip_in, t_res[0], model, experiment_id[0]))
    if len(cmip_file_in) != 0:
        dset_dict[model] = xr.open_mfdataset(sorted(cmip_file_in), combine='nested', compat='override', use_cftime=True)
        # select only years needed for analysis
        dset_dict[model] = dset_dict[model].sel(time = dset_dict[model]['time'].dt.year.isin(year_range)).squeeze()
        # shift longitude to be from -180 to 180
        dset_dict[model] = dset_dict[model].assign_coords(lon=(((dset_dict[model]['lon'] + 180) % 360) - 180)).sortby('lon').sortby('time')
    else:
        continue

# %% [markdown]
# ## Assign attributes to the variables
# 
# We will assign the attributes to the variables as in ERA5 to make CMIP6 and ERA5 variables comperable.
# 
# * [`pr`](http://clipc-services.ceda.ac.uk/dreq/u/62f26742cf240c1b5169a5cd511196b6.html) and [`prsn`](http://clipc-services.ceda.ac.uk/dreq/u/051919eddec810e292c883205c944ceb.html) in **kg m-2 s-1** $\rightarrow$ Multiply by **3600** to get **mm h-1**
# 

# %%
now = datetime.utcnow()
for model in dset_dict.keys():
# 
    for var_id in dset_dict[model].keys():
         
        if var_id == 'prsn':
            dset_dict[model][var_id] = dset_dict[model][var_id]*3600
            dset_dict[model][var_id] = dset_dict[model][var_id].assign_attrs({'standard_name': 'snowfall_flux',
    'long_name': 'Snowfall Flux',
    'comment': 'At surface; includes precipitation of all forms of water in the solid phase',
    'units': 'mm h-1',
    'original_units': 'kg m-2 s-1',
    'history': "{}Z altered by F. Hellmuth: Converted units from 'kg m-2 s-1' to 'mm h-1'.".format(now.strftime("%d/%m/%Y %H:%M:%S")),
    'cell_methods': 'area: time: mean',
    'cell_measures': 'area: areacella'})

# %% [markdown]
#  ## Interpolate from CMIP6 hybrid sigma-pressure levels to ERA5 isobaric pressure levels
# 
# The vertical variables in the CMIP6 models are in hybrid sigma-pressure levels. Hence the vertical variable in the xarray datasets in `dset_dict` will be calculated by using the formula:
# $$ P(i,j,k) = hyam(k) p0 + hybm(k) ps(i,j)$$
# to calculate the pressure
# 
# 
# 

# %%
# Rename datasets with different naming convention for constant hyam
for model in dset_dict.keys():
    if ('a' in list(dset_dict[model].keys())) == True:
        dset_dict[model] = dset_dict[model].rename({'a':'ap', 'a_bnds': 'ap_bnds'})
    if model == 'IPSL-CM6A-LR':
        dset_dict[model] = dset_dict[model].rename({'presnivs':'plev'})
    if model == 'IPSL-CM5A2-INCA':
        dset_dict[model] = dset_dict[model].rename({'lev':'plev'})
    

# %%
for model in dset_dict.keys():
    for var_id in dset_dict[model].keys():#['clw', 'cli']:
        if var_id == 'clw' or var_id == 'cli':
            # Convert the model level to isobaric levels
            #### ap, b, ps, p0
            if ('ap' in list(dset_dict[model].keys())) == True and \
                ('ps' in list(dset_dict[model].keys())) == True and \
                ('p0' in list(dset_dict[model].keys())) == True:
                if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
                    ('lev' in list(dset_dict[model]['ap'].coords)) == True and \
                    ('lev' in list(dset_dict[model]['b'].coords)) == True:
                        print(model, var_id, 'lev, ap, ps, p0')
                        # dset_dict[model][var_id] = gc.interpolation.interp_hybrid_to_pressure(data = dset_dict[model][var_id],
                        #                                                                                 ps   = dset_dict[model]['ps'], 
                        #                                                                                 hyam = dset_dict[model]['ap'], 
                        #                                                                                 hybm = dset_dict[model]['b'], 
                        #                                                                                 p0   = dset_dict[model]['p0'], 
                        #                                                                                 new_levels=new_levels,
                        #                                                                                 lev_dim='lev')
                        dset_dict[model]['plev'] = dset_dict[model]['ap']*dset_dict[model]['p0'] + dset_dict[model]['b']*dset_dict[model]['ps']
                        dset_dict[model]['plev'] = dset_dict[model]['plev'].transpose('time', 'lev','lat','lon')
                
                if ('plev' in list(dset_dict[model][var_id].coords)) == True:
                    print(model, var_id, 'variable on pressure levels', )
                # if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
                #     ('lev' in list(dset_dict[model]['ap'].coords)) == False and \
                #     ('lev' in list(dset_dict[model]['b'].coords)) == False:
                #         print(model, 'variable on pressure levels', 'lev, ap, ps,')
            # Convert the model level to isobaric levels
            #### ap, b, p0
            if ('ap' in list(dset_dict[model].keys())) == True and \
                ('ps' in list(dset_dict[model].keys())) == True and \
                ('p0' in list(dset_dict[model].keys())) == False:
                if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
                    ('lev' in list(dset_dict[model]['ap'].coords)) == True and \
                    ('lev' in list(dset_dict[model]['b'].coords)) == True:
                        print(model,var_id, 'lev, ap, ps,')
                        # dset_dict[model][var_id] = gc.interpolation.interp_hybrid_to_pressure(data = dset_dict[model][var_id],
                        #                                                                                 ps   = dset_dict[model]['ps'], 
                        #                                                                                 hyam = dset_dict[model]['ap'], 
                        #                                                                                 hybm = dset_dict[model]['b'], 
                        #                                                                                 new_levels=new_levels,
                        #                                                                                 lev_dim='lev')
                        dset_dict[model]['plev'] = dset_dict[model]['ap'] + dset_dict[model]['b']*dset_dict[model]['ps']
                        dset_dict[model]['plev'] = dset_dict[model]['plev'].transpose('time', 'lev','lat','lon')
                
                if ('plev' in list(dset_dict[model][var_id].coords)) == True:
                    print(model, var_id, 'variable on pressure levels', )
                
            if ('b' in list(dset_dict[model].keys())) == True and \
                ('orog' in list(dset_dict[model].keys())) == True:
                if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
                    ('lev' in list(dset_dict[model]['pfull'].coords)) == True:
                        print(model, 'hybrid height coordinate')
                


# %% [markdown]
# ## Calculate liquid water path from content

# %%
for model in dset_dict.keys():
    
    if ('plev' in list(dset_dict[model].keys())) == True:
        print(model, 'plev')
        _lwp = xr.DataArray(data=da.full(shape=dset_dict[model]['clw'].shape,fill_value=np.nan),
                                dims=dset_dict[model]['clw'].dims,
                                coords=dset_dict[model]['clw'].coords)
        # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
        for i in range(len(dset_dict[model]['lev'])-1):
            # calculate pressure difference between two levels
            dp = (dset_dict[model]['plev'].isel(lev=i) - dset_dict[model]['plev'].isel(lev=i+1))
            # calculate mean liquid water content between two layers
            dlwc = (dset_dict[model]['clw'].isel(lev=i) + dset_dict[model]['clw'].isel(lev=i+1))/2
            # calculate liquid water path between two layers
            _lwp[:,i,:,:] = dp[:,:,:]/9.81 * dlwc[:,:,:]
        
            # sum over all layers to ge the liquid water path in the atmospheric column
            dset_dict[model]['lwp'] = _lwp.sum(dim='lev',skipna=True)
            
            # assign attributes to data array
            dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs(dset_dict[model]['clw'].attrs)
            dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs({'long_name':'Liquid Water Path', 
                                                                            'units' : 'kg m-2',
                                                                                'mipTable':'', 'out_name': 'lwp',
                                                                                'standard_name': 'atmosphere_mass_content_of_cloud_liquid_water',
                                                                                'title': 'Liquid Water Path',
                                                                                'variable_id': 'lwp', 'original_units': 'kg/kg',
                                                                                'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate lwp with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
        # when ice water path does not exist
        if ('clivi' in list(dset_dict[model].keys())) == False:
            _iwp = xr.DataArray(data=da.full(shape=dset_dict[model]['cli'].shape,fill_value=np.nan),
                                    dims=dset_dict[model]['cli'].dims,
                                    coords=dset_dict[model]['cli'].coords)
            # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
            for i in range(len(dset_dict[model]['lev'])-1):
                # calculate pressure difference between two levels
                dp = (dset_dict[model]['plev'].isel(lev=i) - dset_dict[model]['plev'].isel(lev=i+1))
                # calculate mean liquid water content between two layers
                diwc = (dset_dict[model]['cli'].isel(lev=i) + dset_dict[model]['cli'].isel(lev=i+1))/2
                # calculate liquid water path between two layers
                _iwp[:,i,:,:] = dp[:,:,:]/9.81 * diwc[:,:,:]
            
                
                # sum over all layers to ge the Ice water path in the atmospheric column
                dset_dict[model]['clivi'] = _iwp.sum(dim='lev',skipna=True)
                
                # assign attributes to data array
                dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs(dset_dict[model]['cli'].attrs)
                dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs({'long_name':'Ice Water Path', 
                                                                                'units' : 'kg m-2',
                                                                                    'mipTable':'', 'out_name': 'clivi',
                                                                                    'standard_name': 'atmosphere_mass_content_of_cloud_ice_water',
                                                                                    'title': 'Ice Water Path',
                                                                                    'variable_id': 'clivi', 'original_units': 'kg/kg',
                                                                                    'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate clivi with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
            
    if ('plev' in list(dset_dict[model].coords)) == True:
        print(model, 'plev coord')
        _lwp = xr.DataArray(data=da.full(shape=dset_dict[model]['clw'].shape,fill_value=np.nan),
                                dims=dset_dict[model]['clw'].dims,
                                coords=dset_dict[model]['clw'].coords)
        # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
        for i in range(len(dset_dict[model]['plev'])-1):
            # calculate pressure difference between two levels
            dp = (dset_dict[model]['plev'].isel(plev=i) - dset_dict[model]['plev'].isel(plev=i+1))
            # calculate mean liquid water content between two layers
            dlwc = (dset_dict[model]['clw'].isel(plev=i) + dset_dict[model]['clw'].isel(plev=i+1))/2
            # calculate liquid water path between two layers
            _lwp[:,i,:,:] = dp/9.81 * dlwc[:,:,:]
        
            # sum over all layers to ge the liquid water path in the atmospheric column
            dset_dict[model]['lwp'] = _lwp.sum(dim='plev',skipna=True)
            
            # assign attributes to data array
            dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs(dset_dict[model]['clw'].attrs)
            dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs({'long_name':'Liquid Water Path', 
                                                                            'units' : 'kg m-2',
                                                                                'mipTable':'', 'out_name': 'lwp',
                                                                                'standard_name': 'atmosphere_mass_content_of_cloud_liquid_water',
                                                                                'title': 'Liquid Water Path',
                                                                                'variable_id': 'lwp', 'original_units': 'kg/kg',
                                                                                'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate lwp with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
        # when ice water path does not exist
        if ('clivi' in list(dset_dict[model].keys())) == False:
            _iwp = xr.DataArray(data=da.full(shape=dset_dict[model]['cli'].shape,fill_value=np.nan),
                                    dims=dset_dict[model]['cli'].dims,
                                    coords=dset_dict[model]['cli'].coords)
            # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
            for i in range(len(dset_dict[model]['plev'])-1):
                # calculate pressure difference between two levels
                dp = (dset_dict[model]['plev'].isel(plev=i) - dset_dict[model]['plev'].isel(plev=i+1))
                # calculate mean liquid water content between two layers
                diwc = (dset_dict[model]['cli'].isel(plev=i) + dset_dict[model]['cli'].isel(plev=i+1))/2
                # calculate liquid water path between two layers
                _iwp[:,i,:,:] = dp/9.81 * diwc[:,:,:]
            
                
                # sum over all layers to ge the Ice water path in the atmospheric column
                dset_dict[model]['clivi'] = _iwp.sum(dim='plev',skipna=True)
                
                # assign attributes to data array
                dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs(dset_dict[model]['clw'].attrs)
                dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs({'long_name':'Ice Water Path', 
                                                                                'units' : 'kg m-2',
                                                                                    'mipTable':'', 'out_name': 'clivi',
                                                                                    'standard_name': 'atmosphere_mass_content_of_cloud_ice_water',
                                                                                    'title': 'Ice Water Path',
                                                                                    'variable_id': 'clivi', 'original_units': 'kg/kg',
                                                                                    'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate clivi with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
            

# %%
for model in dset_dict.keys():
    for var in ['prsn', 'tas', 'clivi', 'lwp',]:
        for year in year_range:
            print('Writing files: var: {}, year: {}, model: {}'.format(var, year, model))
            (dset_dict[model][var].sel(time=slice(str(year)), lat=slice(45,90))).to_netcdf('{}/{}_{}_45_90_historical_{}_gn_{}0101-{}1231.nc'.format(cmip_in,var, model, dset_dict[model].attrs['variant_label'],year, year))
            (dset_dict[model][var].sel(time=slice(str(year)), lat=slice(-90,-45))).to_netcdf('{}/{}_{}_-90_-45_historical_{}_gn_{}0101-{}1231.nc'.format(cmip_in,var, model, dset_dict[model].attrs['variant_label'],year, year))
            
            # (dset_dict[model][var].sel(time=slice(str(year)), lat=slice(-90,-45))).to_netcdf('{}/{}_{}_-90_-45_historical_{}_gn_{}0101-{}1231.nc'.format(cmip_in,var, model, dset_dict[model].attrs['variant_label'],year, year))
            # (dset_dict[model][var].sel(time=slice(str(year)), lat=slice(45,90))).to_netcdf('{}/{}_{}_45_90_historical_{}_gn_{}0101-{}1231.nc'.format(cmip_in,var, model, dset_dict[model].attrs['variant_label'],year, year))

# %%


# %% [markdown]
# 
# <img src="https://drive.google.com/uc?id=1zb0LHvipx8JOXLLrCxzYToJM7eNK4eaw"  height="100" />
# <img src="https://reliance.rohub.org/static/media/Reliance-logo.433dc2e9.png"  height="100" />
# 
# <img src="https://www.uio.no/vrtx/decorating/resources/dist/src2/images/footer/uio-logo-en.svg"  height="100" />
# <img src="https://erc.europa.eu/sites/default/files/logo_0.png"  height="100" />
# 

# %% [markdown]
# 


