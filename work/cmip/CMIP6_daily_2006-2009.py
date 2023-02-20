# %%
# export PYTHONPATH="${PYTHONPATH}:/uio/kant/geo-metos-u1/franzihe/Documents/Python/globalsnow/CloudSat_ERA5_CMIP6_analysis/utils/"


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
# - horizonal resolution: depending on model
# - time resolution: daily mean atmospheric data (CFday, day)
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
# `/input/cmip6_hist/daily_means`.
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
UTILS_DIR = os.path.join(WORKDIR, 'utils/')

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
cmip_in = os.path.join(INPUT_DATA_DIR, 'cmip6_hist/daily_means/single_model')
cmip_out = os.path.join(OUTPUT_DATA_DIR, 'cmip6_hist/daily_means/common_grid')

# make output data directory
try:
    os.mkdir(cmip_out)
except OSError:
    pass

# %%
variable_id = ['clw', 'cli', 'clivi', 'tas', 'prsn', 'pr', 'areacella']

# %% [markdown]
# At the moment we have downloaded the end of the historical simulations for CMIP6 models. We define start and end year to ensure to only extract the 4-year period between 2006 and 2009.
# 
# $\rightarrow$ Define a start and end year
# 
# We will load all available models into one dictonary, which includes an xarray dataset with `xarray.open_mfdataset(file)` and select the time range [by name](https://xarray.pydata.org/en/stable/user-guide/indexing.html).

# %%
# source_id
list_models = [
<<<<<<< HEAD
            #    'MIROC6', 
            #    'CESM2', 
            #    'CanESM5', 
            #    'AWI-ESM-1-1-LR', 
            #    'MPI-ESM1-2-LR', 
            #    'UKESM1-0-LL', 
            #    'HadGEM3-GC31-LL',
            #    'CNRM-CM6-1',
            #    'CNRM-ESM2-1',
               'IPSL-CM6A-LR',
            #    'IPSL-CM5A2-INCA'
=======
               'MIROC6', #area
               'CESM2', #area
               'CanESM5', # area
               'AWI-ESM-1-1-LR', # area
               'MPI-ESM1-2-LR', # area
               # 'UKESM1-0-LL', 
               # 'HadGEM3-GC31-LL',
               'CNRM-CM6-1', #area
               'CNRM-ESM2-1',
               'IPSL-CM6A-LR', #area
               'IPSL-CM5A2-INCA' #area
>>>>>>> 2fa997a5cfb50ba087aaccdb18e76c2037dec00f
            ]

## experiment
experiment_id = ['historical']

## time resolution
t_res = ['day',]

# %%
starty = 2006; endy = 2006
year_range = range(starty, endy+1)



# %% [markdown]
# ## Search corresponding data
# Get the data required for the analysis. Define variables, models, experiment, and time resolution as defined in <a href="#data_wrangling">2. Data Wrangling</a>
# . 

# %%
def search_data(cmip_in, t_res, list_models, year_range):
    dset_dict = dict()
    for model in list_models:
        # print(model)
        cmip_file_in = glob('{}/*{}_{}_{}*'.format(cmip_in, t_res[0], model, experiment_id[0]))
        # get also areacella data
        # print(model)
        cmip_file_in.append(glob('{}/areacella*{}_{}*.nc'.format(cmip_in,model,  experiment_id[0]))[0])
        if len(cmip_file_in) != 0:
            dset_dict[model] = xr.open_mfdataset(sorted(cmip_file_in), combine='nested', compat='override', use_cftime=True, parallel =True)
            # select only years needed for analysis
            dset_dict[model] = dset_dict[model].sel(time = dset_dict[model]['time'].dt.year.isin(year_range)).squeeze()
            # shift longitude to be from -180 to 180
            dset_dict[model] = dset_dict[model].assign_coords(lon=(((dset_dict[model]['lon'] + 180) % 360) - 180)).sortby('lon').sortby('time')
        else:
            continue
    
    return dset_dict    

# %%
# variant_label = ['r10i1p1f1',
# 'r11i1p1f1',
# 'r1i1p1f1',
# 'r1i1p1f2',
# 'r1i1p2f1',
# 'r2i1p1f2',
# 'r5i1p1f3',]

# model = 'IPSL-CM6A-LR'
# model = list_models[5]
# # for model in list_models:
# cmip_file_in = sorted(glob('{}/*{}_{}_{}*'.format(cmip_in, t_res[0], model, experiment_id[0])))
# cmip_file_in.append(glob('{}/areacella*{}_{}*.nc'.format(cmip_in,model,  experiment_id[0]))[0])
# _cmip_file_in = []#dict()
# for i in range(len(cmip_file_in)):
#             # k = variant_label[0]
#         for k in variant_label:
#             if cmip_file_in[i].find(str(k)) != -1:
#                 print("Contains "+str(k), cmip_file_in[i][59:63])
#                 _cmip_file_in.append(cmip_file_in[i])
#             # else:
#             # # cmip_file_in.remove(cmip_file_in[i])
#             #     print(cmip_file_in[i])
# print(model, len(_cmip_file_in))

# %%
# dset_dict = search_data(cmip_in, t_res, list_models, year_range)


# %% [markdown]
# ## Assign attributes to the variables
#  
# We will assign the attributes to the variables as in ERA5 to make CMIP6 and ERA5 variables comperable.
#  
# * [`pr`](http://clipc-services.ceda.ac.uk/dreq/u/62f26742cf240c1b5169a5cd511196b6.html) and [`prsn`](http://clipc-services.ceda.ac.uk/dreq/u/051919eddec810e292c883205c944ceb.html) in **kg m-2 s-1** $\rightarrow$ Multiply by **3600** to get **mm h-1** $\rightarrow$ Multiply by **24** to get **mm day-1**
#  

# %%
def assign_att(dset):
    now = datetime.utcnow()
    # 
    for var_id in dset.keys():
            
            if var_id == 'prsn':
                dset[var_id] = dset[var_id]*3600*24
                dset[var_id] = dset[var_id].assign_attrs({'standard_name': 'Total snowfall per day',
        'comment': 'At surface; includes precipitation of all forms of water in the solid phase',
        'units': 'mm day-1',
        'original_units': 'kg m-2 s-1',
        'history': "{}Z altered by F. Hellmuth: Converted units from 'kg m-2 s-1' to 'kg m-2 day-1'.".format(now.strftime("%d/%m/%Y %H:%M:%S")),
        'cell_methods': 'area: time: mean',
        'cell_measures': 'area: areacella'})
                
                
    return dset

# %% [markdown]
# ## Interpolate from CMIP6 hybrid sigma-pressure levels to isobaric pressure levels
# 
# The vertical variables in the CMIP6 models are in hybrid sigma-pressure levels. Hence the vertical variable in the xarray datasets in `dset_dict` will be calculated by using the formula:
# $$ P(i,j,k) = hyam(k) p0 + hybm(k) ps(i,j)$$
# to calculate the pressure

# %%
def interp_hybrid_plev(dset, model):
# Rename datasets with different naming convention for constant hyam
    if ('a' in list(dset.keys())) == True:
        dset = dset.rename({'a':'ap', 'a_bnds': 'ap_bnds'})
    if ('nbnd' in list(dset.dims)) == True:
        dset = dset.rename({'nbnd':'bnds', })

    if model == 'IPSL-CM6A-LR':
        dset = dset.rename({'presnivs':'plev'})
    if model == 'IPSL-CM5A2-INCA':
        dset = dset.rename({'lev':'plev'})  
        
    if ('klevp1' in list(dset.dims)) == True:
        dset = dset.rename({'klevp1':'lev', })
        
    for var_id in dset.keys():#['clw', 'cli']:
        if var_id == 'clw' or var_id == 'cli':
            # Convert the model level to isobaric levels
            #### ap, b, ps, p0
            if ('ap' in list(dset.keys())) == True and \
                ('ps' in list(dset.keys())) == True and \
                ('p0' in list(dset.keys())) == True:
                if ('lev' in list(dset[var_id].coords)) == True and \
                    ('lev' in list(dset['ap'].coords)) == True and \
                    ('lev' in list(dset['b'].coords)) == True:
                        print(model, var_id, 'lev, ap, ps, p0')
                        # dset[var_id] = gc.interpolation.interp_hybrid_to_pressure(data = dset[var_id],
                        #                                                                                 ps   = dset['ps'], 
                        #                                                                                 hyam = dset['ap'], 
                        #                                                                                 hybm = dset['b'], 
                        #                                                                                 p0   = dset['p0'], 
                        #                                                                                 new_levels=new_levels,
                        #                                                                                 lev_dim='lev')
                        dset['plev'] = dset['ap']*dset['p0'] + dset['b']*dset['ps']
                        dset['plev'] = dset['plev'].transpose('time', 'lev','lat','lon')
                        
                        dset['plev_bnds'] = dset['ap_bnds']*dset['p0'] + dset['b_bnds']*dset['ps']
                        dset['plev_bnds'] = dset['plev_bnds'].transpose('time', 'lev','lat','lon', 'bnds')
                
                if ('plev' in list(dset[var_id].coords)) == True:
                    print(model, var_id, 'variable on pressure levels', )
                    dset['plev_bnds'] = dset['ap_bnds']*dset['p0'] + dset['b_bnds']*dset['ps']
                    dset['plev_bnds'] = dset['plev_bnds'].transpose('time', 'lev','lat','lon', 'bnds')
                # if ('lev' in list(dset[var_id].coords)) == True and \
                #     ('lev' in list(dset['ap'].coords)) == False and \
                #     ('lev' in list(dset['b'].coords)) == False:
                #         print(model, 'variable on pressure levels', 'lev, ap, ps,')
            # Convert the model level to isobaric levels
            #### ap, b, p0
            if ('ap' in list(dset.keys())) == True and \
                ('ps' in list(dset.keys())) == True and \
                ('p0' in list(dset.keys())) == False:
                if ('lev' in list(dset[var_id].coords)) == True and \
                    ('lev' in list(dset['ap'].coords)) == True and \
                    ('lev' in list(dset['b'].coords)) == True:
                        print(model,var_id, 'lev, ap, ps,')
                        # dset[var_id] = gc.interpolation.interp_hybrid_to_pressure(data = dset[var_id],
                        #                                                                                 ps   = dset['ps'], 
                        #                                                                                 hyam = dset['ap'], 
                        #                                                                                 hybm = dset['b'], 
                        #                                                                                 new_levels=new_levels,
                        #                                                                                 lev_dim='lev')
                        dset['plev'] = dset['ap'] + dset['b']*dset['ps']
                        dset['plev'] = dset['plev'].transpose('time', 'lev','lat','lon')
                        
                        dset['plev_bnds'] = dset['ap_bnds'] + dset['b_bnds']*dset['ps']
                        dset['plev_bnds'] = dset['plev_bnds'].transpose('time', 'lev','lat','lon', 'bnds')
                        
                        
                
                if ('plev' in list(dset[var_id].coords)) == True:
                    print(model, var_id, 'variable on pressure levels', )
                    dset['plev_bnds'] = dset['ap_bnds'] + dset['b_bnds']*dset['ps']
                    dset['plev_bnds'] = dset['plev_bnds'].transpose('time', 'lev','lat','lon', 'bnds')
                
            if ('b' in list(dset.keys())) == True and \
                ('orog' in list(dset.keys())) == True:
                if ('lev' in list(dset[var_id].coords)) == True and \
                    ('lev' in list(dset['pfull'].coords)) == True:
                        print(model, 'hybrid height coordinate')
                        
    dset = dset.transpose('time', 'lat', 'lon', 'plev', 'lev', 'bnds','axis_nbounds' , missing_dims="ignore" )
    
                
        
    return dset  

# %%
# for model in dset_dict.keys():
#     dset_dict[model] = interp_hybrid_plev(dset_dict[model],model)

# %% [markdown]
# ## Calculate liquid water path from content
# 
# Once the pressure levels are calculated the daily average LWP (IWP) is calculated for each CMIP6 model.
# \begin{equation}
#         LWP = \rho_{air} \cdot \Delta clw \cdot \Delta Z 
# \end{equation}
# 
# with hydrostatic equation
# 
# \begin{equation}
#          \frac{\Delta p}{\Delta Z}  = -\rho_{air} \cdot g  
# \end{equation}
# 
# \begin{equation}
#          \leftrightarrow LWP = - \frac{\rho_{air}}{\rho_{air} g} \cdot \Delta clw \Delta p
# \end{equation}
# 
# with $\Delta clw = clw(NLEV-k)$ and $\Delta p = p(NLEV-k + 1/2) - p(NLEV-k - 1/2)$ follows for the total liquid water path in the column:
# 
# \begin{equation}
#          -\frac{1}{g} \sum_{k=0}^{NLEV+1} LWP(k) = -\frac{1}{g} \sum_{k=0}^{NLEV+1} clw(NLEV-k) \cdot [p(NLEV-k + 1/2) - p(NLEV-k - 1/2)]
# \end{equation}
# 
# 

# %%
def calc_water_path(dset, model):
    now = datetime.utcnow()
    
    if ('plev' in list(dset.keys())) == True:
        print(model, 'plev')
        _lwp = xr.DataArray(data=da.full(shape=dset['clw'].shape,fill_value=np.nan),
                                dims=dset['clw'].dims,
                                coords=dset['clw'].coords)
        # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
        for i in range(len(dset['lev'])):
                        
            # calculate pressure difference between two levels, where the hight of the two layers is given in lev_bnds
            dp = (dset['plev_bnds'].isel(lev=i).diff(dim='bnds'))
            # use liquid water content between two layers, meaning at lev
            dlwc = (dset['clw'].isel(lev=i))
            # calculate liquid water path between two layers
            _lwp[:,:,:,i] = - dp[:,:,:,0]/9.81 * dlwc[:,:,:]
                
            # sum over all layers to ge the liquid water path in the atmospheric column
            dset['lwp'] = _lwp.sum(dim='lev',skipna=True)
            
            
            # assign attributes to data array
            dset['lwp'] = dset['lwp'].assign_attrs(dset['clw'].attrs)
            dset['lwp'] = dset['lwp'].assign_attrs({'long_name':'Daily average Liquid Water Path', 
                                                                            'units' : 'kg m-2',
                                                                                'mipTable':'', 'out_name': 'lwp',
                                                                                'standard_name': 'atmosphere_mass_content_of_cloud_liquid_water',
                                                                                'title': 'Liquid Water Path',
                                                                                'variable_id': 'lwp', 'original_units': 'kg/kg',
                                                                                'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate lwp with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
        # when ice water path does not exist
        if ('clivi' in list(dset.keys())) == False:
        # if ('clivi' in list(dset.keys())) == True:
            _iwp = xr.DataArray(data=da.full(shape=dset['cli'].shape,fill_value=np.nan),
                                    dims=dset['cli'].dims,
                                    coords=dset['cli'].coords)
            # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
            for i in range(len(dset['lev'])):
                                
                # calculate pressure difference between two levels, where the hight of the two layers is given in lev_bnds
                dp = (dset['plev_bnds'].isel(lev=i).diff(dim='bnds'))
                # use liquid water content between two layers, meaning at lev
                diwc = (dset['cli'].isel(lev=i))
                # calculate liquid water path between two layers
                _lwp[:,:,:,i] = - dp[:,:,:,0]/9.81 * dlwc[:,:,:]
                    
                # sum over all layers to ge the liquid water path in the atmospheric column
                dset['clivi'] = _iwp.sum(dim='lev',skipna=True)
                
                # assign attributes to data array
                dset['clivi'] = dset['clivi'].assign_attrs(dset['cli'].attrs)
                dset['clivi'] = dset['clivi'].assign_attrs({'long_name':'Daily average Ice Water Path', 
                                                                                'units' : 'kg m-2',
                                                                                    'mipTable':'', 'out_name': 'clivi',
                                                                                    'standard_name': 'atmosphere_mass_content_of_cloud_ice_water',
                                                                                    'title': 'Ice Water Path',
                                                                                    'variable_id': 'clivi', 'original_units': 'kg/kg',
                                                                                    'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate clivi with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
        if ('clivi' in list(dset.keys())) == True:
            dset['clivi'] = dset['clivi'].assign_attrs({'long_name':'Daily average Ice Water Path', 
                                                                                'units' : 'kg m-2',
                                                                                    'mipTable':'', 'out_name': 'clivi',
                                                                                    'standard_name': 'atmosphere_mass_content_of_cloud_ice_water',
                                                                                    'title': 'Ice Water Path',
                                                                                    'variable_id': 'clivi', 'original_units': 'kg/m2',
                                                                                    'history': "{}Z altered by F. Hellmuth: Rename attributes to daily average ice water path".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
            
            
    if ('plev' in list(dset.coords)) == True:
        print(model, 'plev coord')
        _lwp = xr.DataArray(data=da.full(shape=dset['clw'].shape,fill_value=np.nan),
                                dims=dset['clw'].dims,
                                coords=dset['clw'].coords)
        # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
        for i in range(len(dset['plev'])):
                        
            # calculate pressure difference between two levels, where the hight of the two layers is given in lev_bnds
            dp = (dset['plev_bnds'].isel(lev=i).diff(dim='bnds'))
            # use liquid water content between two layers, meaning at lev
            dlwc = (dset['clw'].isel(plev=i))
            # calculate liquid water path between two layers
            _lwp[:,:,:,i] = - dp[:,:,:,0]/9.81 * dlwc[:,:,:]
                
            # sum over all layers to ge the liquid water path in the atmospheric column
            dset['lwp'] = _lwp.sum(dim='plev',skipna=True)
            
            # assign attributes to data array
            dset['lwp'] = dset['lwp'].assign_attrs(dset['clw'].attrs)
            dset['lwp'] = dset['lwp'].assign_attrs({'long_name':'Daily average Liquid Water Path', 
                                                                            'units' : 'kg m-2',
                                                                                'mipTable':'', 'out_name': 'lwp',
                                                                                'standard_name': 'atmosphere_mass_content_of_cloud_liquid_water',
                                                                                'title': 'Liquid Water Path',
                                                                                'variable_id': 'lwp', 'original_units': 'kg/kg',
                                                                                'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate lwp with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
        # when ice water path does not exist
        if ('clivi' in list(dset.keys())) == False:
        # if ('clivi' in list(dset.keys())) == True:
            _iwp = xr.DataArray(data=da.full(shape=dset['cli'].shape,fill_value=np.nan),
                                    dims=dset['cli'].dims,
                                    coords=dset['cli'].coords)
            # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
            for i in range(len(dset['plev'])):
                                
                # calculate pressure difference between two levels, where the hight of the two layers is given in lev_bnds
                dp = (dset['plev_bnds'].isel(lev=i).diff(dim='bnds'))
                # use liquid water content between two layers, meaning at lev
                diwc = (dset['cli'].isel(lev=i))
                # calculate liquid water path between two layers
                _lwp[:,:,:,i] = - dp[:,:,:,0]/9.81 * dlwc[:,:,:]
                    
                # sum over all layers to ge the liquid water path in the atmospheric column
                dset['clivi'] = _iwp.sum(dim='lev',skipna=True)
                
                # assign attributes to data array
                dset['clivi'] = dset['clivi'].assign_attrs(dset['clw'].attrs)
                dset['clivi'] = dset['clivi'].assign_attrs({'long_name':'Daily average Ice Water Path', 
                                                                                'units' : 'kg m-2',
                                                                                    'mipTable':'', 'out_name': 'clivi',
                                                                                    'standard_name': 'atmosphere_mass_content_of_cloud_ice_water',
                                                                                    'title': 'Ice Water Path',
                                                                                    'variable_id': 'clivi', 'original_units': 'kg/kg',
                                                                                    'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate clivi with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
            
        
        if ('clivi' in list(dset.keys())) == True:
            dset['clivi'] = dset['clivi'].assign_attrs({'long_name':'Daily average Ice Water Path', 
                                                                                'units' : 'kg m-2',
                                                                                    'mipTable':'', 'out_name': 'clivi',
                                                                                    'standard_name': 'atmosphere_mass_content_of_cloud_ice_water',
                                                                                    'title': 'Ice Water Path',
                                                                                    'variable_id': 'clivi', 'original_units': 'kg/m2',
                                                                                    'history': "{}Z altered by F. Hellmuth: Rename attributes to daily average ice water path".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
               
    return dset

    

# %%
# for model in dset_dict.keys():
#     dset_dict[model] = calc_water_path(dset_dict[model], model)

# %%
def process(cmip_in, t_res, list_models, year_range):
    
    dset_dict = search_data(cmip_in, t_res, list_models, year_range)
    for model in dset_dict.keys():
        # dset_dict[model] = assign_att(dset_dict[model])
        dset_dict[model] = interp_hybrid_plev(dset_dict[model],model)
        dset_dict[model] = calc_water_path(dset_dict[model], model)
        
        dset_dict[model] = dset_dict[model][['prsn', 'tas', 'clivi', 'lwp',]]
    
    return dset_dict

# %%
dset_dict = process(cmip_in, t_res, list_models, year_range)

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
# dset_dict = search_data(cmip_in, t_res, list_models, year_range)
# dset_dict = dict()
# for model in list_models:
#     cmip_file_in = glob('{}/*{}_{}_{}*'.format(cmip_in, t_res[0], model, experiment_id[0]))
#     if len(cmip_file_in) != 0:
#         dset_dict[model] = xr.open_mfdataset(sorted(cmip_file_in), combine='nested', compat='override', use_cftime=True)
#         # select only years needed for analysis
#         dset_dict[model] = dset_dict[model].sel(time = dset_dict[model]['time'].dt.year.isin(year_range)).squeeze()
#         # shift longitude to be from -180 to 180
#         dset_dict[model] = dset_dict[model].assign_coords(lon=(((dset_dict[model]['lon'] + 180) % 360) - 180)).sortby('lon').sortby('time')
#     else:
#         continue

# %%
# now = datetime.utcnow()
# for model in dset_dict.keys():
# # 
#     for var_id in dset_dict[model].keys():
         
#         if var_id == 'prsn':
#             dset_dict[model][var_id] = dset_dict[model][var_id]*3600
#             dset_dict[model][var_id] = dset_dict[model][var_id].assign_attrs({'standard_name': 'snowfall_flux',
#     'long_name': 'Snowfall Flux',
#     'comment': 'At surface; includes precipitation of all forms of water in the solid phase',
#     'units': 'mm h-1',
#     'original_units': 'kg m-2 s-1',
#     'history': "{}Z altered by F. Hellmuth: Converted units from 'kg m-2 s-1' to 'mm h-1'.".format(now.strftime("%d/%m/%Y %H:%M:%S")),
#     'cell_methods': 'area: time: mean',
#     'cell_measures': 'area: areacella'})

# %%
# # Rename datasets with different naming convention for constant hyam
# for model in dset_dict.keys():
#     if ('a' in list(dset_dict[model].keys())) == True:
#         dset_dict[model] = dset_dict[model].rename({'a':'ap', 'a_bnds': 'ap_bnds'})
#     if model == 'IPSL-CM6A-LR':
#         dset_dict[model] = dset_dict[model].rename({'presnivs':'plev'})
#     if model == 'IPSL-CM5A2-INCA':
#         dset_dict[model] = dset_dict[model].rename({'lev':'plev'})
    


# %%
# for model in dset_dict.keys():
#     for var_id in dset_dict[model].keys():#['clw', 'cli']:
#         if var_id == 'clw' or var_id == 'cli':
#             # Convert the model level to isobaric levels
#             #### ap, b, ps, p0
#             if ('ap' in list(dset_dict[model].keys())) == True and \
#                 ('ps' in list(dset_dict[model].keys())) == True and \
#                 ('p0' in list(dset_dict[model].keys())) == True:
#                 if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
#                     ('lev' in list(dset_dict[model]['ap'].coords)) == True and \
#                     ('lev' in list(dset_dict[model]['b'].coords)) == True:
#                         print(model, var_id, 'lev, ap, ps, p0')
#                         # dset_dict[model][var_id] = gc.interpolation.interp_hybrid_to_pressure(data = dset_dict[model][var_id],
#                         #                                                                                 ps   = dset_dict[model]['ps'], 
#                         #                                                                                 hyam = dset_dict[model]['ap'], 
#                         #                                                                                 hybm = dset_dict[model]['b'], 
#                         #                                                                                 p0   = dset_dict[model]['p0'], 
#                         #                                                                                 new_levels=new_levels,
#                         #                                                                                 lev_dim='lev')
#                         dset_dict[model]['plev'] = dset_dict[model]['ap']*dset_dict[model]['p0'] + dset_dict[model]['b']*dset_dict[model]['ps']
#                         dset_dict[model]['plev'] = dset_dict[model]['plev'].transpose('time', 'lev','lat','lon')
                
#                 if ('plev' in list(dset_dict[model][var_id].coords)) == True:
#                     print(model, var_id, 'variable on pressure levels', )
#                 # if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
#                 #     ('lev' in list(dset_dict[model]['ap'].coords)) == False and \
#                 #     ('lev' in list(dset_dict[model]['b'].coords)) == False:
#                 #         print(model, 'variable on pressure levels', 'lev, ap, ps,')
#             # Convert the model level to isobaric levels
#             #### ap, b, p0
#             if ('ap' in list(dset_dict[model].keys())) == True and \
#                 ('ps' in list(dset_dict[model].keys())) == True and \
#                 ('p0' in list(dset_dict[model].keys())) == False:
#                 if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
#                     ('lev' in list(dset_dict[model]['ap'].coords)) == True and \
#                     ('lev' in list(dset_dict[model]['b'].coords)) == True:
#                         print(model,var_id, 'lev, ap, ps,')
#                         # dset_dict[model][var_id] = gc.interpolation.interp_hybrid_to_pressure(data = dset_dict[model][var_id],
#                         #                                                                                 ps   = dset_dict[model]['ps'], 
#                         #                                                                                 hyam = dset_dict[model]['ap'], 
#                         #                                                                                 hybm = dset_dict[model]['b'], 
#                         #                                                                                 new_levels=new_levels,
#                         #                                                                                 lev_dim='lev')
#                         dset_dict[model]['plev'] = dset_dict[model]['ap'] + dset_dict[model]['b']*dset_dict[model]['ps']
#                         dset_dict[model]['plev'] = dset_dict[model]['plev'].transpose('time', 'lev','lat','lon')
                
#                 if ('plev' in list(dset_dict[model][var_id].coords)) == True:
#                     print(model, var_id, 'variable on pressure levels', )
                
#             if ('b' in list(dset_dict[model].keys())) == True and \
#                 ('orog' in list(dset_dict[model].keys())) == True:
#                 if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
#                     ('lev' in list(dset_dict[model]['pfull'].coords)) == True:
#                         print(model, 'hybrid height coordinate')
                

# %%

# for model in dset_dict.keys():
    
#     if ('plev' in list(dset_dict[model].keys())) == True:
#         print(model, 'plev')
#         _lwp = xr.DataArray(data=da.full(shape=dset_dict[model]['clw'].shape,fill_value=np.nan),
#                                 dims=dset_dict[model]['clw'].dims,
#                                 coords=dset_dict[model]['clw'].coords)
#         # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
#         for i in range(len(dset_dict[model]['lev'])-1):
#             # calculate pressure difference between two levels
#             dp = (dset_dict[model]['plev'].isel(lev=i) - dset_dict[model]['plev'].isel(lev=i+1))
#             # calculate mean liquid water content between two layers
#             dlwc = (dset_dict[model]['clw'].isel(lev=i) + dset_dict[model]['clw'].isel(lev=i+1))/2
#             # calculate liquid water path between two layers
#             _lwp[:,i,:,:] = dp[:,:,:]/9.81 * dlwc[:,:,:]
        
#             # sum over all layers to ge the liquid water path in the atmospheric column
#             dset_dict[model]['lwp'] = _lwp.sum(dim='lev',skipna=True)
            
#             # assign attributes to data array
#             dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs(dset_dict[model]['clw'].attrs)
#             dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs({'long_name':'Liquid Water Path', 
#                                                                             'units' : 'kg m-2',
#                                                                                 'mipTable':'', 'out_name': 'lwp',
#                                                                                 'standard_name': 'atmosphere_mass_content_of_cloud_liquid_water',
#                                                                                 'title': 'Liquid Water Path',
#                                                                                 'variable_id': 'lwp', 'original_units': 'kg/kg',
#                                                                                 'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate lwp with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
#         # when ice water path does not exist
#         if ('clivi' in list(dset_dict[model].keys())) == False:
#             _iwp = xr.DataArray(data=da.full(shape=dset_dict[model]['cli'].shape,fill_value=np.nan),
#                                     dims=dset_dict[model]['cli'].dims,
#                                     coords=dset_dict[model]['cli'].coords)
#             # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
#             for i in range(len(dset_dict[model]['lev'])-1):
#                 # calculate pressure difference between two levels
#                 dp = (dset_dict[model]['plev'].isel(lev=i) - dset_dict[model]['plev'].isel(lev=i+1))
#                 # calculate mean liquid water content between two layers
#                 diwc = (dset_dict[model]['cli'].isel(lev=i) + dset_dict[model]['cli'].isel(lev=i+1))/2
#                 # calculate liquid water path between two layers
#                 _iwp[:,i,:,:] = dp[:,:,:]/9.81 * diwc[:,:,:]
            
                
#                 # sum over all layers to ge the Ice water path in the atmospheric column
#                 dset_dict[model]['clivi'] = _iwp.sum(dim='lev',skipna=True)
                
#                 # assign attributes to data array
#                 dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs(dset_dict[model]['cli'].attrs)
#                 dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs({'long_name':'Ice Water Path', 
#                                                                                 'units' : 'kg m-2',
#                                                                                     'mipTable':'', 'out_name': 'clivi',
#                                                                                     'standard_name': 'atmosphere_mass_content_of_cloud_ice_water',
#                                                                                     'title': 'Ice Water Path',
#                                                                                     'variable_id': 'clivi', 'original_units': 'kg/kg',
#                                                                                     'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate clivi with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
            
#     if ('plev' in list(dset_dict[model].coords)) == True:
#         print(model, 'plev coord')
#         _lwp = xr.DataArray(data=da.full(shape=dset_dict[model]['clw'].shape,fill_value=np.nan),
#                                 dims=dset_dict[model]['clw'].dims,
#                                 coords=dset_dict[model]['clw'].coords)
#         # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
#         for i in range(len(dset_dict[model]['plev'])-1):
#             # calculate pressure difference between two levels
#             dp = (dset_dict[model]['plev'].isel(plev=i) - dset_dict[model]['plev'].isel(plev=i+1))
#             # calculate mean liquid water content between two layers
#             dlwc = (dset_dict[model]['clw'].isel(plev=i) + dset_dict[model]['clw'].isel(plev=i+1))/2
#             # calculate liquid water path between two layers
#             _lwp[:,i,:,:] = dp/9.81 * dlwc[:,:,:]
        
#             # sum over all layers to ge the liquid water path in the atmospheric column
#             dset_dict[model]['lwp'] = _lwp.sum(dim='plev',skipna=True)
            
#             # assign attributes to data array
#             dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs(dset_dict[model]['clw'].attrs)
#             dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs({'long_name':'Liquid Water Path', 
#                                                                             'units' : 'kg m-2',
#                                                                                 'mipTable':'', 'out_name': 'lwp',
#                                                                                 'standard_name': 'atmosphere_mass_content_of_cloud_liquid_water',
#                                                                                 'title': 'Liquid Water Path',
#                                                                                 'variable_id': 'lwp', 'original_units': 'kg/kg',
#                                                                                 'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate lwp with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
#         # when ice water path does not exist
#         if ('clivi' in list(dset_dict[model].keys())) == False:
#             _iwp = xr.DataArray(data=da.full(shape=dset_dict[model]['cli'].shape,fill_value=np.nan),
#                                     dims=dset_dict[model]['cli'].dims,
#                                     coords=dset_dict[model]['cli'].coords)
#             # lev2 is the atmospheric pressure, lower in the atmosphere than lev. Sigma-pressure coordinates are from 1 to 0, with 1 at the surface
#             for i in range(len(dset_dict[model]['plev'])-1):
#                 # calculate pressure difference between two levels
#                 dp = (dset_dict[model]['plev'].isel(plev=i) - dset_dict[model]['plev'].isel(plev=i+1))
#                 # calculate mean liquid water content between two layers
#                 diwc = (dset_dict[model]['cli'].isel(plev=i) + dset_dict[model]['cli'].isel(plev=i+1))/2
#                 # calculate liquid water path between two layers
#                 _iwp[:,i,:,:] = dp/9.81 * diwc[:,:,:]
            
                
#                 # sum over all layers to ge the Ice water path in the atmospheric column
#                 dset_dict[model]['clivi'] = _iwp.sum(dim='plev',skipna=True)
                
#                 # assign attributes to data array
#                 dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs(dset_dict[model]['clw'].attrs)
#                 dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs({'long_name':'Ice Water Path', 
#                                                                                 'units' : 'kg m-2',
#                                                                                     'mipTable':'', 'out_name': 'clivi',
#                                                                                     'standard_name': 'atmosphere_mass_content_of_cloud_ice_water',
#                                                                                     'title': 'Ice Water Path',
#                                                                                     'variable_id': 'clivi', 'original_units': 'kg/kg',
#                                                                                     'history': "{}Z altered by F. Hellmuth: Interpolate data from hybrid-sigma levels to isobaric levels with P=a*p0 + b*psfc. Calculate clivi with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
            


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


