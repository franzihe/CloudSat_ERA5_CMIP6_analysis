# %% [markdown]
# # Example with high-resolution CMIP6 models (~100 km) using Pangeo catalog 
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
# * Retrieve CMIP6 data through [Pangeo](https://pangeo-data.github.io/pangeo-cmip6-cloud/)
# * Hybrid sigma-pressure coordinates to isobaric pressure levels of the European Centre for Medium-Range Weather Forecast Re-Analysis 5 (ERA5) with [GeoCAT-comb](https://geocat-comp.readthedocs.io/en/latest/index.html)
# * Regridd the CMIP6 variables to the exact horizontal resolution with [`xesmf`](https://xesmf.readthedocs.io/en/latest/)
# * Calculate an ensemble mean of all used models
# * Calculate and plot the seasonal mean of the ensemble mean
# 
# **Questions**
# * How is the cloud phase and snowfall varying between 1985 and 2014?
# 
# > **_NOTE:_** We answer questions related to the comparison of CMIP models to ERA5 in another [Jupyter Notebook](../CMIP6_ERA5_CloudSat/plt_seasonal_mean.ipynb).

# %% [markdown]
# # 2. Data Wrangling <a id='data_wrangling'></a>
# 
# This study will compare surface snowfall, ice, and liquid water content from the Coupled Model Intercomparison Project Phase 6 ([CMIP6](https://esgf-node.llnl.gov/projects/cmip6/)) climate models (accessed through [Pangeo](https://pangeo.io/)) to the European Centre for Medium-Range Weather Forecast Re-Analysis 5 ([ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)) data from **1985 to 2014**. We conduct statistical analysis at the annual and seasonal timescales to determine the biases in cloud phase and precipitation (liquid and solid) in the CMIP6 models and their potential connection between them. 
# 
# - Time period: 1985 to 2014
# - horizonal resolution: ~100km
# - time resolution: monthly atmospheric data (Amon, AERmon)
# - Variables:
#   
# | shortname     |             Long name                   |      Units    |  levels |
# | ------------- |:---------------------------------------:| -------------:|--------:|
# |  prsn         |    Snowfall Flux                        | [kg m-2 s-1]  | surface |
# | clw           |    Mass Fraction of Cloud Liquid Water  |  [kg kg-1]    |    ml   | 
# |               |                                         | to calculate lwp use integral clw -dp/dg | |
# | cli           |    Mass Fraction of Cloud Ice           | [kg kg-1]     |    ml   |
# | tas           |    Near-Surface Air Temperature         |   [K]         | surface |
# | ta            |    Air Temperature                      |  [K]          |  plev   |
# | clivi         |    Ice Water Path                       | [kg m-2]      |         |
# | lwp           |    Liquid Water Path                    | [kg m-2]      |         |
# | pr            |    Precipitation                        | [kg m-2 s-1]  | surface |
# 
# - CMIP6 models:
# 
# | Institution                                            |     Model name    | Reference                                                     |
# | ------------------------------------------------------ |:-----------------:|--------------------------------------------------------------:|
# | [AS-RCEC](https://www.rcec.sinica.edu.tw/index_en.php) | TaiESM1           | [Lee et al. (2020)](https://doi.org/10.5194/gmd-13-3887-2020) |
# | [BCC](http://bcc.ncc-cma.net/)                         | BCC-CSM2-M        | [Wu et al. (2019)](https://doi.org/10.5194/gmd-12-1573-2019)  |
# | [CAMS](http://www.cma.gov.cn/en2014/)                  | CAMS-CSM1-0       |                                                               |
# | [CAS](http://english.iap.cas.cn/)                      | FGOALS-f3-L       | [Bian et al. (2020)](https://doi-org.ezproxy.uio.no/10.1080/16742834.2020.1778419) |
# | [CMCC](https://www.cmcc.it/)                           | CMCC-CM2-SR5      | [Cherchi et al. (2019)](https://doi-org.ezproxy.uio.no/10.1029/2018MS001369)|
# |                                                        | CMCC-CM2-HR4      | [Cherchi et al. (2019)](https://doi-org.ezproxy.uio.no/10.1029/2018MS001369)|
# |                                                        | CMCC-ESM2         | [CMCC website](https://www.cmcc.it/models/cmcc-esm-earth-system-model)    |
# | [EC-Earth-Consortium](http://www.ec-earth.org/)        | EC-Earth3-AerChem | [van Noije et al. (2021)](https://doi.org/10.5194/gmd-14-5637-2021)  |
# | [E3SM-Project](https://e3sm.org/)                      | E3SM-1-1          | [Golaz et al. (2019)](https://doi-org.ezproxy.uio.no/10.1029/2018MS001603); [Burrows et al. (2020)](https://doi-org.ezproxy.uio.no/10.1029/2019MS001766) Text S8|
# |                                                        | E3SM-1-1-ECA      | |
# | [MPI-M](https://mpimet.mpg.de/en/homepage)             | MPI-ESM1-2-HR     | [Müller et al. (2018)](https://doi-org.ezproxy.uio.no/10.1029/2017MS001217)|
# | [MRI](https://www.mri-jma.go.jp/index_en.html)         | MRI-ESM2-0        | [Yukimoto et al. (2019)](https://doi.org/10.2151/jmsj.2019-051) |
# | [NCC](https://folk.uib.no/ngfhd/EarthClim/index.htm)   | NorESM2-MM        | [Seland et al. (2020)](https://doi.org/10.5194/gmd-13-6165-2020)|
# | [NOAA-GFDL](https://www.gfdl.noaa.gov/)                | GFDL-CM4          | [Held et al. (2019)](https://doi-org.ezproxy.uio.no/10.1029/2019MS001829) |
# |                                                        | GFDL-ESM4         | [Dunne et al. (2020)](https://doi-org.ezproxy.uio.no/10.1029/2019MS002015) |
# | [SNU](https://en.snu.ac.kr/index.html)                 | SAM0-UNICON       | [Park et al. (2019)](https://doi-org.ezproxy.uio.no/10.1175/JCLI-D-18-0796.1) |
# | [THU](https://www.tsinghua.edu.cn/en/)                 | CIESM             | [Lin et al. (2020)](https://doi-org.ezproxy.uio.no/10.1029/2019MS002036) |
# 

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

# %%
# reload imports
# %load_ext autoreload
# %autoreload 2

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
list_models = ['MIROC6', 
               'CESM2', 
               'CanESM5', 
               'AWI-ESM-1-1-LR', 
               'MPI-ESM1-2-LR', 
            #    'UKESM1-0-LL', 
            #    'HadGEM3-GC31-LL',
               'CNRM-CM6-1',
               'CNRM-ESM2-1',
               'IPSL-CM6A-LR',
               'IPSL-CM5A2-INCA']

## experiment
experiment_id = ['historical']

## time resolution
t_res = ['day',]

# %% [markdown]
# ## Search corresponding data
# Get the data required for the analysis. Define variables, models, experiment, and time resolution as defined in <a href="#data_wrangling">2. Data Wrangling</a>
# . 

# %%
starty = 2006; endy = 2009
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
# ## Calendar
# Not all models in CMIP6 use the same calendar. Hence we double check the time axis. Later, when we regrid to the same horizontal resolution (<a href="#regrid_hz">Regrid CMIP6 data</a>) we will assign the same calendars for each model. 

# %%
# metadata of the historical run:
_d2 = pd.Series(["calendar",
                 "branch_time_in_parent", #"parent_activity_id", "parent_experiment_id",	"parent_mip_era",
                 "parent_source_id",#"parent_sub_experiment_id", 
                 "parent_time_units",# "parent_variant_label"
                  ])
_d2 = pd.DataFrame(_d2).rename(columns={0:'index'})
for model in dset_dict.keys():
    _data = []
    _names =[]
    _data.append(dset_dict[model].time.to_index().calendar)
    for k, v in dset_dict[model].attrs.items():
        
        if 'parent_time_units' in k or 'branch_time_in_parent' in k or 'parent_source_id' in k:
            _data.append(v)
            _names.append(k)
    _d2 = pd.concat([_d2,   pd.Series(_data)], axis=1)

_d2.dropna(how='all', axis=1, inplace=True)
_d2 = _d2.set_index('index')
_d2.columns = _d2.loc['parent_source_id']
_d2.drop('parent_source_id').T

# %% [markdown]
# ## Show attributes and individual identifier
# ... is going to be the reference model for the horizontal grid. The `xarray` datasets inside `dset_dict` can be extracted as any value in a Python dictionary.
# 
# The dictonary key is the source_id from `list_models`.

# %%
for model in dset_dict.keys():
    print('Institution: {}, \
          Model: {},   \
          Nominal res: {},  \
          lon x lat, level, top,: {}, \
          tracking_id: {}'.format(dset_dict[model].attrs['institution_id'],
                                  dset_dict[model].attrs['source_id'], 
                                  dset_dict[model].attrs['nominal_resolution'], 
                                  dset_dict[model].attrs['source'],
                                  dset_dict[model].attrs['tracking_id']))

# %% [markdown]
# ## Assign attributes to the variables
# 
# We will assign the attributes to the variables as in ERA5 to make CMIP6 and ERA5 variables comperable.
# 
# * [`cli`](http://clipc-services.ceda.ac.uk/dreq/u/dd916e3e2eca18cda5d9f81749d0c91c.html) and [`clw`](http://clipc-services.ceda.ac.uk/dreq/u/86b2b3318a73839edfafa9d46864aadc.html) in **kg kg-1** $\rightarrow$ Multiply by **1000** to get **g kg-1**
# * [`clivi`](http://clipc-services.ceda.ac.uk/dreq/u/73c496f5669cc122cf1cddfe4df2a27a.html) and [`lwp`](http://clipc-services.ceda.ac.uk/dreq/u/e6b31a1928879fcd3c92fe7b592f070e.html) in **kg m-2** $\rightarrow$ Multiply by **1000** to get **g m-2**
# * [`pr`](http://clipc-services.ceda.ac.uk/dreq/u/62f26742cf240c1b5169a5cd511196b6.html) and [`prsn`](http://clipc-services.ceda.ac.uk/dreq/u/051919eddec810e292c883205c944ceb.html) in **kg m-2 s-1** $\rightarrow$ Multiply by **3600** to get **mm h-1**
# 

# %%
now = datetime.utcnow()
for model in dset_dict.keys():
# 
    for var_id in dset_dict[model].keys():
        if var_id == 'clivi' or var_id == 'clw' or var_id == 'cli':
            dset_dict[model][var_id] = dset_dict[model][var_id]*1000
        if var_id == 'cli':
            dset_dict[model][var_id] = dset_dict[model][var_id].assign_attrs({'standard_name': 'mass_fraction_of_cloud_ice_in_air',
    'long_name': 'Mass Fraction of Cloud Ice',
    'comment': 'Includes both large-scale and convective cloud. This is calculated as the mass of cloud ice in the grid cell divided by the mass of air (including the water in all phases) in the grid cell. It includes precipitating hydrometeors ONLY if the precipitating hydrometeors affect the calculation of radiative transfer in model.',
    'units': 'g kg-1',
    'original_units': 'kg/kg',
    'history': "{}Z altered by F. Hellmuth: Converted units from 'kg kg-1' to 'g kg-1'. Interpolate data from hybrid-sigma levels to isobaric levels with geocat.comp.interpolation.interp_hybrid_to_pressure".format(now.strftime("%d/%m/%Y %H:%M:%S")),
    'cell_methods': 'area: time: mean',
    'cell_measures': 'area: areacella'})
                
        if var_id == 'clw':
            dset_dict[model][var_id] = dset_dict[model][var_id].assign_attrs({'standard_name': 'mass_fraction_of_cloud_liquid_water_in_air',
    'long_name': 'Mass Fraction of Cloud Liquid Water',
    'comment': 'Includes both large-scale and convective cloud. Calculate as the mass of cloud liquid water in the grid cell divided by the mass of air (including the water in all phases) in the grid cells. Precipitating hydrometeors are included ONLY if the precipitating hydrometeors affect the calculation of radiative transfer in model.',
    'units': 'g kg-1',
    'original_units': 'kg/kg',
    'history': "{}Z altered by F. Hellmuth: Converted units from 'kg kg-1' to 'g kg-1'. Interpolate data from hybrid-sigma levels to isobaric levels with geocat.comp.interpolation.interp_hybrid_to_pressure".format(now.strftime("%d/%m/%Y %H:%M:%S")),
    'cell_methods': 'area: time: mean',
    'cell_measures': 'area: areacella'})
            
        if var_id == 'clivi':
            dset_dict[model][var_id] = dset_dict[model][var_id].assign_attrs({'standard_name': 'atmosphere_mass_content_of_cloud_ice',
    'long_name': 'Ice Water Path',
    'comment': 'mass of ice water in the column divided by the area of the column (not just the area of the cloudy portion of the column). Includes precipitating frozen hydrometeors ONLY if the precipitating hydrometeor affects the calculation of radiative transfer in model.',
    'units': 'g m-2',
    'original_units': 'kg m-2',
    'history': "{}Z altered by F. Hellmuth: Converted units from 'kg m-2' to 'g m-2'.".format(now.strftime("%d/%m/%Y %H:%M:%S")),
    'cell_methods': 'area: time: mean',
    'cell_measures': 'area: areacella',})         
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
# The vertical variables in the CMIP6 models are in hybrid sigma-pressure levels. Hence the vertical variable in the xarray datasets in `dset_dict` will be calculated by using the [GeoCAT-comb](https://geocat-comp.readthedocs.io/en/latest/index.html#) function to [interpolate data from hybrid-sigma levels to isobaric levels](https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.interpolation.interp_hybrid_to_pressure.html#geocat.comp.interpolation.interp_hybrid_to_pressure).
# 
# The GeoCAT-comb function takes the following input:
# * `data`:   Multidimensional data array, which holds hybrid-sigma levels and has a lev_dim coordinate.
# * `ps`:     A multi-dimensional array of surface pressures (Pa), same time/space shape as data. Not all variables include the surface pressure, hence we will search the `Pangeo.io` catalog to find the surface pressure associated with the model. 
# * `hyam`:     One-dimensional arrays containing the hybrid A coefficients. Must have the same dimension size as the lev_dim dimension of data.
# * `hybm`:     One-dimensional arrays containing the hybrid B coefficients. Must have the same dimension size as the lev_dim dimension of data.
# * `p0`:       Scalar numeric value equal to surface reference pressure (Pa). Defaults to 100000 Pa.
# * `new_levels`: A one-dimensional array of output pressure levels (Pa). We will use the pressure of `air_temperature` (19 levels).
# 
# 
# $$ P(i,j,k) = hyam(k) p0 + hybm(k) ps(i,j)$$
# 
# ```
# import geocat
# 
# geocat.comp.interpolation.interp_hybrid_to_pressure(data      =ds['variable'], 
#                                                     ps        =ds['ps'], 
#                                                     hyam      =ds['a'], 
#                                                     hybm      =ds['b'], 
#                                                     p0        =ds['p0'], 
#                                                     new_levels=ds['plev'])
# ```
# 

# %%
new_levels = np.array([100000., 97500., 95000., 92500., 90000., 
                       87500., 85000., 82500., 80000.,
                       77500., 75000., 70000., 
                       65000., 60000., 
                       55000., 50000., 
                       45000., 40000., 
                       35000., 30000., 
                       25000., 22500., 20000., 
                       17500., 15000., 12500., 10000., 
                       7000., 5000., 3000., 2000., 1000., 700., 500., 300., 200., 100.], dtype=np.float32)

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
                        print(model, 'lev, ap, ps, p0')
                        dset_dict[model][var_id] = gc.interpolation.interp_hybrid_to_pressure(data = dset_dict[model][var_id],
                                                                                                        ps   = dset_dict[model]['ps'], 
                                                                                                        hyam = dset_dict[model]['ap'], 
                                                                                                        hybm = dset_dict[model]['b'], 
                                                                                                        p0   = dset_dict[model]['p0'], 
                                                                                                        new_levels=new_levels,
                                                                                                        lev_dim='lev')

                
                if ('plev' in list(dset_dict[model][var_id].coords)) == True:
                    print(model, 'variable on pressure levels', )
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
                        print(model, 'lev, ap, ps,')
                        dset_dict[model][var_id] = gc.interpolation.interp_hybrid_to_pressure(data = dset_dict[model][var_id],
                                                                                                        ps   = dset_dict[model]['ps'], 
                                                                                                        hyam = dset_dict[model]['ap'], 
                                                                                                        hybm = dset_dict[model]['b'], 
                                                                                                        new_levels=new_levels,
                                                                                                        lev_dim='lev')

                
                if ('plev' in list(dset_dict[model][var_id].coords)) == True:
                    print(model, 'variable on pressure levels', )
                
            if ('b' in list(dset_dict[model].keys())) == True and \
                ('orog' in list(dset_dict[model].keys())) == True:
                if ('lev' in list(dset_dict[model][var_id].coords)) == True and \
                    ('lev' in list(dset_dict[model]['pfull'].coords)) == True:
                        print(model, 'hybrid height coordinate')
                
                    

# %% [markdown]
# ## Calculate liquid water path from content

# %%
for model in dset_dict.keys():
    if ('plev' in list(dset_dict[model]['clw'].coords)) == True:
        _lwp = xr.DataArray(data=da.full(shape=dset_dict[model]['clw'].shape,fill_value=np.nan),
                            dims=dset_dict[model]['clw'].dims,
                            coords=dset_dict[model]['clw'].coords)

        dset_dict[model] = dset_dict[model].reindex(plev=dset_dict[model]['plev'][::-1])
        for lev2, lev, i in zip(dset_dict[model]['plev'].values, dset_dict[model]['plev'][1:].values, range(1,len(dset_dict[model]['plev']))):
            
            # calculate pressure difference between two levels
            dp = dset_dict[model]['plev'].sel(plev=slice(lev2,lev)).diff(dim='plev')
            # calculate liquid water path in each layer
            _lwp[:,i,:,:] = (dset_dict[model]['clw'].sel(plev=slice(lev2,lev)).diff(dim='plev') * 9.81 * dp)[:,0,:,:]
            # sum over all layers to get the liquid water path in the atmospheric column
            dset_dict[model]['lwp'] = _lwp.sum(dim='plev')
            # assign attributes to data array
            dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs(dset_dict[model]['clw'].attrs)
            dset_dict[model]['lwp'] = dset_dict[model]['lwp'].assign_attrs({'long_name':'Liquid Water Path', 
                                                                            'mipTable':'', 'out_name': 'lwp',
                                                                            'standard_name': 'atmosphere_mass_content_of_cloud_liquid_water',
                                                                            'title': 'Liquid Water Path',
                                                                            'variable_id': 'lwp', 'original_units': 'kg/kg',
                                                                            'history': "{}Z altered by F. Hellmuth: Converted units from 'kg kg-1' to 'g kg-1'. Interpolate data from hybrid-sigma levels to isobaric levels with geocat.comp.interpolation.interp_hybrid_to_pressure. Calculate lwp with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
        # when ice water path does not exist
        if ('clivi' in list(dset_dict[model].keys())) == False:
            # print(model)
            _iwp = xr.DataArray(data=da.full(shape=dset_dict[model]['cli'].shape, fill_value=np.nan,),
                                dims=dset_dict[model]['cli'].dims,
                                coords=dset_dict[model]['cli'].coords)
            dset_dict[model] = dset_dict[model].reindex(plev=dset_dict[model]['plev'][::-1])
            for lev2, lev, i in zip(dset_dict[model]['plev'].values, dset_dict[model]['plev'][1:].values, range(1,len(dset_dict[model]['plev']))):
                
                # calculate pressure difference between two levels
                dp = dset_dict[model]['plev'].sel(plev=slice(lev2,lev)).diff(dim='plev')
                # calculate Ice water path in each layer
                _iwp[:,i,:,:] = (dset_dict[model]['cli'].sel(plev=slice(lev2,lev)).diff(dim='plev') * 9.81 * dp)[:,0,:,:]
                # sum over all layers to get the Ice water path in the atmospheric column
                dset_dict[model]['clivi'] = _iwp.sum(dim='plev')
                # assign attributes to data array
                dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs(dset_dict[model]['cli'].attrs)
                dset_dict[model]['clivi'] = dset_dict[model]['clivi'].assign_attrs({'long_name':'Ice Water Path', 
                                                                                    'mipTable':'', 'out_name': 'clivi',
                                                                                    'standard_name': 'atmosphere_mass_content_of_cloud_ice',
                                                                                    'title': 'Ice Water Path',
                                                                                    'variable_id': 'clivi', 'original_units': 'kg/kg',
                                                                                    'history': "{}Z altered by F. Hellmuth: Converted units from 'kg kg-1' to 'g kg-1'. Interpolate data from hybrid-sigma levels to isobaric levels with geocat.comp.interpolation.interp_hybrid_to_pressure. Calculate clivi with hydrostatic equation.".format(now.strftime("%d/%m/%Y %H:%M:%S"))})
        
            
    
    if ('pfull' in list(dset_dict[model].keys())) == True:
        _lwp = xr.DataArray(data=da.full(shape=dset_dict[model]['clw'].shape,fill_value=np.nan),
                            dims=dset_dict[model]['clw'].dims,
                            coords=dset_dict[model]['clw'].coords)
        dset_dict[model] = dset_dict[model].reindex(lev=dset_dict[model]['lev'][::-1])
        

# %% [markdown]
# # Set values between -45S and 45N to nan

# %%
def set3D_lat_values_nan(array, upper_lat, lower_lat):
    l_lat = array['lat'].loc[array['lat'] == (array['lat'].sel(lat=slice(lower_lat,upper_lat))).min()][0].values
    u_lat = array['lat'].loc[array['lat'] == (array['lat'].sel(lat=slice(lower_lat,upper_lat))).max()][0].values
    array[:,array['lat'].where(array['lat']==l_lat).argmin('lat').values: array['lat'].where(array['lat']==u_lat).argmin('lat').values,:] = xr.DataArray(data=da.full(shape=(array[:,array['lat'].where(array['lat']==l_lat).argmin('lat').values: array['lat'].where(array['lat']==u_lat).argmin('lat').values,:]).shape,
                                                                                                                                                                      fill_value=np.nan),
                                                                                                                                                         dims=(array[:,array['lat'].where(array['lat']==l_lat).argmin('lat').values: array['lat'].where(array['lat']==u_lat).argmin('lat').values,:]).dims,
                                                                                                                                                         coords=(array[:,array['lat'].where(array['lat']==l_lat).argmin('lat').values: array['lat'].where(array['lat']==u_lat).argmin('lat').values,:]).coords)
    return array
    

# %%
for model in dset_dict.keys():
    
    dset_dict[model]['prsn'] = set3D_lat_values_nan(dset_dict[model]['prsn'], 45, -45)
    dset_dict[model]['lwp']  = set3D_lat_values_nan(dset_dict[model]['lwp'], 45, -45)
    dset_dict[model]['clivi']= set3D_lat_values_nan(dset_dict[model]['clivi'], 45, -45)
    dset_dict[model]['tas']  = set3D_lat_values_nan(dset_dict[model]['tas'], 45, -45)

# %% [markdown]
# ## Statistics
# For variables:
# - Snowfall [sf]
# - Total column cloud liquid, supercooled liqid, and rain water [tclslrw]
# - Total column cloud ice, snow water [tcisw]
# - 2m-Temperature [2t]
# 
# 1. Find where liquid water path is $\ge$ 5 g m-2 
# 2. Find where snowfall is $\ge$ 0.01mm h-1
# 3. Find where 2m-temperature $\le$ 0 $^o$ C 

# %%
def find_precip_cloud(dset):
    # 1. find where LWP >=5 gm-2
    sf = dset['prsn'].where(dset['lwp']>=5)
    lwp = dset['lwp'].where(dset['lwp']>=5)
    iwp = dset['clivi'].where(dset['lwp']>=5)
    t2 = dset['tas'].where(dset['lwp']>=5)
    
    # 2. find where snowfall >= 0.01mmh-1
    unit_sf = dset['prsn']
    sf = sf.where(unit_sf>=0.01)
    lwp = lwp.where(unit_sf>=0.01)
    iwp = iwp.where(unit_sf>=0.01)
    t2 = t2.where(unit_sf>=0.01)
        
    # 3. find where 2m-temperature <= 0C
    sf = sf.where(dset['tas']<=273.15)
    lwp = lwp.where(dset['tas']<=273.15)
    iwp = iwp.where(dset['tas']<=273.15)
    t2 = t2.where(dset['tas']<=273.15)
        
    sf_count = sf.groupby('time.season').count(dim='time',keep_attrs=True)
    lwp_count = lwp.groupby('time.season').count(dim='time',keep_attrs=True)
    iwp_count = iwp.groupby('time.season').count(dim='time', keep_attrs=True)
    t2_count = t2.groupby('time.season').count(dim='time', keep_attrs=True)
    
    return(sf_count, sf, iwp, lwp)

# %%
def plt_seasonal_NH_SH(variable,levels,cbar_label,plt_title):

    f, axsm = plt.subplots(nrows=2,ncols=4,figsize =[10,7], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0,globe=None)})

    for ax, season in zip(axsm.flatten()[:4], variable.season):
        ax.add_feature(cy.feature.COASTLINE, alpha=0.5)
        ax.set_extent([-180, 180, 90, 45], ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels   = False
        gl.right_labels = False
        variable.sel(season=season, lat=slice(45,90)).plot(ax=ax, transform=ccrs.PlateCarree(), 
                                                           extend='max', add_colorbar=False,
                            cmap=cm.hawaii_r, levels=levels)
        ax.set(title ='season = {}'.format(season.values))

    for ax, i, season in zip(axsm.flatten()[4:], np.arange(5,9), variable.season):
        ax.remove()
        ax = f.add_subplot(2,4,i, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None))
        ax.add_feature(cy.feature.COASTLINE, alpha=0.5)
        ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels   = False
        gl.right_labels = False
        cf = variable.sel(season=season, lat=slice(-90,-45)).plot(ax=ax, transform=ccrs.PlateCarree(), 
                                                                  extend='max', add_colorbar=False,
                            cmap=cm.hawaii_r, levels=levels)
        ax.set(title ='season = {}'.format(season.values))

    cbaxes = f.add_axes([1.0125, 0.025, 0.025, 0.9])
    cbar = plt.colorbar(cf, cax=cbaxes, shrink=0.5,extend='max', orientation='vertical', label=cbar_label)
    f.suptitle(plt_title, fontweight="bold");
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.)

# %%
for model in dset_dict.keys():
        

# %%
    dset_dict[model]['lcc_count'],dset_dict[model]['sf_lcc'],dset_dict[model]['iwp_lcc'],dset_dict[model]['lwp_lcc'] = find_precip_cloud(dset_dict[model])
    
    
    
# Cummulative snowfall days
    figname = '{}_cum_sf_days_season_mean_{}_{}.png'.format(model,starty, endy)
    plt_seasonal_NH_SH(dset_dict[model]['lcc_count'].where(dset_dict[model]['lcc_count']>0.), levels=np.arange(0,350,10), cbar_label='{} frequency (month)'.format(dset_dict[model]['lcc_count'].attrs['long_name'],), plt_title='{} {} ({} - {})'.format(model, dset_dict[model]['lcc_count'].attrs['long_name'], starty,endy))
    plt.savefig(FIG_DIR + figname, format = 'png', bbox_inches = 'tight', transparent = False)

    # Seasonal precipictation efficency ICE
    figname = '{}_ice_precip_eff_season_mean_{}_{}.png'.format(model,starty, endy)
    sf_iwp = ((dset_dict[model]['sf_lcc']/(dset_dict[model]['iwp_lcc'])).groupby('time.season').mean('time', keep_attrs=True, skipna=True))*1000
    plt_seasonal_NH_SH(sf_iwp,np.arange(0, 2.6,0.1),cbar_label='Mean seasonal ice precip. efficency',plt_title='{} Snowfall/IWP ({} - {})'.format(model,starty,endy))
    plt.savefig(FIG_DIR + figname, format = 'png', bbox_inches = 'tight', transparent = False)

    # seasonal precip efficeny ICE+LIQUID
    figname = '{}_ice_liquid_precip_eff_season_mean_{}_{}.png'.format(model,starty, endy)
    sf_iwp_lwp = ((dset_dict[model]['sf_lcc']/(dset_dict[model]['lwp_lcc']+dset_dict[model]['iwp_lcc'])).groupby('time.season').mean('time', keep_attrs=True, skipna=True))*1000
    plt_seasonal_NH_SH(sf_iwp_lwp, np.arange(0, 2.6, 0.1),cbar_label='Mean seasonal precip. efficency', plt_title='{} Snowfall/(IWP + LWP) ({} {})'.format(model,starty,endy))
    plt.savefig(FIG_DIR + figname, format = 'png', bbox_inches = 'tight', transparent = False)

    # seasonal liquid water path
    figname = '{}_lwp_season_mean_{}_{}.png'.format(model,starty, endy)
    lwp_season = dset_dict[model]['lwp_lcc'].groupby('time.season').mean(dim='time',keep_attrs=True, skipna=True)
    plt_seasonal_NH_SH(lwp_season, np.arange(0, 310, 10),cbar_label='{} ({})'.format(lwp_season.attrs['long_name'], lwp_season.attrs['units']),plt_title='ERA5 {} ({} - {})'.format(lwp_season.attrs['long_name'], starty,endy))
    plt.savefig(FIG_DIR + figname, format = 'png', bbox_inches = 'tight', transparent = False)


# %% [markdown]
# ## Calculating bin and bin sizes
# https://www.statisticshowto.com/choose-bin-sizes-statistics/
# %%
def plt_seasonal_2dhist_wp_sf(x_value, y_value, plt_title, xlabel, ylabel):
    f, axsm = plt.subplots(nrows=2,ncols=4,figsize =[10,5], sharex=True, sharey=True)
    cmap = cm.batlow
    # levels = np.arange(0.1,65000,5000)
    # norm = BoundaryNorm(levels, ncolors=cmap.N, )
    norm = LogNorm(vmin=1, vmax=50000)



    for ax, season in zip(axsm.flatten()[:4], x_value.season):
        Z, xedges, yedges = np.histogram2d((x_value.where(x_value['lat'] >=45).sel(season=season).values.flatten()), 
                                        (y_value.where(y_value['lat'] >=45).sel(season=season).values.flatten()), 
                                        bins=[40, 40], 
                                        range=[[0,4],[0, 4]])   

        im = ax.pcolormesh(xedges, yedges, Z.transpose(),cmap=cmap,norm=norm,)
        # cbar = f.colorbar(im, ax=ax,)
        ax.set(title =r'lat$\geq 45^\circ$N; season = {}'.format(season.values))
        ax.grid()
        
        _corr = xr.corr(x_value.where(x_value['lat'] >=45).sel(season=season), y_value.where(y_value['lat'] >=45).sel(season=season))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.55, 0.95, 'Corr: {}'.format(np.round(_corr,3).values), transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
        

    for ax, season in zip(axsm.flatten()[4:], x_value.season):
        Z, xedges, yedges = np.histogram2d((x_value.where(x_value['lat'] <=-45).sel(season=season).values.flatten()), 
                                        (y_value.where(y_value['lat'] <=-45).sel(season=season).values.flatten()), 
                                        bins=[40, 40], 
                                        range=[[0,4],[0, 4]])   

        im = ax.pcolormesh(xedges, yedges, Z.transpose(), cmap=cmap,norm=norm,)
        # cbar = f.colorbar(im, ax=ax, )
        ax.set(title =r'lat$\leq-45^\circ$S; season = {}'.format(season.values))
        
        ax.set_xlabel('{} ({})'.format(xlabel,x_value.attrs['units']))
        ax.grid()
        
        _corr = xr.corr(x_value.where(x_value['lat'] <=-45).sel(season=season), y_value.where(y_value['lat'] <=-45).sel(season=season))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.55, 0.95, 'Corr: {}'.format(np.round(_corr,3).values), transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    axsm.flatten()[0].set_ylabel('{} ({})'.format(ylabel, y_value.attrs['units']))
    axsm.flatten()[4].set_ylabel('{} ({})'.format(ylabel, y_value.attrs['units']))


    cbaxes = f.add_axes([1.0125, 0.025, 0.025, 0.9])
    cbar = plt.colorbar(im, cax=cbaxes, shrink=0.5, orientation='vertical', label='Frequency')
    f.suptitle(plt_title, fontweight="bold");
    
    
    
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)


# %%
for model in dset_dict[model]:
    lwp = dset_dict[model]['lwp_lcc']/1000
    lwp.attrs = {'units': 'kg m-2', 'long_name': 'Total column cloud liquid water'}
    iwp = dset_dict[model]['iwp_lcc']/1000
    iwp.attrs = {'units': 'kg m-2', 'long_name': 'Total column cloud ice and snow water'}

# %%
# dset_dict[model]['lwp'] = dset_dict[model]['lwp']/1000
# dset_dict[model]['lwp'].attrs = {'units': 'kg m-2', }#'long_name': 'Total column cloud liquid water'}
# dset_dict[model]['iwp'] = dset_dict[model]['iwp']/1000
# dset_dict[model]['iwp'].attrs = {'units': 'kg m-2',}# 'long_name': 'Total column cloud ice and snow water'}

# %%
    _lwp = lwp.groupby('time.season').mean(('time', ), keep_attrs=True, skipna=True)
    _iwp = iwp.groupby('time.season').mean(('time', ), keep_attrs=True, skipna=True)
    _sf = dset_dict[model]['sf_lcc'].groupby('time.season').mean(('time', ), keep_attrs=True, skipna=True)



    # %%
    # precip efficency from ice
    figname = '{}_2dhist_iwp_sf_season_mean_{}_{}.png'.format(model,starty, endy)
    plt_seasonal_2dhist_wp_sf(_iwp, _sf, '{} ({} - {}) Mean seasonal ice precip. efficency'.format(model,starty,endy), 'Ice Water Path', 'Snowfall')
    plt.savefig(FIG_DIR + figname, format = 'png', bbox_inches = 'tight', transparent = False)

    # precip efficency from liquid
    figname = '{}_2dhist_lwp_sf_season_mean_{}_{}.png'.format(model,starty, endy)
    plt_seasonal_2dhist_wp_sf(_lwp, _sf, '{} ({} - {}) Mean seasonal liquid precip. efficency'.format(model,starty,endy), 'Liquid Water Path', 'Snowfall')
    plt.savefig(FIG_DIR + figname, format = 'png', bbox_inches = 'tight', transparent = False)

    # precip efficency from mixed-phase clouds
    figname = '{}_2dhist_lwp_iwp_sf_season_mean_{}_{}.png'.format(model,starty, endy)
    plt_seasonal_2dhist_wp_sf(_iwp+_lwp, _sf, '{} ({} - {}) Mean seasonal ice+liquid precip. efficency'.format(model,starty,endy), 'Liquid + Ice Water Path', 'Snowfall')
    plt.savefig(FIG_DIR + figname, format = 'png', bbox_inches = 'tight', transparent = False)


