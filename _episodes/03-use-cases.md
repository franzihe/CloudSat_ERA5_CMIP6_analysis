---
layout: episode
title: "EOSC Nordic Climate Use cases"
---

> ## Use case 1: Reproducible ecosystem models
> **Goal**:  Better take into account typical ecosystems only found at high latitudes (and currently poorly represented) by integrating actual data and empirical knowledge in the model development and operation.
>
>
> **Target audience**: Experts of ecosystems, biologists and environmental scientists with no background in climate modeling
> 
> **Lead**: Hui Tang, University of Oslo, Norway
>
> **Partners**:
> - Lund University (Sweden)
> - University of Oslo (Norway)
> - Natural History Museum (Norway)
>
> > ## What?
> > 
> > Provide:
> > - a virtual laboratory for developing and testing parameterizations of ecosystems and related processes to scientists whose expertise is not numerical modeling, and thereby increase the pace of model improvement from the bottom up; 
> > - workflow management tools for operating these ecosystem models in a reproducible manner to realize their full potential without need for constant technical assistance;
> > - operational tools and infrastructure to facilitate the exploitation of ecosystem modeling for a wider audience and by non-specialists. 
> > 
> > In concrete terms, we will aim at working on:
> > 
> > - single-point/region simulations (FATES, LPJ-GUESS) at a number of specified locations in particular close to monitored sites;the number of sites and location should be flexible enough so that new sites can be easily added.
> > - allowing researchers to access and share data collected either during field campaigns or on a regular basis at selected measurement stations; 
> > - allowing researchers to control data access: researchers need to be able to choose who they want to share with and what they want to share; however, metadata should always be available to everyone.
> > - offering the necessary tools to analyze and visualize the resulting data and in particular comparison between observations and model outputs.
> {: .solution}
> > 
> > ## How?
> > 
> > - Integration of FATES (Functionally Assembled Terrestrial Ecosystem Simulator) as a new tool in Galaxy;
> > - Integration of LPJ-GUESS Ecosystem Model as a new tool in Galaxy;
> > - Integration in Galaxy of tools for analyzing and visualizing the model outputs; ideally, we would develop the visualization tool using [voila dashboards](https://voila.readthedocs.io/en/stable/) (or [R shiny](https://shiny.rstudio.com/)).
> > - Publication of in-situ measurements (with comprehensive meta-data, citation, access control, etc.).
> > 
> > **Reference**: [PeCAN project](http://pecanproject.github.io/)
> > 
> {: .solution}
>
>
> ### EOSC-Nordic tasks
> 
> Needs to be covered by the following EOSC-Nordic tasks:
> - T5.2.1: Cross-border data processing workflows 
> - T5.2.2: Code Repositories, Containerization and "virtual laboratories" (M1-36)
> - T5.3.1: Integrated Data Management Workflows (M1-36)
{: .callout}

> ## Use case 2: FAIR Climate data for the Nordics
> 
> **Goal**: Provide FAIR climate data to all the communities interested in climate mitigation and climate change impact assessment.
> 
> **Target audience**: Scientists, local authorities, policy makers and general public.
> 
> **Lead**: Hamish Struthers, National Supercomputing Centre, Sweden
> 
> **Partners**:
> - DMI, Denmark
> - CSC, Finland
> - NORCE, Norway
> - MetNo, Norway
> - UoI, Iceland
> 
> > ## What?
> > 
> > Facilitate publication and access to climate data (including but not restricted to CMIP) from Galaxy.
> {: .solution}
> > 
> > ## How? 
> > 
> > - Integration of CMIP data in Galaxy (stored on Nordic ESGF nodes);
> > - Integration of other climate data in Galaxy (could also include model outputs from research work not directly related to CMIP, and any other relevant observations);
> > - Deployment of simple tools for visualizing climate data over the Nordics from Galaxy;
> > - Deployment of statistical tools (global mean, regional mean, time series, climate indices) for characterizing climate change in the nordic countries;
> > - Deployment of Parallel Machine Learning and Deep Learning tools in Galaxy (for example to identify local trends, develop classifications, extrapolate mitigation impacts, have an insight into downscaling, etc.).
> > 
> {: .solution}
> ### EOSC-Nordic tasks
> 
> Needs to be covered by the following EOSC-Nordic tasks:
> - T5.2.1: Cross-border data processing workflows 
> - T5.2.2: Code Repositories, Containerization and "virtual laboratories" (M1-36)
> - T5.3.1: Integrated Data Management Workflows (M1-36)
> - T5.3.2: Machine actionable DMPs (M1-36)
{: .callout}

> ## Use case 3: A community Virtual Laboratory for developing Climate diagnostics for the Nordics
> 
> **Goal**: Share and develop Earth System Model EValuation diagnostics and analysis along with the related data.
> 
> **Target audience**: Climate experts and data scientists in the field.
> 
> **Lead**: Risto Makkonen, University of Helsinki, Finland
> 
> **Partners**: 
> - MetNo, Norway
> - NORCE, Norway
> - University of Oslo, Norway
> - FMI, Finland
> - DMI, Denmark
> - NERSC, Norway
> - University of Helsinki/INAR
> 
> > ## What? 
> > 
> > Provide a platform for facilitating the development of:
> > - new Earth System Model Evaluation Tool (ESMValTool) diagnostics for the Nordic regions (e.g., polar lows, European and Greenland blocking events) with strong emphasis on (high-latitude) observations;
> > - assemble reference datasets to validate models;
> > - ad-hoc analysis and visualization.
> > 
> > With these new tools, researchers will be able to compare outputs from different climate models in the Nordic countries (where numerical issues are often exacerbated and because the tuning was made difficult due to the relative lack of reference data) in order to identify deficiencies and develop strategies to improve the models e.g. [EC-EARTH](http://www.ec-earth.org/) and [NorESM](https://noresm-docs.readthedocs.io/en/latest/) (eventually combining the best parameterizations/modules).
> {: .solution}
> > 
> > ## How?
> > 
> > - Deployment of an e-infrastructure similar to [Pangeo](https://pangeo.io/) e.g. [binderhub](https://binderhub.readthedocs.io/en/latest/) and [jupyterhub](https://jupyterhub.readthedocs.io/en/stable/);
> > - Customized software stack e.g., pangeo software stack and additional packages such as [ESMValTool](https://www.esmvaltool.org/) for developing ad-hoc diagnostics for the Nordics;
> > - Access to HPC, including GPUs to be able to handle, analyze and visualize large amount of climate data;
> > - Access to CMIP data, re-analysis, model outputs produced by researchers (see use case 2);
> > - Access to observations (see use case 2).
> > 
> {: .solution}
>
> ### EOSC-Nordic tasks
> 
> Needs to be covered by the following EOSC-Nordic tasks:
> - T5.2.1: Cross-border data processing workflows 
> - T5.2.2: Code Repositories, Containerization and "virtual laboratories" (M1-36)
> - T5.3.1: Integrated Data Management Workflows (M1-36)
> - T5.3.2: Machine actionable DMPs (M1-36)
{: .callout}

