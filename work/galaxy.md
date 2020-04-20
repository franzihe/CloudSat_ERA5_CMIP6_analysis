---
layout: episode
title: "Galaxy for Climate"
---

# Source code repositories

All source codes are available on github. Forks (to allow the Nordic Climate community to make joint development) are made available in the [NordicESMHub github organization](https://github.com/NordicESMhub). 

Based on [Climate workbench](https://climate.usegalaxy.eu/):

To be used in Galaxy, one needs to:
- create an xml wrapper and corresponding scripts (python, bash, etc.) to be used from the Galaxy web portal or via bioblend (command line)
- create (if it does not exist yet) a conda package (and corresponding container) to facilitate deployment of the Galaxy tool

# Galaxy climate tools

- Specific tools for climate are developed and maintained [here](https://github.com/NordicESMhub/galaxy-tools). New tools that are "pushed" to the master branch are published automatically to the [Galaxy toolshed](https://toolshed.g2.bx.psu.edu/) in the [climate Analysis category](https://toolshed.g2.bx.psu.edu/).
Climate tools currently published:
	* Get Copernicus Essential Climate Variables for assessing climate variability
	* Create climate stripes from a tabular input file
	* Visualization of regular geographical data on a map with psyplot
	* Shift longitudes ranging from 0. and 360 degrees to -180. and 180. degrees
	* GDAL Geospatial Data Abstraction Library functions
	* Creates a png image showing statistic over areas as defined in the vector file
- Interactive tools for climate are maintained by [Galaxy Europe github](https://github.com/usegalaxy-eu/galaxy) in [interactive tools](https://github.com/usegalaxy-eu/galaxy/tree/release_19.09_europe/tools/interactive). Currently, we have deployed two interactive tools:
	* [Panoply viewer](https://github.com/usegalaxy-eu/galaxy/blob/release_19.09_europe/tools/interactive/interactivetool_panoply.xml)
	* [JupyterLab for Climate](https://github.com/usegalaxy-eu/galaxy/blob/release_19.09_europe/tools/interactive/interactivetool_climate_notebook.xml)

Interactive tools are currently not published in the toolshed (thus may not be easily findable and more difficult to deploy on other Galaxy instances).

# Galaxy Training material

[Galaxy Training Material github](https://github.com/galaxyproject/training-material)
The fork in the NordicESMHub github organization is available [here](https://github.com/NordicESMhub/galaxy-training-material).

The [Galaxy training material](https://training.galaxyproject.org/) is generated automatically. The procedure to develop and publish new training material is explained [here](https://training.galaxyproject.org/training-material/topics/contributing/). 

**Training material relevant for the Climate community**:

- [Introduction to Galaxy](https://training.galaxyproject.org/training-material/topics/introduction/slides/introduction.html#1)
- [Galaxy 101 for everyone](https://training.galaxyproject.org/training-material/topics/introduction/tutorials/galaxy-intro-101-everyone/tutorial.html)
- [JupyterLab in Galaxy](https://training.galaxyproject.org/training-material/topics/galaxy-ui/tutorials/jupyterlab/tutorial.html)

[A new topic called **Climate**](https://training.galaxyproject.org/training-material/topics/climate/) gathers all the training material specifically related to Climate Analysis:

- [Visualize Climate data with Panoply netCDF viewer](https://training.galaxyproject.org/training-material/topics/climate/tutorials/panoply/tutorial.html) is under review for final publication.
- Climate 101  is [in preparation on NordicESMHub](https://github.com/NordicESMhub/galaxy-training-material/tree/climate101); corresponding [PR](https://github.com/galaxyproject/training-material/pull/1871).

New training material planned:
- Analyzing CMIP6 data with Galaxy Climate JupyterLab (in preparation; NICEST2 - not started yet)
- ESMValTool with Galaxy Climate JupyterLab (in preparation; NICEST2 - not started yet)
- Running CESM with Galaxy Climate JupyterLab (in preparation; it will be based on [GEO4962](https://nordicesmhub.github.io/GEO4962/), a course that is regularly given at the University of Oslo. See [GEO4962](https://www.uio.no/studier/emner/matnat/geofag/GEO4962/index.html)).
