---
layout: episode
title: "EOSC Nordic Climate tasks"
---

# EOSC Nordic WP5 tasks and status

- [EOSC Nordic Task 5.2.1](#eosc-nordic-task-521)
- [EOSC Nordic Task 5.2.2](#eosc-nordic-task-522)
- [EOSC Nordic Task 5.3.1](#eosc-nordic-task-531)
- [EOSC Nordic Task 5.3.2](#eosc-nordic-task-532)

We describe below WP5 tasks as defined in the [submitted application](https://wiki.neic.no/w/ext/img_auth.php/3/35/Proposal-SEP-210556427.pdf).
Then we give an overview of the status at the start of the project (state of the art), then the current status of each task. Status of each task is regularly updated (add new section with a new date).

# EOSC Nordic Task 5.2.1

> ## T5.2.1: Cross-border data processing workflows (M1-36) - Lead: UT/ETAIS Participants: UIO-USIT/SIGMA2,
> UICE, SNIC, UGOT/GGBC, FMI, SU/SNIC, UIO-GEO/SIGMA2
> In this subtask, we will facilitate data pre- and post-processing workflows (High Performance computation or High
> Throughput computation) on distributed data and computing resources by enabling community specific or thematic
> portals, such as [PlutoF](https://plutof.ut.ee/) and [Galaxy-based](https://usegalaxy.eu/) geoportal, traditionally
> designed to submit jobs on local clusters, to allow scheduling of jobs on remote resources. The API modules that
> will be developed to support the most commonly and widely used Distributed Resource Managers (DRMs) will be
> designed to become a generic solution, i.e. independent from the architecture and the technology of any given portal.
{: .callout}

> ## What is known/available at the start of the project
>
>
> Here we list what is relevant for the Climate science demonstrator only.
> 
> - [Galaxy pulsar](https://pulsar-network.readthedocs.io/en/latest/) for wide job execution system distributed across several European datacenters, allowing to scale Galaxy instances computing power over heterogeneous resources.
> - [Galaxy workflows](https://galaxyproject.org/learn/advanced-workflow/) are text files that can be easily exchanged. Galaxy workflows can be searched per Galaxy instances. For instance on Galaxy Europe, shared workflows are [published](https://climate.usegalaxy.eu/workflows/list_published).
> - [Galaxy shared histories](https://galaxyproject.org/learn/share/). Galaxy allows users to share their "histories" (data, processing, etc.) via a link. Users can set permissions to restrict access to a group of users if necessary (or a single user).
{: .solution}

> ## Status: January 2020
> 
> Preliminary list of tasks to enable T5.2.1:
> 
> - The plan is to install one single Galaxy instance for all the Nordics. Pulsar will be used to submit jobs on various platforms (HPCs and cloud computing). The objective is to use a similar setting as the one used by Galaxy Europe to ease maintenance and facilitate deployment of new tools by the Climate community,
> - The [list of Galaxy tools available/needed](../work/galaxy#galaxy-climate-tools) is provided and maintained by the Climate community. T5.2.1 will install Galaxy tools that are made available in the [Galaxy Toolshed](https://toolshed.g2.bx.psu.edu/) or available as interactive environment in the [Galaxy Europe github repository](https://github.com/NordicESMhub/galaxy/tree/release_19.09_europe/tools/interactive). 
> - The [list of available/needed training material](../work/work/galaxy#galaxy-training-material) is also provided and maintained by the Climate community (NICEST2). T5.2.1 will install Galaxy tools and datasets (Galaxy data libraries) necessary for users to use these training material on the Nordic Galaxy instance.
> 
{: .solution}

> ## Status: April 2020
> 
> ### Galaxy Training material
> 
> New training material under review for publication:
> - [Visualize Climate data with Panoply netCDF viewer](https://training.galaxyproject.org/training-material/topics/climate/tutorials/panoply/tutorial.html).
> 
> New training material under development:
> - Climate 101  is [in preparation on NordicESMHub](https://github.com/NordicESMhub/galaxy-training-material/tree/climate101); corresponding [PR](https://github.com/galaxyproject/training-material/pull/1871).
> 
> New training material planned:
> - Analyzing CMIP6 data with Galaxy Climate JupyterLab (in preparation; NICEST2 - not started yet)
> - ESMValTool with Galaxy Climate JupyterLab (in preparation; NICEST2 - not started yet)
> - Running CESM with Galaxy Climate JupyterLab (in preparation; it will be based on [GEO4962](https://nordicesmhub.github.io/GEO4962/), a course that is regularly given at the University of Oslo. See [GEO4962](https://www.uio.no/studier/emner/matnat/geofag/GEO4962/index.html)).
> 
{: .solution}

> ## Galaxy climate workbench framework for EOSC-Nordic
> More information on Galaxy climate workbench can be found [here](../work/galaxy).
> - [Galaxy tools for Climate Analysis](../work/galaxy#galaxy-climate-tools)
> - [Galaxy Training material for Climate Analysis](../work/galaxy#galaxy-training-material).
> This page is regularly updated to reflect status and progress.
>
{: .challenge}

# EOSC Nordic Task 5.2.2

> ## T5.2.2: Code Repositories, Containerization and “virtual laboratories” (M1-36) - Lead: SIGMA2 Participants:
> UICE, CSC, UIO-GEO/SIGMA2, UIO-INF/SIGMA2, UH
> In this subtask, we will pilot solutions for cross-borders “virtual laboratories” to allow researchers to work in a
> common software and data environment regardless which computing infrastructure the analysis is performed on,
> thus ensuring the highest reproducibility of the results. The work will encompass evaluations of different Docker
> Hub technologies provided by the EOSC-hub as well as mechanisms for build automation, package management
> and containerization. The subtask will focus on building a natural language processing laboratory, but the overall
> goal will be to create a generic recipe for building virtual laboratories.
{: .callout}


> ## What is known/available at the start of the project
> 
> - Tools developed in the framework of Galaxy are available in the [NordicESMHub github organization](https://github.com/NordicESMhub) as [galaxy-tools github repository](https://github.com/NordicESMhub/galaxy-tools).
> - conda package manager has been used by the Norwegian Climate community for packaging tools (for instance [cesm](https://bioconda.github.io/recipes/cesm/README.html)) in [bioconda](https://github.com/bioconda/bioconda-recipes). [conda-forge](https://conda-forge.org/) could be used too (but corresponding containers may not be created automatically).
> - Each package added to Bioconda also has a corresponding Docker [BioContainer](https://biocontainers.pro/) automatically created and uploaded to [Quay.io](https://quay.io/organization/biocontainers). A list of these and other containers can be found at the [Biocontainers Registry](https://biocontainers.pro/#/registry).
> For instance, CESM bioconda container can be found [here](https://biocontainers.pro/#/tools/cesm) wit both docker and singularity containers available.
> - Tools/models developed outside the Galaxy framework are stored in various places. We do not have a full overview yet.
{: .solution}

> ## Status: January 2020
> 
> Preliminary list of tasks to enable T5.2.2:
> - Discussion on possible solutions for submitting jobs from Galaxy to different platforms (in Sweden and Norway). Pulsar seems to be the best solution for Galaxy. This is already what is used by Galaxy Europe where [Galaxy Climate](https://climate.usegalaxy.eu/) is currently deployed.
> - Usage of conda package manager is recommended along with containers (as done with bioconda and biocontainers).
> - There is no equivalent container community repository for climate: should we set up something similar to [biocontainers](https://biocontainers.pro/#/)?
{: .solution}

> ## Status: April 2020
> 
> Target backend systems have been identified:
> 
> **Norway**:
> 
> - [Norwegian Research and Education Cloud Openstack (NREC)](https://docs.nrec.no/)
> - [NIRD Toolkit k8s](https://www.sigma2.no/nird-toolkit)
> - [Saga (low workloads)](https://documentation.sigma2.no/hpc_machines/saga.html)
> - [Betsy](https://documentation.sigma2.no/hpc_machines/betzy.html)
> 
> **Sweden**:
> - [SNIC Science Cloud Openstack](https://cloud.snic.se/)
> - [Tetralith SLURM cluster](https://www.nsc.liu.se/systems/tetralith/)
> 
> Discussion with SNIC has been initiated with Sweden for using HPC resources.
> 
{: .solution}

# EOSC Nordic Task 5.3.1

> ## T5.3.1: Integrated Data Management Workflows (M1-36) Lead: CSC – Participants: UIO-INF/SIGMA2, UH, SNIC,
> UICE, UIO-GEO/SIGMA2, FMI, SIGMA2
> This task will provide solutions for facilitating complex data workflows involving disciplines specific repositories,
> data sharing portals (such as Earth System Grid Federation, ESGF) and storage for active computing. An emerging
> HTTP API solution integrated with B2SAFE workflows will be adopted to streamline the creation of replicas of
> community specific data repositories towards the computing sites, where computations can be performed. This task
> will comprise also the adaptation of portals
{: .callout}

> ## What is known/available at the start of the project
>
> - [B2share Nordic](https://neic.no/affiliate-B2share/)
> - [B2safe](https://www.eudat.eu/b2safe)
> - [CernVM File System (CernVM-FS)](https://cernvm.cern.ch/portal/filesystem)
> - [ownCloud](https://owncloud.org/)
> - [Galaxy data libraries](https://galaxyproject.org/data-libraries/). Per instance (same as for Galaxy workflows) but possible to "replicate" between Galaxy instances through CVMFS.
> - [Earth Systm Grid Federation (ESGF)](https://esgf.llnl.gov/)
>
{: .solution}

> ## Status: January 2020
>
> CVFMS is used to replicate Galaxy reference data on any Galaxy instance. Look at Galaxy [Reference Data with CVMFS](https://training.galaxyproject.org/training-material/topics/admin/tutorials/cvmfs/tutorial.html) tutorial for more information on the usage of CVMFS in Galaxy for deploying/replicating reference data. This approach is probably suitable for small climate datasets (for instance teaching datasets, in-situ observations) but is not appropriate for the bulk amount of climate data. We suggest to investigate other remote access solutions.
>
> Preliminary list of tasks to enable T5.3.1:
> - Prepare test dataset in [zarr](https://zarr.readthedocs.io/en/stable/) for parallel access through python.
> - Create data catalog using [intake](https://intake.readthedocs.io/en/latest/catalog.html). The goal will be to automatically create/update data catalog as new data is harvested (link to T5.3.2).
> - Check [esm-intake](https://intake-esm.readthedocs.io/en/latest/) that is specific to [CMIP](https://www.wcrp-climate.org/wgcm-cmip) and [CESM Large Ensemble Community Project](http://www.cesm.ucar.edu/projects/community-projects/LENS/) 
> - Install [ownCloud](https://owncloud.org/) on [NIRD research data project area](http://www.cesm.ucar.edu/projects/community-projects/LENS/) to test access and processing of climate data from Galaxy (using intake catalog and zarr data format).
>
{: .solution}

> ## Status: April 2020
>
> - Preliminary tests using ownCloud have been successful. However, no performance analysis has been performed yet. So far we only tested the functionalities. Larger datasets will be harvested for further testing.
{: .solution}


> ## Climate data relevant for EOSC-Nordic
> The list of data relevant for the Climate community can be found [here](../work/data).
{: .challenge}

# EOSC Nordic Task 5.3.2

> ## T5.3.2: Machine actionable DMPs (M1-36) Lead: SIGMA2 – Participants: GFF, SNIC, UGOT/SND
> Link DMP with storage & computing reservation.
{: .callout}

> ## What is known/available at the start of the project
>
{: .solution}

> ## Status: January 2020
>
> Preliminary list of tasks to enable T5.3.2:
>
{: .solution}

> ## Status: April 2020
>
{: .solution}
