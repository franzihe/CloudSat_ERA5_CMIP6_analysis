---
layout: episode
title: "EOSC Nordic Climate tasks"
---

# EOSC Nordic WP5 tasks

We describe below WP5 tasks as defined in the [submitted application](https://wiki.neic.no/w/ext/img_auth.php/3/35/Proposal-SEP-210556427.pdf).

> ## T5.2.1: Cross-border data processing workflows (M1-36) - Lead: UT/ETAIS Participants: UIO-USIT/SIGMA2,
> UICE, SNIC, UGOT/GGBC, FMI, SU/SNIC, UIO-GEO/SIGMA2
> In this subtask, we will facilitate data pre- and post-processing workflows (High Performance computation or High
> Throughput computation) on distributed data and computing resources by enabling community specific or thematic
> portals, such as PlutoF [https://plutof.ut.ee/] and Galaxy-based [https://usegalaxy.eu/] geoportal, traditionally
> designed to submit jobs on local clusters, to allow scheduling of jobs on remote resources. The API modules that
> will be developed to support the most commonly and widely used Distributed Resource Managers (DRMs) will be
> designed to become a generic solution, i.e. independent from the architecture and the technology of any given portal.
{: .callout}

> ## T5.2.2: Code Repositories, Containerization and “virtual laboratories” (M1-36) - Lead: SIGMA2 Participants:
> UICE, CSC, UIO-GEO/SIGMA2, UIO-INF/SIGMA2, UH
> In this subtask, we will pilot solutions for cross-borders “virtual laboratories” to allow researchers to work in a
> common software and data environment regardless which computing infrastructure the analysis is performed on,
> thus ensuring the highest reproducibility of the results. The work will encompass evaluations of different Docker
> Hub technologies provided by the EOSC-hub as well as mechanisms for build automation, package management
> and containerization. The subtask will focus on building a natural language processing laboratory, but the overall
> goal will be to create a generic recipe for building virtual laboratories.
{: .callout}

> ## T5.3.1: Integrated Data Management Workflows (M1-36) Lead: CSC – Participants: UIO-INF/SIGMA2, UH, SNIC,
> UICE, UIO-GEO/SIGMA2, FMI, SIGMA2
> This task will provide solutions for facilitating complex data workflows involving disciplines specific repositories,
> data sharing portals (such as Earth System Grid Federation, ESGF) and storage for active computing. An emerging
> HTTP API solution integrated with B2SAFE workflows will be adopted to streamline the creation of replicas of
> community specific data repositories towards the computing sites, where computations can be performed. This task
> will comprise also the adaptation of portals
{: .callout}

> ## T5.3.2: Machine actionable DMPs (M1-36) Lead: SIGMA2 – Participants: GFF, SNIC, UGOT/SND
>
{: .callout}


# Tasks for enabling EOSC Nordic Climate Use case


Based on defined personas and pathways, we have defined the following list of tasks and milestones:


## Information on EOSC-Nordic website 

All potential users need to find relevant information on the [EOSC Nordic](https://www.eosc-nordic.eu/) to start their "journey" with EOSC Nordic.

   - A short description of all the use cases with information about what a user can expect to find by the end of the project. In addition, we should have clear information on the current progress so that users can clearly identify what is already available and what is still under development. 
   
> ## Remark
> The plan for each use case is **NOT** static and will evolve to take into account user and developer feedback. Users should see a progression but still be able to find out information in a similar way when coming back to the EOSC-Nordic website.
{: .callout}


## Existing climate services

- Link to existing services such as NIRD toolkit / European Climate Galaxy / PlutoF with success stories and examples illustrating all the different aspects of these available services.
- Give information on how Nordic & Baltic users can get access to these services
- How to create an account and login, how to apply for resources, how to transfer my data, what tools are available, etc.
- Collect feedback on the usage of these services and how to improve them

## Open call for EOSC-Nordic Champions for each Use case

- Award digital badges to users to motivate and reward participation (beginners to champion users)
- Common code repositories (we have github.com/NordicESMhub but we may want to have services provided by Nordics such as CodeRefienry) to share python/R/… codes where contributions are automatically tested with relevant dataset and with the possibility to get a DOI. 
- Make available online training material specific to the Climate community (CMIP6 data analysis, ESMValTool, observations & model comparisons, etc.)
- Organize a training workshop at a conference relevant for the discipline (or co-organized with another project such NICEST)  to attract new users (and potential champions)
- Deploy Galaxy Open infrastructure with access to relevant data storage (CMIP6, reanalysis, biodiversity, etc.):
      
	*  Interactive environments:
		+ Jupyterhub with climate software stack and the possibility to easily add new packages (containers already exist for this and are already available both on Climate Galaxy and NIRD toolkit) 
	* Galaxy tools:
		+ Organize hackathon for the deployment of relevant tools (machine learning, ESMs and/or regional models such as WRF, ESMValTool and other standard climate analysis tools) in Galaxy: create conda-forge recipes, docker containers and publish in Galaxy toolshed
		+ Add tools to NordicESMHub Climate Galaxy (https://github.com/NordicESMhub/galaxy-tools) or other relevant galaxy-tools repositories. All with continuous integration for testing and deploying the tools on Galaxy toolshed.
	* Galaxy Data Libraries:
		+ Make existing dataset accessible as Data Libraries to the Nordic Galaxy portal(s)
		+ Make Galaxy data FAIR (collaborate with other EOSC-Nordic workpackage and EOSC-Life?)
- Extend Galaxy Open infrastructure:
	* New interactive environments:
		+ Repo2docker (mybinder service with access to Nordic data)
	* DMP (that would “trigger” allocation of the necessary compute and storage resources)
	* Access to HPC, including GPUs for running heavy computing tools (ESM, machine learning, etc.)
	* Possibility to archive and retrieve data from the major data repository (national/international archive) with the relevant metadata (automatically generated) 

