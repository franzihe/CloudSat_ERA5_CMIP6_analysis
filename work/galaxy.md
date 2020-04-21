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

# Tools available (or to be available) in Galaxy

## Galaxy climate tools

- Specific tools for climate are developed and maintained [here](https://github.com/NordicESMhub/galaxy-tools). New tools that are "pushed" to the master branch are published automatically to the [Galaxy toolshed](https://toolshed.g2.bx.psu.edu/) in the [climate Analysis category](https://toolshed.g2.bx.psu.edu/).
Climate tools currently published:

Tool | Description | Reference
--- | --- | ---
cds_essential_variability | Get Copernicus Essential Climate Variables for assessing climate variability | [Copernicus CDS](https://cds.climate.copernicus.eu/cdsapp#!/dataset/ecv-for-climate-change?tab=overview)
climate_stripes | Create climate stripes from a tabular input file | 
psy_maps | Visualization of regular geographical data on a map with psyplot |
shift_longitudes | Shift longitudes ranging from 0. and 360 degrees to -180. and 180. degrees |
gdal | GDAL Geospatial Data Abstraction Library functions |
mean_per_zone | Creates a png image showing statistic over areas as defined in the vector file |

- Interactive tools for climate are maintained by [Galaxy Europe github](https://github.com/usegalaxy-eu/galaxy) in [interactive tools](https://github.com/usegalaxy-eu/galaxy/tree/release_19.09_europe/tools/interactive). Currently, we have deployed two interactive tools:
	* [Panoply viewer](https://github.com/usegalaxy-eu/galaxy/blob/release_19.09_europe/tools/interactive/interactivetool_panoply.xml)
	* [JupyterLab for Climate](https://github.com/usegalaxy-eu/galaxy/blob/release_19.09_europe/tools/interactive/interactivetool_climate_notebook.xml)

Interactive tools are currently not published in the toolshed (thus may not be easily findable and more difficult to deploy on other Galaxy instances).

## Galaxy Machine Learning Tools

Some Machine Learning tools are already available as Galaxy Tools on the default [Galaxy Europe](https://usegalaxy.eu/) instance. 

Users can access it through the [Machine Learning Workbench](https://ml.usegalaxy.eu/):

In this section we list the most important tools that have been integrated into the Machine Learning workbench.
There are many more tools available so please have a more detailed look at the tool panel.
For better readability, we have divided them into categories.

#### Classification

Identifying which category an object belongs to.

Tool | Description | Reference
--- | --- | ---
"SVM Classifier"  | Support vector machines (SVMs) for classification| [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"NN Classifier"  | Nearest Neighbors Classification | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Ensemble classification" | Ensemble methods for classification and regression | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Discriminant Classifier"  | Linear and Quadratic Discriminant Analysis| [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Generalized linear" | Generalized linear models for classification and regression | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"CLF Metrics" | Calculate metrics for classification performance  | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}

#### Regression

Predicting a continuous-valued attribute associated with an object.

Tool | Description | Reference
--- | --- | ---
"Ensemble regression" | Ensemble methods for classification and regression | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Generalized linear" | Generalized linear models for classification and regression | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Regression metrics" | Calculate metrics for regression performance | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}

#### Clustering

Automatic grouping of similar objects into sets.

Tool | Description | Reference
--- | --- | ---
"Numeric clustering"  | Different numerical clustering algorithms | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}

#### Model building

Building general machine learning models.

Tool | Description | Reference
--- | --- | ---
"Estimator Attributes" | Estimator attributes to get all attributes from an estimator or scikit object | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Stacking Ensemble Models" | Stacking Ensembles to build stacking, voting ensemble models with numerous base options | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Search CV" | Hyperparameter Search performs hyperparameter optimization using various SearchCVs  | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Build Pipeline" | Pipeline Builder as an all-in-one platform to build pipeline, single estimator, preprocessor and custom wrappers | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}

#### Model evaluation

Evaluation, validating and choosing parameters and models.

Tool | Description | Reference
--- | --- | ---
"Model validation" | Model Validation includes cross_validate, cross_val_predict, learning_curve, and more | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Pairwise Metrics" | Evaluate pairwise distances or compute affinity or kernel for sets of samples | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Train/Test evaluation" | Train, Test and Evaluation to fit a model using part of dataset and evaluate using the rest | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Model Prediction" | Model Prediction predicts on new data using a preffited model | [Chollet et al. 2011](https://keras.io){:target="_blank"}
"Fitted model evaluation" | Evaluate a Fitted Model using a new batch of labeled data | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Model fitting" | Fit a Pipeline, Ensemble or other models using a labeled dataset | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}


#### Preprocessing and feature selection

Feature selection and preprocessing.

Tool | Description | Reference
--- | --- | ---
"Data preprocessing" | Preprocess raw feature vectors into standardized datasets  | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Feature selection" | Feature Selection module, including univariate filter selection methods and recursive feature elimination algorithm | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}

#### Deep learning

Build and use deep neural networks.

Tool | Description | Reference
--- | --- | ---
"Batch Models" | Build Deep learning Batch Training Models with online data generator for Genomic/Protein sequences and images | [Chollet et al. 2011](https://keras.io){:target="_blank"}
"Model Builder" | Create deep learning model with an optimizer, loss function and fit parameters | [Chollet et al. 2011](https://keras.io){:target="_blank"}
"Model Config" | Create a deep learning model architecture using Keras | [Chollet et al. 2011](https://keras.io){:target="_blank"}
"Train and evaluation" | Deep learning training and evaluation either implicitly or explicitly  | [Chollet et al. 2011](https://keras.io){:target="_blank"}

#### Visualization

Plotting and visualization.

Tool | Description | Reference
--- | --- | ---
"Regression performance plots" | Plot actual vs predicted curves and residual plots of tabular data |
ML performance plots" | Plot confusion matrix, precision, recall and ROC and AUC curves of tabular data |
"Visualization" | Machine Learning Visualization Extension includes several types of plotting for machine learning | [Chollet et al. 2011](https://keras.io){:target="_blank"}

#### Utilities

General data and table manipulation tools.

Tool | Description | Reference
--- | --- | ---
"Table compute" | The power of the pandas data library for manipulating and computing expressions upon tabular data and matrices. |
"Datamash operations" | Datamash operations on tabular data |
"Datamash transpose" | Transpose rows/columns in a tabular file |
"Sample Generator" | Generate random samples with controlled size and complexity | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}
"Train/Test splitting" | Split Dataset into training and test subsets | [Pedregosa et al. 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html){:target="_blank"}


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


**Training material on Machine Learning**:

- [Basics of machine learning](https://training.galaxyproject.org/training-material/topics/statistics/tutorials/machinelearning/tutorial.html)
- [Introduction to deep learning](https://training.galaxyproject.org/training-material/topics/statistics/tutorials/intro_deep_learning/tutorial.html)
- [Machine learning: classification and regression](https://training.galaxyproject.org/training-material/topics/statistics/tutorials/classification_regression/tutorial.html)
- [Regression in machine learning](https://training.galaxyproject.org/training-material/topics/statistics/tutorials/regression_machinelearning/tutorial.html)

