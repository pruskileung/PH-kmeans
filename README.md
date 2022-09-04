# $k$-means Clustering in Persistent Homology 

This git repository contains code for my MSc thesis on adapting and comparing methods of $k$-means clustering for use on persistent homology. 

# Usage

The repository is structed as follows: 
* `PD_subsample` contains code from [https://github.com/YueqiCao/PD-subsample], namely the `ApproxPH.py` that we use and adapt code from to construct persistence measures and calculate distances between. 
* `src` contains all code written for this project: 
  *  `data_utils` contains scripts generating the simulated data and processing the real-world data used in the project. This directory also contains methods used to clean and process persistent homology representations. 
  * `pd_pm_kmeans.py` contains the persistence diagram and persistence measure $k$-means algorithms. 
  * `synthetic_data_clustering.ipynb` demonstrates how the code was used to generate results on simulated data for the project. 
