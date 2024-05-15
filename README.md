# PHDStream (Private Hierarchical Decomposition Stream)
A streaming algorithm for differentially private synthetic stream generation using hierarchical decomposition.
This repository contains the code for the paper: [An Algorithm for Streaming Differentially Private Data]([url](https://arxiv.org/abs/2401.14577))

### Environment setup

1. Install OSMNX package and create a new conda environment "phdstream". Note that installing with PIP can result in strange behavior.
```
conda create -n phdstream -c conda-forge --strict-channel-priority osmnx
```
2. Ensure "phdstream" is active
```
conda activate phdstream
```
3. Ensure you are in the project root directory
4. Install other dependencies
```
conda install -c conda-forge geopandas geodatasets shapely pandas numpy tqdm jupyterlab
```

### Run

The execution start at the ```main.py``` file. By default it runs a parallel processing code on at most 7 cores. Some things to note about this file:

1. Multiple experiments are run over all possible combinations of the hyperparameter values provided.
   1. For example, to run all experiments for both privacy budgets $1$ and $2$, set ```epsilons = [1.0, 2.0]```.
2. Regarding datasets
   1. By default the code runs with the Toy dataset "Circles with deletion".
   2. Gowalla and NY Taxi datasets are not provided with the code however they can be downloaded from their respective websites and processed using the files in ```src/onetime/gowalla_data_processing.py``` and ```src/onetime/ny_data_processing.py``` respectively.
   3. Dataset specific config can be updated in ```config.py```
