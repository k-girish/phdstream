# PHDStream (Private Hierarchical Decomposition Stream)
A streaming algorithm for differentially private synthetic stream generation using hierarchical decomposition.
This repository contains the code for the paper: [An Algorithm for Streaming Differentially Private Data]([url](https://arxiv.org/abs/2401.14577))

### Environment setup

1. Install OSMNX package and create a new conda environment "phdstream". Note that installing with PIP can result in strange behavior.
```
conda create -n phdstream -c conda-forge --strict-channel-priority osmnx @github/clipboard-copy-element
```
2. Ensure "phdstream" is active
```
conda activate phdstream @github/clipboard-copy-element
```
3. Ensure you are in the project root directory
4. Install other dependencies
```
conda install -c conda-forge geopandas geodatasets shapely pandas numpy tqdm jupyterlab @github/clipboard-copy-element
```

