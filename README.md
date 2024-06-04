# Contrastive Metric Distillation

Code for the paper "Learning Temporal Distances: Contrastive Successor Features Can Provide a Metric Structure for Decision-Making."


## Installation

1. Build conda environment: `conda env create -f environment.yml`
2. Activate the environment: `conda activate metrics`
3. Install pip dependencies: `pip install -r requirements.txt --no-deps`
4. Export environment variables
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```


## Running experiments

Check `run.py` for available tasks and specific hyperparameters. You can turn on `--debug` and `--run_tf_eagerly` to run the code in debug mode.

