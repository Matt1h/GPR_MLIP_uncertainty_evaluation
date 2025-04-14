# Short description
Code to reproduce the experiments of the paper 
[Evaluation of uncertainty estimations for Gaussian process regression based machine 
learning interatomic potentials](https://arxiv.org/abs/2410.20398).
All experiments can be run from the command line via [hydra](https://hydra.cc/docs/intro/). The repository also already 
contains the results of the experiments, as well as notebooks and code to plot them.


# Installation
Clone the repository and run
```bash
pip install -e .
```
inside it.

# Prepare data for experiments
Before running experiments, the data has to be stored appropriately. To run the experiments with the datasets used in the paper, create a directory named `datasets` in the same directory as this repository. Inside the `datasets` directory, include the following:

- For the **rMD17** dataset:
  - Download and extract the `.zip` file from [rMD17 on Figshare](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038).
  - Place the extracted `rmd17` directory directly inside the `datasets` directory, i.e., `datasets/rmd17`.

- For the **WS22** dataset:
  - Download the `.npz` files for the individual molecules from [WS22 on Zenodo](https://zenodo.org/records/7032334).
  - Create a directory `WS22` inside the `datasets` directory and place all `.npz` files there, i.e., `datasets/WS22/*.npz`.

- For the **Porphyrin** dataset:
  - Download the `.npz` file from [WS22 on Zenodo](https://zenodo.org/records/15206611).
  - Create a directory `dftb` inside the `datasets` directory and place the `.npz` file there, i.e., `datasets/dftb/porphyrin.npz`.

If your datasets are stored in different locations, or if you want to use other datasets, you can specify the dataset paths by overriding the corresponding Hydra configurations.


# Run experiments
All experiments from the paper can be reproduced using Hydra commands. To run an experiment, navigate to the `src/GPR_MLIP` directory and execute the following command:

```bash
python cli.py
```

You can choose between three experiments  by overriding the `experiment` value in the command line. By default, the results are stored in a directory named `experiments` in the same directory as the repository. The results are organized by experiment, dataset, GPR method used, and a specified `result_name`. If no `dataset` is specified by overriding Hydra configurations, the experiment will run using the benzene molecule from the rMD17 dataset. If no method configurations or representations are specified, the default model used will be GPR with Coulomb representations. All hydra configurations are defined in `src/GPR_MLIP/config`.

## Hyperparameter optimization
```bash
python cli.py experiment=cross_validation result_name=default
```
This command runs cross-validation and maximum marginal likelihood hyperparameter optimization. Cross-validation is performed with different initial values for the maximum marginal likelihood optimization.

## Uncertainty error calculation
```bash
python cli.py experiment=uncertainty_error_calculation result_name=default
```
This command runs an experiment that calculates errors and uncertainties for a test set. The hyperparameters optimized during cross-validation are used. It will automatically load the hyperparameters from the directory corresponding to the dataset and GPR method settings, with `result_name=default`.

## Uncertainty sampling
```bash
python cli.py --multirun hydra/launcher=submitit_slurm  experiment=uncertainty_sampling experiment.uncertainty=absolute_error,bootstrap_aggregation,random,std_dev,two_sets hydra=gpu_pleiades result_name=default
```
This command runs the uncertainty sampling experiment. Again the hyperparameters corresponding to the dataset and GPR method settings with `result_name=default` will be used. Extensive uncertainty sampling runs should be executed on a GPU. To submit jobs via SLURM, specify the job script in a .`yaml` file under `config/hydra`, as done in `config/hydra/gpu_pleiades.yaml`. The presented command performs a multirun over five uncertainties, launching separate jobs (e.g., on five GPUs). Results are stored in subdirectories `0`, `1`, ..., following the order of uncertainties listed in the command. 

## Data preparation behavior
Before each experiment, the dataset is randomly shuffled and split into training, active learning, and test sets. The number of training samples is set via `experiment.n_train`, and the number of test samples via `prepare_data.function.n_test` (default: 2000). The remaining samples are used for active learning.

For hyperparameter optimization and uncertainty error calculation, the default number of training samples is 1000. If the seed `prepare_data.function.seed` is not overridden, both experiments will use the same split and therefore the same training data and models.

In the uncertainty error calculation experiment, predictions are made for the active learning set by default. In the uncertainty sampling experiment, the initial model is trained on 200 samples.

# Reproduce all results of the paper

The paper includes a calibration analysis and uncertainty sampling runs for benzene and aspirin from rMD17, SMA and O-HBDI from WS22, as well as data of porphyrin calculated with DFTB. Running the commands from the previous section will reproduce the results for benzene using GPR with the Coulomb representation.

To reproduce the results for GPR with Coulomb on the other datasets, run:

```bash
python cli.py experiment=... dataset=rmd17 dataset.molecule_name=aspirin result_name=default
```

```bash
python cli.py experiment=... dataset=ws22 dataset.molecule_name=sma result_name=default
```

```bash
python cli.py experiment=... dataset=ws22 dataset.molecule_name=o-hbdi result_name=default
```

```bash
python cli.py experiment=... dataset=dftb dataset.molecule_name=porphyrin result_name=default
```
To do the same for GPR with SOAP one has to run:
```bash
python cli.py experiment=... dataset=... dataset.molecule_name=... method/kernel=atomistic_sum1 representation=soap_atomistic result_name=default
```
For `uncertainty_sampling` one should add the respective additional settings as done in the prior section.