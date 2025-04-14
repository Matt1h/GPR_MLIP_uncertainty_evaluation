#!/bin/bash

# Parameters
#SBATCH --account=imacm_gpu
#SBATCH --array=0-4%5
#SBATCH --error=/beegfs/holzenkamp/dev/active_learning/active/../../experiments/sma/candidate_based_al/ws22_full_ws22_energies/GPR_coulomb_rbf/combinations3_1000-likelihood-no_fast_pred/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=combinations3_1000-likelihood-no_fast_pred
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/beegfs/holzenkamp/dev/active_learning/active/../../experiments/sma/candidate_based_al/ws22_full_ws22_energies/GPR_coulomb_rbf/combinations3_1000-likelihood-no_fast_pred/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=1800
#SBATCH --wckey=submitit

# setup
module load 2021a CUDA/11.4.2

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /beegfs/holzenkamp/dev/active_learning/active/../../experiments/sma/candidate_based_al/ws22_full_ws22_energies/GPR_coulomb_rbf/combinations3_1000-likelihood-no_fast_pred/.submitit/%A_%a/%A_%a_%t_log.out --error /beegfs/holzenkamp/dev/active_learning/active/../../experiments/sma/candidate_based_al/ws22_full_ws22_energies/GPR_coulomb_rbf/combinations3_1000-likelihood-no_fast_pred/.submitit/%A_%a/%A_%a_%t_log.err /beegfs/holzenkamp/anaconda3/envs/schnetpack/bin/python -u -m submitit.core._submit /beegfs/holzenkamp/dev/active_learning/active/../../experiments/sma/candidate_based_al/ws22_full_ws22_energies/GPR_coulomb_rbf/combinations3_1000-likelihood-no_fast_pred/.submitit/%j
