import os
from os.path import join
from typing import Callable
import copy
import itertools
from contextlib import ExitStack
import numpy as np
import torch

from GPR_MLIP.maths import mae


def run_cross_validation(
        inputs,
        targets,
        inds,
        offset,
        result_dir: str,
        method,
        possible_params: dict,
        contexts,
        seed: int,
):
    if offset is not None:
        targets = targets - offset
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)

        cv(
            inputs[inds['train']],
            targets[inds['train']],
            method,
            possible_params,
            result_dir=result_dir,
            seed=seed,
        )


def cv(
        inputs: torch.tensor,
        targets: torch.tensor,
        method,
        possible_params: dict,
        loss_function: Callable = mae,
        n_folds: int = 5,
        n_repetitions: int = 5,
        result_dir: str = 'results',
        seed=42,
):

    torch.set_default_tensor_type(torch.DoubleTensor)
    create_cv_folders_(result_dir)
    param_combinations = combination_dicts(possible_params)
    fold_inds = np.arange(n_folds)
    cv_loss = 0
    lowest_cv_loss = torch.inf
    for (p, params) in enumerate(param_combinations):
        print(f'\033[4mParameter combination {p}:\033[0m\n{params}')
        torch.save(params, join(result_dir, 'params', f'param_combination{p}'))
        for i in range(n_repetitions):
            # permute and separate into partitions
            rng = np.random.default_rng(seed)
            permutation_inds = rng.permutation(len(inputs))
            inputs_permutation = inputs[permutation_inds]
            targets_permutation = targets[permutation_inds]
            input_partitions = torch.chunk(inputs_permutation, n_folds)
            target_partitions = torch.chunk(targets_permutation, n_folds)

            for j in range(n_folds):
                test_inds = j == fold_inds
                train_inds = ~test_inds

                # concatenate train and test data
                train_inputs = torch.cat(list(itertools.compress(input_partitions, train_inds)))
                train_targets = torch.cat(list(itertools.compress(target_partitions, train_inds)))
                test_inputs = torch.cat(list(itertools.compress(input_partitions, test_inds)))
                test_targets = torch.cat(list(itertools.compress(target_partitions, test_inds)))

                # train and calculate loss
                model = method(**params)
                if torch.cuda.is_available():
                    model = model.cuda()
                    model.likelihood = model.likelihood.cuda()
                    for k in model.hyper_inits.keys():
                        model.hyper_inits[k] = model.hyper_inits[k].cuda()
                model.fit(train_inputs, train_targets)
                preds = model.predict(test_inputs)
                loss = loss_function(test_targets, preds)
                print(f'Loss model{j}: {loss}')
                cv_loss += loss_function(test_targets, preds)
            print('\n')

        cv_loss = cv_loss / (n_folds * n_repetitions)
        torch.save(cv_loss, join(result_dir, 'cv_loss', f'param_combination{p}'))
        if cv_loss < lowest_cv_loss:
            lowest_cv_loss = cv_loss
            best_params = copy.deepcopy(params)
        print(f'\nCV loss: {cv_loss}\n\n')
        cv_loss = 0
    torch.save(best_params, join(result_dir, 'best_params'))
    best_model = method(**best_params)
    if torch.cuda.is_available():
        best_model = best_model.cuda()
    best_model.fit(inputs, targets)
    torch.save(best_model.state_dict(), join(result_dir, 'best_model.pt'))
    return best_model, best_params


def combination_dicts(params):
    prefix = 'covar_module.base_kernel.atomistic_kernel'

    # Split the parameters into common and individual
    atomistic_lengthscales = {}
    atomistic_outputscales = {}
    individual_params = {}
    for key, value in params.items():
        if key.startswith(prefix) and 'lengthscale' in key:
            atomistic_lengthscales[key] = value
        elif key.startswith(prefix) and 'outputscale' in key:
            atomistic_outputscales[key] = value
        else:
            individual_params[key] = value

    # Generate combinations for individual parameters
    individual_combinations = list(itertools.product(*individual_params.values()))
    individual_keys = list(individual_params.keys())
    individual_dicts = [dict(zip(individual_keys, comb)) for comb in individual_combinations]

    # Check if there are common parameters
    if not atomistic_lengthscales and not atomistic_outputscales:
        # If no common parameters, return the individual combinations
        return individual_dicts

    # Generate combinations for common parameters, ensuring they all have the same value
    lengthscale_combinations = []
    if atomistic_lengthscales:
        for common_value in atomistic_lengthscales[list(atomistic_lengthscales.keys())[0]]:
            common_combination = {key: common_value for key in atomistic_lengthscales.keys()}
            lengthscale_combinations.append(common_combination)

    outputscale_combinations = []
    if atomistic_outputscales:
        for common_value in atomistic_outputscales[list(atomistic_outputscales.keys())[0]]:
            common_combination = {key: common_value for key in atomistic_outputscales.keys()}
            outputscale_combinations.append(common_combination)

    lengthscale_combinations = lengthscale_combinations or [{}]
    outputscale_combinations = outputscale_combinations or [{}]

    # Combine common and individual combinations
    final_combinations = []
    for lengthscale_comb in lengthscale_combinations:
        for outputscale_comb in outputscale_combinations:
            for ind_comb in individual_dicts:
                combined = {**lengthscale_comb, **outputscale_comb, **ind_comb}
                final_combinations.append(combined)

    return final_combinations


def create_cv_folders_(path):
    params_path = join(path, 'params')
    cv_loss_path = join(path, 'cv_loss')

    if os.path.exists(params_path):
        raise ValueError(f'{params_path} already exists')

    if os.path.exists(cv_loss_path):
        raise ValueError(f'{cv_loss_path} already exists')

    if not os.path.exists(path):
        os.mkdir(path)
    os.mkdir(params_path)
    os.mkdir(cv_loss_path)


def get_diff_combination_hyper_inits(
        likelihood_hyper_names,
        kernel_hyper_names,
        general_hypers
):
    print(general_hypers)
    hyper_names = likelihood_hyper_names + kernel_hyper_names

    separate_diff_hyper_inits = {}
    for general_name, v in general_hypers.items():
        for name in hyper_names:
            if general_name in name:
                separate_diff_hyper_inits[name] = v
    diff_hyper_inits = combination_dicts(separate_diff_hyper_inits)
    return diff_hyper_inits


def get_diff_fixed_set_hyper_inits(
        likelihood_hyper_names,
        kernel_hyper_names,
        general_hypers
):
    hyper_names = likelihood_hyper_names + kernel_hyper_names

    diff_hyper_inits = []
    for hypers in general_hypers:
        hyper_inits = {}
        for k in hypers:
            for name in hyper_names:
                if k in name:
                    hyper_inits[name] = hypers[k]
        diff_hyper_inits.append(hyper_inits)
    return diff_hyper_inits


def get_hyper_names_atomistic_sum_kernel2(n_atoms):
    lengthscale_hyper_names = [f'covar_module.base_kernel.atomistic_kernel{i}.base_kernel.lengthscale' for i in range(n_atoms)]
    outputscale_hyper_names = [f'covar_module.base_kernel.atomistic_kernel{i}.outputscale' for i in range(n_atoms)]
    return lengthscale_hyper_names + outputscale_hyper_names


def convert_list_entries_to_tensor(list_):
    return [torch.tensor(v) for v in list_]


def get_log_scale(lengthscale_log_start, lengthscale_log_end, n_lengthscales):
    return list(torch.logspace(lengthscale_log_start, lengthscale_log_end, steps=n_lengthscales))
