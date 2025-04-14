from typing import Callable
import numpy as np
import torch


UNITS_SCALE = {
    'eV': 1,
    'kcal/mol': 23.0621,
}


def load_from_npz(
    npz_path: str, 
    coordinates_key: str,
    energy_key: str,
    representation_fn: Callable,
    calc_representation: bool,
    unit='eV',
):
    data = np.load(npz_path)

    inputs = data[coordinates_key]
    inputs = torch.tensor(inputs).reshape(len(inputs), -1)
    if calc_representation:
        inputs = representation_fn(list(inputs))

    targets = torch.tensor(data[energy_key]/UNITS_SCALE[unit])
    if targets.shape[-1] == 1:
        targets = targets.squeeze()


    return inputs, targets


def load_charges_from_npz(
    npz_path: str, 
    charges_key: str,
):
    return torch.tensor(np.load(npz_path)[charges_key])


def prepare_data(
        targets,
        n_train: int,
        n_test: int,
        remove_offset: bool,
        seed: int,
    ):
    n_configs = len(targets)

    # shuffle
    rng = np.random.default_rng(seed)
    shuffle_inds = rng.permutation(n_configs)

    # separate
    inds = separate(shuffle_inds, n_configs, n_train, n_test)

    # remove offset
    offset = None
    if remove_offset:
        offset = targets[inds['train']].mean()

    return inds, offset

    


def get_ard_dim_from_npz(npz_path: str, coordinates_key, representation_fn):
    """Load a single geometry and calculate representation for ARD dim."""
    data = np.load(npz_path)
    inputs = data[coordinates_key]
    inputs = torch.tensor(inputs).reshape(len(inputs), -1)
    inputs = representation_fn(list(inputs))
    return inputs.shape[-1]


def separate(inputs, n, n_train, n_test):
    if n_test > 0:
        res = {
            'train': inputs[:n_train],
            'al': inputs[n_train:n - n_test],
            'test': inputs[-n_test:]
        }
        return res
    elif n_test == 0:
        res = {
            'train': inputs[:n_train],
            'al': inputs[n_train:n - n_test],
            'test': inputs[:0]
        }
        return res