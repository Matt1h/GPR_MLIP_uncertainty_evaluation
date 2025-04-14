import os
from os.path import join
from typing import Type
import importlib


def get_hyper_inits(
        likelihood_hyper_names: list,
        kernel_hyper_names: list,
        general_hypers: dict,
):
    hyper_names = likelihood_hyper_names + kernel_hyper_names

    hyper_inits = {}
    for k in general_hypers:
        for name in hyper_names:
            if k in name:
                hyper_inits[name] = general_hypers[k]
    return hyper_inits


def get_model_path(experiment_name, result_dir, dataset_name, method_name, result_name, model_name):

    if model_name is None and experiment_name == 'cross_validation':
        model_name = 'best_model.pt'
    elif model_name is None and experiment_name != 'cross_validation':
        model_name = 'model.pt'

    if dataset_name is not None:
        result_dataset_name = dataset_name
    else:
        result_dataset_name = result_dir.split(os.sep)[-4]

    result_molecule_name = result_dir.split(os.sep)[-3]

    if method_name is not None:
        result_method_name = method_name
    else:
        result_method_name = result_dir.split(os.sep)[-2]

    if result_name is not None:
        result_name = result_name
    else:
        result_name = result_dir.split(os.sep)[-1]

    model_path = join(
        *result_dir.split(os.sep)[:4],
        experiment_name,
        result_dataset_name,
        result_molecule_name,
        result_method_name,
        result_name,
        model_name,
    )
    return model_path


def str2class(class_path: str) -> Type:
    """
    Obtain a class type from a string

    Args:
        class_path: module path to class, e.g. ``module.submodule.classname``

    Returns:
        class type
    """

    class_path = class_path.split(".")
    class_name = class_path[-1]
    module_name = ".".join(class_path[:-1])
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls