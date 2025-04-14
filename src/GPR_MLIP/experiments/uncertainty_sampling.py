from os.path import join
import logging
from typing import Callable
import importlib
from contextlib import ExitStack
import copy
import gc
import inspect
import torch

from GPR_MLIP.models import dump_al_model
from .utils import move_gpr_model_to_cuda


log = logging.getLogger(__name__)


def run_uncertainty_sampling(
        inputs,
        targets,
        inds,
        offset,
        result_dir: str,
        model,
        n_init: int,
        n_al_iter: int,
        uncertainty_function: Callable,
        use_likelihood: bool,
        batch_size_uncertainty: int,
        contexts=None,
        batch_size_prediction=None,
):

    if offset is not None:
        targets = targets - offset
        model.offset = offset

    if torch.cuda.is_available():
        inputs = inputs.to(torch.device("cuda"))
        targets = targets.to(torch.device("cuda"))
        model = move_gpr_model_to_cuda(model)

    inputs_test = inputs[inds['test']]
    targets_test = targets[inds['test']]

    train_inds = list(inds['train'])
    al_inds = list(inds['al'])

    metrics = {'maes': [], 'max_errors': [], 'variances': []}
    n_pool = len(al_inds)
    idx = [i for i in range(n_pool)]
    idx_al = []

    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)

        for al_iter in range(n_al_iter):
            torch.cuda.empty_cache()
            print(f'\nAL interation {al_iter}/{n_al_iter}.\n')

            model_ = copy.deepcopy(model)
            model_.fit(inputs[train_inds], targets[train_inds])

            metrics = calc_metrics(
                model_,
                inputs_test,
                targets_test,
                metrics
            )

            if uncertainty_function is not None:
                kwargs = get_uncertainty_kwargs(
                    uncertainty_function,
                    n_init,
                    targets[al_inds],
                    use_likelihood,
                    batch_size_uncertainty
                )
                u = predict_uncertainty_in_batches(
                    batch_size_prediction,
                    uncertainty_function,
                    model_,
                    inputs[al_inds],
                    **kwargs
                )
                _, max_u_idx = torch.topk(u, 1)

            else:  # random
                max_u_idx = torch.tensor([0])

            save_metrics(result_dir, metrics)

            train_inds, al_inds = update_inds(train_inds, al_inds, max_u_idx)

            idx_al.append(idx[max_u_idx])
            del idx[max_u_idx]
            torch.save(idx_al, join(result_dir, 'idx_al'))

            dump_al_model(model_, result_dir, n_init)
            del model_
            gc.collect()
            torch.cuda.empty_cache()


def predict_uncertainty_in_batches(batch_size_prediction, uncertainty_function, model, inputs, **kwargs):
    if batch_size_prediction is None:
        u = uncertainty_function(model, inputs, **kwargs)
    else:
        inputs_chunks = torch.chunk(inputs, batch_size_prediction)

        targets_chunks = [None] * batch_size_prediction
        if 'targets' in kwargs:
            targets_chunks = torch.chunk(kwargs.pop('targets'), batch_size_prediction)
        u = []
        for inputs_, targets_ in zip(inputs_chunks, targets_chunks):
            model_ = copy.deepcopy(model)
            if 'targets' in inspect.getfullargspec(uncertainty_function).args:
                kwargs['targets'] = targets_

            u.append(uncertainty_function(model_, inputs_, **kwargs))
            del model_
            gc.collect()
            torch.cuda.empty_cache()
        u = torch.cat(u)
    return u


def calc_metrics(model, inputs_test, targets_test, metrics):
    preds = model.predict(inputs_test)
    errors = abs(preds - targets_test)

    metrics['maes'].append(torch.mean(errors))
    metrics['max_errors'].append(torch.max(errors))
    metrics['variances'].append(torch.var(errors))

    return metrics


def predict_in_batches(batch_size_prediction, model, inputs):
    if batch_size_prediction is None:
        preds = model.predict(inputs)
    else:
        inputs_al_chunks = torch.chunk(inputs, batch_size_prediction)
        preds = []
        for inputs in inputs_al_chunks:
            model_ = copy.deepcopy(model)
            preds.append(model_.predict(inputs))
            del model_
            gc.collect()
            torch.cuda.empty_cache()
        preds = torch.cat(preds)
    return preds


def save_metrics(result_dir, metrics):
    for k, v in metrics.items():
        print(f'{k[:-1]}: {v[-1]}')
        torch.save(v, join(result_dir, k))


def update_inds(train_inds, al_inds, max_u_idx):
    train_inds.append(al_inds[max_u_idx])
    del al_inds[max_u_idx]
    return train_inds, al_inds


def get_uncertainty_kwargs(uncertainty_function, n_init, targets, use_likelihood, batch_size_uncertainty):
    uncertainty_kwargs = {}

    if 'use_likelihood' in inspect.getfullargspec(uncertainty_function).args:
        uncertainty_kwargs['use_likelihood'] = use_likelihood

    if 'n_init' in inspect.getfullargspec(uncertainty_function).args:
        uncertainty_kwargs['n_init'] = n_init

    if 'targets' in inspect.getfullargspec(uncertainty_function).args:
        uncertainty_kwargs['targets'] = targets

    if 'batch_size' in inspect.getfullargspec(uncertainty_function).args:
        uncertainty_kwargs['batch_size'] = batch_size_uncertainty

    return uncertainty_kwargs


def name2uncertainty_function(method_name, uncertainty_function_name):
    if uncertainty_function_name == 'random':
        return None

    if method_name == 'GPR':
        cls = getattr(importlib.import_module('GPR_MLIP.models.uncertainties'), uncertainty_function_name)
    else:
        raise ValueError(f'No uncertainty function with name {uncertainty_function_name} '
                         f'exists for method with name {method_name}')

    return cls
