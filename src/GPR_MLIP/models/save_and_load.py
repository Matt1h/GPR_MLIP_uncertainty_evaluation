from os.path import join
import numpy as np
import torch
import gpytorch


from .model import GaussianProcessRegression


def init_gpr_model(state_dict_path, likelihood=None, kernel=None, **kwargs):
    state_dict = torch.load(state_dict_path, weights_only=False, **kwargs)

    if likelihood is None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(10 ** -12)
        )

    if kernel is None:
        kernel = gpytorch.kernels.RBFKernel()

    model = GaussianProcessRegression(likelihood=likelihood, kernel=kernel)
    model.load_state_dict(state_dict)
    return model


def dump_model(model, parent_dir):
    # save model state dict
    torch.save(model.state_dict(), join(parent_dir, 'model.pt'))

    # save data
    data = {
        'inputs': np.array(model.train_inputs[0].cpu()),
        'labels': np.array(model.train_targets.cpu()),
    }

    np.savez(join(parent_dir, 'data.npz'), **data)


def load_model(parent_dir, n_train=None, **kwargs):
    # load data
    data = np.load(join(parent_dir, 'data.npz'))

    # load model state dict
    model = init_gpr_model(join(parent_dir, 'model.pt'), **kwargs)

    # add data to model
    if n_train is not None:
        model.train_inputs = (torch.tensor(data['inputs'][:n_train]),)
        model.train_targets = torch.tensor(data['labels'][:n_train])
    else:
        model.train_inputs = (torch.tensor(data['inputs']),)
        model.train_targets = torch.tensor(data['labels'])

    return model


def dump_al_model(model, parent_dir, n_init):
    # save model state dict
    torch.save(model.state_dict(), join(parent_dir, 'model.pt'))

    # save data
    data = {
        'n_init': n_init,
        'inputs_init': np.array(model.train_inputs[0][:n_init].cpu()),
        'labels_init': np.array(model.train_targets[:n_init].cpu()),
        'inputs_al': np.array(model.train_inputs[0][n_init:].cpu()),
        'labels_al': np.array(model.train_targets[n_init:].cpu())
    }

    np.savez(join(parent_dir, 'data.npz'), **data)


def load_al_model(parent_dir, al_iter=None, **kwargs):
    # load data
    data = np.load(join(parent_dir, 'data.npz'))

    if al_iter is not None and al_iter > len(data['labels_al']):
        raise ValueError(
            f'al_iter = {al_iter} is higher than number of AL samples in data which is {len(data["labels_al"])}'
        )

    # load model state dict
    model = init_gpr_model(join(parent_dir, 'model.pt'), **kwargs)

    # add data to model
    if al_iter is not None:
        model.train_inputs = (
            torch.cat((torch.tensor(data['inputs_init']), torch.tensor(data['inputs_al'][:al_iter]))),
        )
        model.train_targets = torch.cat((torch.tensor(data['labels_init']), torch.tensor(data['labels_al'][:al_iter])))
    else:
        model.train_inputs = (torch.cat((torch.tensor(data['inputs_init']), torch.tensor(data['inputs_al']))),)
        model.train_targets = torch.cat((torch.tensor(data['labels_init']), torch.tensor(data['labels_al'])))

    return model


def load_state_dict(method, fit_mode, state_dict_path):
    model = method(fit_mode=fit_mode)
    model.load_state_dict(torch.load(state_dict_path))
    return model