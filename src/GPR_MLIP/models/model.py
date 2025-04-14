from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.checkpoint import checkpoint
import gpytorch

from GPR_MLIP.models.uncertainties import variance, absolute_error


class GaussianProcessRegression(gpytorch.models.ExactGP):

    def __init__(
            self,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(10 ** -12)
            ),
            kernel=gpytorch.kernels.RBFKernel(),
            fit_mode='train_hypers',
            hyper_inits=None,
            lr=0.1,
            n_steps=200,
            offset=None,
    ):
        super(GaussianProcessRegression, self).__init__(None, None, likelihood)
        self.likelihood = likelihood
        self.kernel = kernel
        self.fit_mode = fit_mode
        self.hyper_inits = hyper_inits
        self.lr = lr
        self.n_steps = n_steps

        self.offset = offset
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

        self.losses, self.lengthscales, self.noises = [], [], []

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, train_inputs, train_targets, verbose=False):
        if verbose:
            print('-Fit.')
        if self.fit_mode == 'fixed_hypers':
            self.fit_with_fixed_hypers(train_inputs, train_targets)
        elif self.fit_mode == 'train_hypers':
            self.fit_and_train_hyper_inits(train_inputs, train_targets)
        elif self.fit_mode == 'train_hypers_with_checkpoint':
            self.fit_and_train_hyper_inits(train_inputs, train_targets, use_checkpoint=True)

    def add_and_fit(self, new_train_inputs, new_train_targets, verbose=False):
        if verbose:
            print('-Add and fit.')
        old_train_inputs = self.train_inputs[0]
        old_train_targets = self.train_targets
        train_inputs = torch.cat((new_train_inputs, old_train_inputs))
        train_targets = torch.cat((new_train_targets, old_train_targets))
        self.fit(train_inputs, train_targets, verbose=verbose)

    def predict(self, inputs, verbose=False):
        """Returns the mean of the prediction."""
        if verbose:
            print('-Predict.')
        inputs = inputs[None, :]
        self.eval()
        self.likelihood.eval()
        observed_pred = self.likelihood(self(inputs))
        return torch.flatten(observed_pred.mean)

    def add_and_fit_topk(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            k: int,
            verbose: bool = False,
            **kwargs
    ):
        if verbose:
            print('-Add and fit top k.')

        if 'uncertainty_function' in kwargs and kwargs['uncertainty_function'] == absolute_error:
            kwargs['targets'] = targets

        uncertainty = self.uncertainty(inputs, **kwargs)
        top_uncertainties, top_inds = torch.topk(uncertainty, k)
        if verbose:
            print(f'  └─ Top {k} uncertainties: {top_uncertainties}')
            bottom_uncertainties, bottom_inds = torch.topk(uncertainty, k, largest=False)
            print(f'  └─ Bottom {k} uncertainties: {bottom_uncertainties}')

        self.add_and_fit(inputs[top_inds], targets[top_inds], verbose=verbose)
        return top_inds

    def uncertainty(self, inputs, uncertainty_function=variance, **kwargs):
        return uncertainty_function(self, inputs, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        original_state_dict = super(GaussianProcessRegression, self).state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

        original_state_dict['hyper_inits'] = self.hyper_inits
        if hasattr(self, 'offset'):
            original_state_dict['offset'] = self.offset

        return original_state_dict

    def load_state_dict(self, state_dict, strict=True):

        self.hyper_inits = state_dict.pop('hyper_inits', None)
        if 'offset' in state_dict:
            self.offset = state_dict.pop('offset', None)

        super(GaussianProcessRegression, self).load_state_dict(state_dict, strict)

    def update_hyper_inits(self):
        for k in self.hyper_inits.keys():
            self.hyper_inits[k] = reduce(getattr, k.split('.'), self).detach()

    def init_hyper_inits(self):
        if self.hyper_inits is not None:
            self.initialize(**self.hyper_inits)

    def fit_and_train_hyper_inits(self, train_inputs, train_targets, use_checkpoint=False):
        self.init_hyper_inits()

        if isinstance(train_inputs, np.ndarray):
            train_inputs = torch.as_tensor(train_inputs)
        if isinstance(train_targets, np.ndarray):
            train_targets = torch.as_tensor(train_targets)

        # set training values
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)

        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None

        # train mode on
        self.train()
        self.likelihood.train()

        # inits
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        def custom_forward():
            output = self.forward(self.train_inputs[0])
            return output

        for i in range(self.n_steps):
            optimizer.zero_grad()
            if use_checkpoint:
                output = checkpoint(custom_forward)
            else:
                output = self.forward(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()

            self.losses += [loss.item()]
            # print(loss.item())
            # if self.covar_module.base_kernel.lengthscale.shape[1] == 1:
            #     self.lengthscales += [self.covar_module.base_kernel.lengthscale.item()]
            # self.noises += [self.likelihood.noise.item()]
            optimizer.step()

    def fit_with_fixed_hypers(self, train_inputs, train_targets):

        if isinstance(train_inputs, np.ndarray):
            train_inputs = torch.as_tensor(train_inputs)
        if isinstance(train_targets, np.ndarray):
            train_targets = torch.as_tensor(train_targets)

        # set training values
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)

        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None

        # train mode on
        self.train()
        self.likelihood.train()
