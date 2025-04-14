from os.path import join
from contextlib import ExitStack
import torch

from GPR_MLIP.models import variance, two_sets_ensemble, bootstrap_aggregation_ensemble


def run_uncertainty_error_calculation(
        inputs,
        targets,
        inds,
        offset,
        result_dir: str,
        model,
        use_likelihood: bool,
        batch_size_uncertainty: int,
        contexts,
        part='al'
):

    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)

        if offset is not None:
            targets = targets - offset

        model.fit(inputs[inds['train']], targets[inds['train']])

        inputs_part = inputs[inds[part]]
        refs_part = targets[inds[part]]

        uncertainties = {}
        errors = {}

        preds = model.predict(inputs_part)

        errors['GPR model'] = (preds - refs_part).detach()
        uncertainties['GPR model'] = variance(
            model, 
            inputs_part, 
            use_likelihood=use_likelihood, 
            batch_size=batch_size_uncertainty
        ).detach()

        preds_two_sets, uncertainties['Two sets'] = two_sets_ensemble(model, inputs_part, n_init=len(inds['train']))
        errors['Two sets'] = (preds_two_sets - refs_part).detach()

        preds_bootstrap, uncertainties['Bootstrap aggregation'] = bootstrap_aggregation_ensemble(model, inputs_part)
        errors['Bootstrap aggregation'] = (preds_bootstrap - refs_part).detach()

        torch.save(uncertainties, join(result_dir, 'uncertainties'))
        torch.save(errors, join(result_dir, 'errors'))
        torch.save(refs_part, join(result_dir, 'refs'))
