import numpy as np
import torch


def move_gpr_model_to_cuda(model):
    model = model.cuda()
    model.likelihood = model.likelihood.cuda()
    if model.hyper_inits is not None:
        for k in model.hyper_inits.keys():
            if isinstance(model.hyper_inits[k], torch.Tensor):
                model.hyper_inits[k] = model.hyper_inits[k].cuda()

    if hasattr(model.kernel, 'representation_kwargs'):
        model.kernel.representation_kwargs['charges'] = (model.kernel.representation_kwargs['charges'].cuda())
    return model