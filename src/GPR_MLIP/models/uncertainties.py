import copy
import torch
import gc


def std_dev(model, inputs, batch_size=1000, use_likelihood=False):
    var = variance(model, inputs, batch_size=batch_size, use_likelihood=use_likelihood)
    return torch.sqrt(var)


def variance_(model, inputs, use_likelihood=False):
    inputs = inputs[None, :]
    model.eval()
    model.likelihood.eval()
    if use_likelihood:
        observed_pred = model.likelihood(model(inputs))
    else:
        observed_pred = model(inputs)

    return torch.flatten(observed_pred.variance)


def variance(model, inputs, batch_size=1000, use_likelihood=False):

    n_samples = len(inputs)

    if batch_size is None:
        variances = variance_(model, inputs, use_likelihood=use_likelihood)
        return torch.cat(variances)

    variances = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)

        batch_variances = variance_(model, inputs[start_idx:end_idx], use_likelihood=use_likelihood)
        variances.append(batch_variances)

    return torch.cat(variances)


def absolute_error(model, inputs, targets=None):
    if targets is None:
        raise ValueError('For absolute error as uncertainty the properties have to be provided.')
    return abs(model.predict(inputs) - targets)


def two_sets(model, inputs, n_init=None):
    pred1, pred2 = two_sets_predictions(model, inputs, n_init=n_init)
    return abs(pred1 - pred2)


def two_sets_ensemble(*args, **kwargs):
    pred1, pred2 = two_sets_predictions(*args, **kwargs)
    m = torch.mean(torch.stack([pred1, pred2]), dim=0)
    u = abs(pred1 - pred2)
    return m, u


def two_sets_predictions(model, inputs, n_init=None):
    if n_init is None:
        raise ValueError('For two sets as uncertainty n_init has to be provided.')

    n_al = len(model.train_inputs[0]) - n_init

    inds_initial_sets = [
        torch.arange(int(n_init/2)),
        torch.arange(int(n_init/2), n_init)
    ]

    inds_all = torch.arange(len(model.train_inputs[0]))
    inds_initial_sets_combined = torch.cat([inds_initial_sets[i] for i in range(len(inds_initial_sets))]).unique()
    inds_left = torch.tensor(list(set(inds_all.tolist()) - set(inds_initial_sets_combined.tolist())))
    if list(inds_left):
        train_inputs1 = torch.cat((model.train_inputs[0][inds_initial_sets[0]], model.train_inputs[0][inds_left]))
        train_inputs2 = torch.cat((model.train_inputs[0][inds_initial_sets[1]], model.train_inputs[0][inds_left]))
        train_targets1 = torch.cat((model.train_targets[inds_initial_sets[0]], model.train_targets[inds_left]))
        train_targets2 = torch.cat((model.train_targets[inds_initial_sets[1]], model.train_targets[inds_left]))
    else:
        train_inputs1 = model.train_inputs[0][inds_initial_sets[0]]
        train_inputs2 = model.train_inputs[0][inds_initial_sets[1]]
        train_targets1 = model.train_targets[inds_initial_sets[0]]
        train_targets2 = model.train_targets[inds_initial_sets[1]]

    model1 = copy.copy(model)
    model2 = copy.copy(model)

    model1.fit(train_inputs1, train_targets1)
    model2.fit(train_inputs2, train_targets2)

    pred1 = model1.predict(inputs)
    pred2 = model2.predict(inputs)
    return pred1, pred2


def bootstrap_aggregation(*args, **kwargs):
    predictions = bootstrap_aggregation_predictions(*args, **kwargs)
    pred_std_dev = torch.std(predictions, dim=0)
    return pred_std_dev


def bootstrap_aggregation_ensemble(*args, **kwargs):
    predictions = bootstrap_aggregation_predictions(*args, **kwargs)
    pred_mean = torch.mean(predictions, dim=0)
    pred_std_dev = torch.std(predictions, dim=0)

    return pred_mean, pred_std_dev


def bootstrap_aggregation_predictions(model, inputs, n_models=5):
    n_total = len(model.train_inputs[0])
    predictions = []

    for _ in range(n_models):
        indices = torch.randint(0, n_total, (n_total,))
        unique_indices = torch.tensor(list(set(indices.numpy())))

        train_inputs = model.train_inputs[0][unique_indices]
        train_targets = model.train_targets[unique_indices]

        model_copy = copy.deepcopy(model)
        model_copy.fit(train_inputs, train_targets)

        pred = model_copy.predict(inputs)
        predictions.append(pred)
        del model_copy
        gc.collect()

    predictions = torch.stack(predictions)
    return predictions
