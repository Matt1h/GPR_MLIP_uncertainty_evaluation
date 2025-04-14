import copy
import math
import torch
import matplotlib.lines as mlines

from .utils import set_ticks


def extended_reliablility_diagram(
        ax,
        uncertainty,
        error,
        num_bins=None,
        bin_size=None,
        min_samples=50,
        num_top_values=None,
        max_uncertainty=None,
        absolute=False,
        max_value=None,
        max_value_theory=None,
        ticks=None,
):
    if max_value is None:
        max_value = max([max(uncertainty), max(error)])

    if max_value_theory is None:
        max_value_theory = max_value

    ax.scatter(uncertainty, error, s=0.01, c='gray')

    # average in bins
    positions, averages, stds = calc_bin_dist(
        uncertainty,
        error,
        min_samples=min_samples,
        num_bins=num_bins,
        bin_size=bin_size
    )
    ax.scatter(positions, averages, s=15, c='red', marker='x', label='Bin mean')
    ax.scatter(positions, stds, s=15, c='green', marker='o', label='Bin standard\ndeviation')

    if absolute:
        ax.plot(positions, (positions * math.sqrt(2)) / math.pi, 'k--', c='red', label='Theoretical mean')
        ax.plot(
            positions,
            torch.sqrt(positions**2 * (1 - (2/math.pi))),
            'k--',
            c='green',
            label='Theoretical standard deviation'
        )
    else:
        ax.plot([0, max_value_theory], [0, 0], '--', c='red')
        ax.plot([0, max_value_theory] , [0, max_value_theory] , '--', c='green')

    # Show top values
    if num_top_values is not None:
        _, top_error_inds = torch.topk(abs(error), num_top_values)
        _, top_var_inds = torch.topk(uncertainty, num_top_values)
        ax.scatter(uncertainty[top_error_inds], error[top_error_inds], s=15, c='blue', label='Top 10 error')
        ax.scatter(uncertainty[top_var_inds], error[top_var_inds], s=15, c='green', marker='>', label='Top 10 variance')

    ax.set_xlim(0, max_value)
    if absolute:
        ax.set_ylim(0, max_value)
    else:
        ax.set_ylim(-max_value, max_value)

    if max_uncertainty is not None:
        ax.vlines(max_uncertainty, 0, max_value, linestyles='dashed', color='grey', zorder=0)

    if ticks is not None and absolute:
        set_ticks(ax, ticks, ticks, grid=False)
    elif ticks is not None and not absolute:
        y_ticks = sorted(ticks + [-x for x in ticks if x != 0])
        set_ticks(ax, ticks, y_ticks, grid=False)

    return ax


def calc_bin_dist(*args, min_samples=50, **kwargs):
    positions, bin_values, counts = separate_into_bins(*args, **kwargs)

    expectations = torch.stack([torch.mean(stack_or_zero(values)) for values in bin_values])
    stds = torch.stack([
        torch.std(stack_or_zero(values)) if len(values) > 1 else torch.tensor(float('nan'))
        for values in bin_values
    ])

    enough_samples = (counts > min_samples)
    return positions[enough_samples], expectations[enough_samples], stds[enough_samples]


def separate_into_bins(var, error, num_bins=None, bin_size=None):
    num_bins, bin_size = get_num_bins_and_bin_size(var, num_bins, bin_size)

    positions = torch.tensor([bin_size * i for i in range(num_bins)]) + bin_size / 2

    x_min = 0
    bin_indices = ((var - x_min) / bin_size).floor().long()

    bin_values = [[] for _ in range(num_bins)]
    counts = torch.zeros(num_bins)
    for idx, bin_idx in enumerate(bin_indices):
        bin_values[bin_idx].append(error[idx])
        counts[bin_idx] += 1

    return positions, bin_values, counts


def get_num_bins_and_bin_size(var, num_bins, bin_size):
    x_min = 0
    x_max = var.max()
    if bin_size is None and num_bins is not None:
        bin_size = (x_max - x_min) / (num_bins - 1)
    elif bin_size is not None and num_bins is None:
        num_bins = int((x_max/bin_size).ceil())
    return num_bins, bin_size


def stack_or_zero(values):
    if values:
        return torch.stack(values)
    else:
        return torch.tensor([0.0])


def plot_legend(ax, **kwargs):
    dashed_line = mlines.Line2D([], [], color='black', linestyle='--', label='Theoretical line')

    handles, labels = ax.get_legend_handles_labels()

    handles.append(dashed_line)
    labels.append('Theory')

    legend_handles = []
    for handle in handles:
        try:
            legend_handle = copy.copy(handle)
            legend_handle.set_sizes([100])
            legend_handles.append(legend_handle)
        except AttributeError:
            legend_handles.append(handle)

    legend = ax.legend(handles=legend_handles, labels=labels, **kwargs)
    legend.get_frame().set_alpha(1)
    return ax