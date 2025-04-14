from os.path import join
import numpy as np
import torch
import matplotlib.pyplot as plt

from .utils import set_ticks, COLORS, UNCERTAINTY_INDS


def plot_us_results(
        al_dir,
        n_init,
        n_al,
        x_ticks,
        y_ticks,
        uncertainty_names=None,
        plot_correlation=True,
        plot_p=False
):
    n_samples = range(n_init, n_init + n_al)

    if plot_correlation:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # plot metrics
    if uncertainty_names is None:
        uncertainty_names = ['Absolute error', 'Random', 'Two sets', 'GPR variance']
    metrics = load_us_metrics(al_dir, uncertainty_names)
    metrics = adjust_size(metrics, n_al)
    # metrics = pad_with_nans(metrics, n_al)

    for i, (metric_name, diff_uncertainty_metric) in enumerate(metrics.items()):
        axes[i] = plot_single_us_metric(
            axes[i],
            n_samples,
            diff_uncertainty_metric,
            x_ticks,
            y_ticks[i],
            title=metric_name
        )
    axes[0].legend(fontsize=12, loc='lower left')

    # plot correlation
    if plot_correlation:
        correlation_uncertainty_inds = [1, 3, 4]
        correlation = load_us_correlation(
            al_dir,
            ['Bootstrap aggregation', 'Two sets', 'GPR variance'],
            correlation_uncertainty_inds
        )
        correlation = adjust_size(correlation, n_al)

        y_ticks_corr = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]
        colors = list(COLORS.values())
        colors = [colors[i] for i in correlation_uncertainty_inds]
        axes[-1] = plot_single_cbal_metric(
            axes[-1],
            n_samples,
            correlation['pearsonr'],
            x_ticks,
            y_ticks_corr,
            title='Pearson correlation coefficient',
            logy=False,
            colors=colors,
            label_suffix=' - error'
        )

        if plot_p:
            axes[-1] = plot_single_cbal_metric(
                axes[-1],
                n_samples,
                correlation['p-value'],
                x_ticks,
                y_ticks_corr,
                logy=False,
                linestyle='--',
                colors=colors,
                label_prefix='PCC ',
                label_suffix=' - error'
            )
        axes[-1].legend(fontsize=14)

    plt.tight_layout()


def load_us_metrics(al_dir, uncertainty_names):
    metric_file_names = ['maes', 'max_errors', 'variances']
    metric_plot_names = ['MAE in meV', 'Max absolute error in meV', 'Test variance in meV']

    metrics = {}
    for metric_file_name, metric_plot_name in zip(metric_file_names, metric_plot_names):
        metrics[metric_plot_name] = {}
        for uncertainty_name in uncertainty_names:
            metrics[metric_plot_name][uncertainty_name] = torch.tensor(torch.load(
                join(al_dir, f'{UNCERTAINTY_INDS[uncertainty_name]}', f'{metric_file_name}'),
                map_location=torch.device('cpu'),
                weights_only=False
            ))*1000
    return metrics


def load_us_correlation(al_dir, correlation_uncertainty_names, correlation_uncertainty_inds):
    metric_names = ['pearsonr', 'p-value']

    metrics = {}
    for metric_name in metric_names:
        metrics[metric_name] = {}
        for i, uncertainty_name in zip(correlation_uncertainty_inds, correlation_uncertainty_names):
            metrics[metric_name][uncertainty_name] = torch.load(
                join(al_dir, f'{i}', f'{metric_name}'),
                map_location=torch.device('cpu'),
                weights_only=False
            )
    return metrics


def adjust_size(metrics, target_length):
    for metric_name, diff_uncertainty_metric in metrics.items():
        for uncertainty_name, metric in diff_uncertainty_metric.items():
            diff_uncertainty_metric[uncertainty_name] = pad_array_with_nans(np.array(metric), target_length)
    return metrics


def pad_array_with_nans(array, target_length):
    padding_size = target_length - len(array)
    if target_length > len(array):
        return np.pad(array, (0, padding_size), mode='constant', constant_values=np.nan)
    else:
        return array[:target_length]


def plot_single_us_metric(
        ax,
        n_samples,
        diff_uncertainty_metric,
        x_ticks=None,
        y_ticks=None,
        title=None,
        logx=True,
        logy=True,
        linestyle='-',
        colors=None,
        label_prefix='',
        label_suffix='',
):
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if colors is None:
        colors = list(COLORS.values())

    for i, (uncertainty_name, metric) in enumerate(diff_uncertainty_metric.items()):
        ax.plot(
            n_samples,
            torch.tensor(metric),
            linestyle=linestyle,
            c=colors[i],
            label=f'{label_prefix}{uncertainty_name}{label_suffix}'
        )
        set_ticks(ax, x_ticks, y_ticks)
        if title is not None:
            ax.set_title(title, fontsize=16)
    ax.set_xlabel(f'Number of training samples', fontsize=13)
    return ax
