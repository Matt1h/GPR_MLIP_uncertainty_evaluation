import matplotlib.ticker as ticker


UNCERTAINTY_INDS = {
    'Absolute error': 0,
    'Bootstrap aggregation': 1,
    'Random': 2,
    'Two sets': 3,
    'GPR standard deviation': 4,
}


COLORS = {
    'Red': (0.89, 0.29, 0.2),
    'Blue': (0.29, 0.39, 0.55),
    'Green': (0.35, 0.68, 0.45),
    'Orange': (0.96, 0.61, 0.2),
    'Olive': (0.52, 0.55, 0.26),
    'Purple': (0.50196078, 0., 0.50196078),
    'Light blue': (0.10980392, 0.30980392, 0.34901961),
}


def set_ticks(ax, x_ticks=None, y_ticks=None, grid=True, show_x_ticks=True, show_y_ticks=True):
    tick_params = {'labelsize': 12, 'which': 'both', 'zorder': 0}
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))

    if x_ticks is not None and show_x_ticks:
        ax.set_xticklabels(x_ticks, zorder=0)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

    if y_ticks is not None and show_y_ticks:
        ax.set_yticklabels(y_ticks, zorder=0)


    ax.xaxis.set_tick_params(**tick_params, labelbottom=show_x_ticks)
    ax.yaxis.set_tick_params(**tick_params, labelleft=show_y_ticks)
    if grid:
        ax.grid(True, zorder=0)
    return ax
