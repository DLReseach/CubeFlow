# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from matplotlib.backends.backend_pgf import FigureCanvasPgf

# mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

plt.style.use('default')

params = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # "figure.figsize": mpl.figsize(0.9),
    "text.latex.preamble": [
        r"\usepackage{mathpazo}"
    ]
}

mpl.rcParams.update(params)


def plot_error_in_bin(own, opponent, metric, bins_range, bins_type):
    fig, ax = plt.subplots()
    ax.hist(
        own,
        bins='fd',
        alpha=0.5,
        histtype='stepfilled',
        label=r'CubeFlow'
    )
    ax.hist(
        opponent,
        bins='fd',
        alpha=0.5,
        histtype='stepfilled',
        label=r'IceCube'
    )
    ax.set(
        ylabel=r'Frequency',
        title=r'{0} resolution in {1} {2} bin'.format(
            metric.title(),
            bins_range,
            bins_type
        )
    )
    if metric == 'azimuth':
        ax.set(xlabel=r'$\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}} \; [\mathrm{Rad}]$')
    elif metric == 'zenith':
        ax.set(xlabel=r'$\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}} \; [\mathrm{Rad}]$')
    elif metric == 'energy':
        ax.set(xlabel=r'$\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}} \; [\%]$')
    elif metric == 'time':
        ax.set(xlabel=r'$t_{\mathrm{reco}} - t_{\mathrm{true}} \; [\mathrm{ns}]$')
    ax.legend()
    text_str = r'{} events in bin'.format(len(own))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props
    )
    fig.tight_layout()
    return fig


def comparison_plot(performance_data, train_data):
    fig, (reso_ax, ratio_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={
            'height_ratios': [3, 1],
            'hspace': 0.05
        }
    )
    hist_ax = reso_ax.twinx()
    if performance_data.bin_type == 'energy':
        hist_ax.hist(
            performance_data.comparison_df.true_energy.values,
            bins=len(performance_data.bins),
            histtype='step',
            color='grey',
            label='Validation data',
            alpha=0.5,
            linestyle='solid'
        )
        hist_ax.hist(
            train_data,
            bins=len(performance_data.bins),
            histtype='step',
            color='grey',
            label='Train data',
            alpha=0.5,
            linestyle='dashed'
        )
    elif performance_data.bin_type == 'doms':
        hist_ax.hist(
            performance_data.comparison_df.event_length.values,
            bins=len(performance_data.bins),
            histtype='step',
            color='grey'
        )
    reso_ax.xaxis.set_ticks_position('none')
    markers_own, caps, bars = reso_ax.errorbar(
        performance_data.bin_centers,
        performance_data.own_performances,
        yerr=performance_data.own_sigmas,
        xerr=performance_data.bin_widths,
        marker='.',
        markersize=1,
        ls='none',
        label=r'CubeFlow'
    )
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    markers, caps, bars = reso_ax.errorbar(
        performance_data.bin_centers,
        performance_data.opponent_performances,
        yerr=performance_data.opponent_sigmas,
        xerr=performance_data.bin_widths,
        marker='.',
        markersize=1,
        ls='none',
        label=r'IceCube'
    )
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    ratio_ax.axhline(y=0, linestyle='dashed', linewidth=0.5, color='black')
    ratio_ax.axhline(y=1, linestyle='dashdot', linewidth=0.5, color='black')
    ratio_ax.axhline(y=-1, linestyle='dashdot', linewidth=0.5, color='black')
    ratio_ax.plot(
        performance_data.bin_centers,
        performance_data.relative_improvement,
        '.',
        markersize=4,
        color='red'
    )
    ratio_ax.set(
        ylim=[-2, 2],
        yticks=[-1, 0, 1],
        ylabel=r'Rel. imp.'
    )
    hist_ax.set_yscale('symlog')
    hist_ax.set_ylim(ymin=0)
    if performance_data.metric == 'energy':
        reso_ax.set_ylim(ymin=0, ymax=2)
    else:
        reso_ax.set_ylim(ymin=0)
    if performance_data.bin_type == 'energy':
        ratio_ax.set(xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$')
    elif performance_data.bin_type == 'doms':
        ratio_ax.set(xlabel=r'No. of DOMs')
    reso_ax.set(
        title=r'{} reconstruction comparison'.format(
            performance_data.metric.title()
        )
    )
    if performance_data.metric == 'azimuth':
        reso_ax.set(
            ylabel=r'$\sigma\left(\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}}\right) \; [\mathrm{Rad}]$'
        )
    if performance_data.metric == 'zenith':
        reso_ax.set(
            ylabel=r'$\sigma\left(\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}}\right) \; [\mathrm{Rad}]$'
        )
    elif performance_data.metric == 'energy':
        reso_ax.set(
            ylabel=r'$\sigma\left(\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}}\right) \; [\%]$'
        )
    elif performance_data.metric == 'time':
        reso_ax.set(
            ylabel=r'$\sigma\left(t_{\mathrm{reco}} - t_{\mathrm{true}}\right) \; [\mathrm{ns}]$'
        )
    hist_ax.set(ylabel=r'Events')
    reso_ax.legend(loc='upper right')
    hist_ax.legend(loc='upper left')
    # fig.tight_layout()
    return fig, markers_own


def icecube_2d_histogram(performance_data):
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16.0, 6.0)
    )
    indexer = performance_data.comparison_df.metric == performance_data.metric
    x_values = performance_data.comparison_df[indexer].true_energy.values
    y_values_own = performance_data.comparison_df[indexer].own_error.values
    _, x_bin_edges = np.histogram(x_values, bins='fd', range=[0, 4])
    if performance_data.metric == 'azimuth':
        plot_range = [-2.5, 2.5]
    if performance_data.metric == 'energy':
        plot_range = [-1, 4]
    if performance_data.metric == 'time':
        plot_range = [-150, 250]
    if performance_data.metric == 'zenith':
        plot_range = [-2, 2]
    _, y_bin_edges_own = np.histogram(
        y_values_own,
        bins='fd',
        range=plot_range
    )
    widths1_own = np.linspace(
        min(x_bin_edges),
        max(x_bin_edges),
        int(0.5 + x_bin_edges.shape[0] / 4.0)
    )
    widths2_own = np.linspace(
        min(y_bin_edges_own),
        max(y_bin_edges_own),
        int(0.5 + y_bin_edges_own.shape[0] / 4.0)
    )
    y_values_opponent = performance_data.comparison_df[indexer].opponent_error.values
    _, y_bin_edges_opponent = np.histogram(
        y_values_opponent,
        bins='fd',
        range=plot_range
    )
    widths1_opponent = np.linspace(
        min(x_bin_edges),
        max(x_bin_edges),
        int(0.5 + x_bin_edges.shape[0] / 4.0)
    )
    widths2_opponent = np.linspace(
        min(y_bin_edges_opponent),
        max(y_bin_edges_opponent),
        int(0.5 + y_bin_edges_opponent.shape[0] / 4.0)
    )
    counts_own, xedges_own, yedges_own, im_own = ax1.hist2d(
        performance_data.comparison_df[indexer].true_energy.values,
        performance_data.comparison_df[indexer].own_error.values,
        bins=[widths1_own, widths2_own],
        cmap='Oranges'
    )
    ax1.plot(
        performance_data.own_centers,
        performance_data.own_medians,
        linestyle='solid',
        color='red',
        alpha=0.5,
        label=r'50 \%  (CubeFlow)'
    )
    ax1.plot(
        performance_data.own_centers,
        performance_data.own_lows,
        linestyle='dashed',
        color='red',
        alpha=0.5,
        label=r'16 \% (CubeFlow)'
    )
    ax1.plot(
        performance_data.own_centers,
        performance_data.own_highs,
        linestyle='dotted',
        color='red',
        alpha=0.5,
        label=r'84 \% (CubeFlow)'
    )
    ax1.plot(
        performance_data.opponent_centers,
        performance_data.opponent_medians,
        linestyle='solid',
        color='green',
        alpha=0.5,
        label=r'50 \% (IceCube)'
    )
    ax1.plot(
        performance_data.opponent_centers,
        performance_data.opponent_lows,
        linestyle='dashed',
        color='green',
        alpha=0.5,
        label=r'16 \% (IceCube)'
    )
    ax1.plot(
        performance_data.opponent_centers,
        performance_data.opponent_highs,
        linestyle='dotted',
        color='green',
        alpha=0.5,
        label=r'84 \% (IceCube)'
    )
    cb_own = fig.colorbar(im_own, ax=ax1)
    cb_own.set_label('Frequency')
    ax1.set(
        title=r'{} reconstruction results, CubeFlow'.format(
            performance_data.metric.title()
        ),
        xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$'
    )
    if performance_data.metric == 'energy':
        ax1.set(
            ylabel=r'$\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}} \; [\%]$'
        )
    elif performance_data.metric == 'azimuth':
        ax1.set(
            ylabel=r'$\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}} \; [\mathrm{Rad}]$'
        )
    elif performance_data.metric == 'zenith':
        ax1.set(
            ylabel=r'$\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}} \; [\mathrm{Rad}]$'
        )
    elif performance_data.metric == 'time':
        ax1.set(
            ylabel=r'$t_{\mathrm{reco}} - t_{\mathrm{true}} \; [\mathrm{ns}]$'
        )
    ax1.legend()
    counts_opponent, xedges_opponent, yedges_opponent, im_opponent = ax2.hist2d(
        performance_data.comparison_df[indexer].true_energy.values,
        performance_data.comparison_df[indexer].opponent_error.values,
        bins=[widths1_opponent, widths2_opponent],
        cmap='Oranges'
    )
    ax2.plot(
        performance_data.own_centers,
        performance_data.own_medians,
        linestyle='solid',
        color='red',
        alpha=0.5,
        label=r'50 \% (CubeFlow)'
    )
    ax2.plot(
        performance_data.own_centers,
        performance_data.own_lows,
        linestyle='dashed',
        color='red',
        alpha=0.5,
        label=r'16 \% (CubeFlow)'
    )
    ax2.plot(
        performance_data.own_centers,
        performance_data.own_highs,
        linestyle='dotted',
        color='red',
        alpha=0.5,
        label=r'84 \% (CubeFlow)'
    )
    ax2.plot(
        performance_data.opponent_centers,
        performance_data.opponent_medians,
        linestyle='solid',
        alpha=0.5,
        color='green',
        label=r'50 \% (IceCube)'
    )
    ax2.plot(
        performance_data.opponent_centers,
        performance_data.opponent_lows,
        linestyle='dashed',
        alpha=0.5,
        color='green',
        label=r'16 \% (IceCube)'
    )
    ax2.plot(
        performance_data.opponent_centers,
        performance_data.opponent_highs,
        linestyle='dotted',
        alpha=0.5,
        color='green',
        label=r'84 \% (IceCube)'
    )
    cb_opponent = fig.colorbar(im_opponent, ax=ax2)
    cb_opponent.set_label('Frequency')
    ax2.set(
        title=r'{} reconstruction results, IceCube'.format(
            performance_data.metric.title()
        ),
        xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$')
    if performance_data.metric == 'energy':
        ax2.set(
            ylabel=r'$\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}} \; [\%]$'
        )
    elif performance_data.metric == 'azimuth':
        ax2.set(
            ylabel=r'$\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}} \; [\mathrm{Rad}]$'
        )
    elif performance_data.metric == 'zenith':
        ax2.set(
            ylabel=r'$\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}} \; [\mathrm{Rad}]$'
        )
    elif performance_data.metric == 'time':
        ax2.set(
            ylabel=r'$t_{\mathrm{reco}} - t_{\mathrm{true}} \; [\mathrm{ns}]$'
        )
    ax2.legend()
    fig.tight_layout()
    return fig
