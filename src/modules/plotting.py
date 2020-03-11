# %%
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.texmanager').disabled = True

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


def plot_error_in_bin(own, opponent, metric, bins_range, bins_type, legends=True):
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
    if legends:
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


def comparison_plot(metric, performance_data, train_data, legends=True):
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
        if train_data is None:
            hist_ax.hist(
                performance_data.df.energy.values,
                bins=len(performance_data.bins),
                histtype='step',
                color='grey',
                label='Validation data',
                alpha=0.5,
                linestyle='solid'
            )
        else:
            hist_ax.hist(
                train_data,
                bins=len(performance_data.bins),
                histtype='step',
                color='grey',
                label='Train data',
                alpha=0.5,
                linestyle='solid'
            )
    elif performance_data.bin_type == 'doms':
        if train_data is None:
            hist_ax.hist(
                performance_data.df.event_length.values,
                bins=len(performance_data.bins),
                histtype='step',
                color='grey',
                label='Validation data',
                alpha=0.5,
                linestyle='solid'
            )
        else:
            hist_ax.hist(
                train_data,
                bins=len(performance_data.bins),
                histtype='step',
                color='grey',
                label='Train data',
                alpha=0.5,
                linestyle='solid'
            )
    reso_ax.xaxis.set_ticks_position('none')
    markers_own, caps, bars = reso_ax.errorbar(
        performance_data.bin_centers,
        performance_data.performances_dict[metric]['own_performances'],
        yerr=performance_data.performances_dict[metric]['own_sigmas'],
        xerr=performance_data.bin_widths,
        marker='.',
        ls='none',
        label=r'CubeFlow'
    )
    markers, caps, bars = reso_ax.errorbar(
        performance_data.bin_centers,
        performance_data.performances_dict[metric]['opponent_performances'],
        yerr=performance_data.performances_dict[metric]['opponent_sigmas'],
        xerr=performance_data.bin_widths,
        marker='.',
        ls='none',
        label=r'IceCube'
    )
    ratio_ax.axhline(y=0, linestyle='dashed', linewidth=0.5, color='black')
    # ratio_ax.axhline(y=1, linestyle='dashdot', linewidth=0.5, color='black')
    # ratio_ax.axhline(y=-1, linestyle='dashdot', linewidth=0.5, color='black')
    ratio_ax.errorbar(
        performance_data.bin_centers,
        performance_data.performances_dict[metric]['relative_improvement'],
        yerr=performance_data.performances_dict[metric]['relative_improvement_sigmas'],
        xerr=performance_data.bin_widths,
        marker='.',
        ls='none'
    )
    ratio_ax.set(
        # xticks=[0, 1, 2, 3],
        ylim=[-0.5, 0.5],
        yticks=[-0.25, 0, 0.25],
        ylabel=r'Rel. imp.'
    )
    hist_ax.set_yscale('log')
    hist_ax.set_ylim(ymin=1e2)
    if metric == 'energy':
        reso_ax.set_ylim(ymin=0, ymax=2)
    else:
        reso_ax.set_ylim(ymin=0)
    if performance_data.bin_type == 'energy':
        ratio_ax.set(xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$')
        # ratio_ax.set(xlabel=r'$E_{\mathrm{true}} \; [\mathrm{GeV}]$')
    elif performance_data.bin_type == 'doms':
        ratio_ax.set(xlabel=r'No. of DOMs')
    reso_ax.set(
        title=r'{} reconstruction comparison'.format(
            metric.title()
        )
    )
    if metric == 'azimuth':
        reso_ax.set(
            ylabel=r'$\sigma\left(\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}}\right) \; [\mathrm{Rad}]$'
        )
    if metric == 'zenith':
        reso_ax.set(
            ylabel=r'$\sigma\left(\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}}\right) \; [\mathrm{Rad}]$'
        )
    elif metric == 'energy':
        reso_ax.set(
            ylabel=r'$\sigma\left(\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}}\right) \; [\%]$'
        )
    elif metric == 'time':
        reso_ax.set(
            ylabel=r'$\sigma\left(t_{\mathrm{reco}} - t_{\mathrm{true}}\right) \; [\mathrm{ns}]$'
        )
    # ratio_ax.set_xticklabels([10**x for x in ratio_ax.get_xticks()])
    # labels = [item.get_text() for item in ratio_ax.get_xticklabels()]
    # labels[0] = '0'
    # ratio_ax.set_xticklabels(labels)
    hist_ax.set(ylabel=r'Events')
    if legends:
        reso_lines, reso_labels = reso_ax.get_legend_handles_labels()
        hist_lines, hist_labels = hist_ax.get_legend_handles_labels()
        hist_ax.legend(reso_lines + hist_lines, reso_labels + hist_labels)
    return fig, markers_own


def icecube_2d_histogram(metric, performance_data, legends=True):
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16.0, 6.0)
    )
    x_values = performance_data.df.energy.values
    y_values_own = performance_data.df['own_' + metric + '_error'].values
    _, x_bin_edges = np.histogram(x_values, bins='fd', range=[0, 4])
    if metric == 'azimuth':
        plot_range = [-2.5, 2.5]
    if metric == 'energy':
        plot_range = [-1, 4]
    if metric == 'time':
        plot_range = [-150, 250]
    if metric == 'zenith':
        plot_range = [-2, 2]
    if metric == 'x' or metric == 'y' or metric == 'z':
        plot_range = [-50, 75]
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
    y_values_opponent = performance_data.df['opponent_' + metric + '_error'].values
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
        performance_data.df.energy.values,
        performance_data.df['own_' + metric + '_error'].values,
        bins=[widths1_own, widths2_own],
        cmap='Oranges',
        density=True
    )
    ax1.plot(
        performance_data.performances_dict[metric]['own_centers'],
        performance_data.performances_dict[metric]['own_medians'],
        linestyle='solid',
        color='red',
        alpha=0.5,
        label=r'50 \%  (CubeFlow)'
    )
    ax1.plot(
        performance_data.performances_dict[metric]['own_centers'],
        performance_data.performances_dict[metric]['own_lows'],
        linestyle='dashed',
        color='red',
        alpha=0.5,
        label=r'16 \% (CubeFlow)'
    )
    ax1.plot(
        performance_data.performances_dict[metric]['own_centers'],
        performance_data.performances_dict[metric]['own_highs'],
        linestyle='dotted',
        color='red',
        alpha=0.5,
        label=r'84 \% (CubeFlow)'
    )
    ax1.plot(
        performance_data.performances_dict[metric]['opponent_centers'],
        performance_data.performances_dict[metric]['opponent_medians'],
        linestyle='solid',
        color='green',
        alpha=0.5,
        label=r'50 \% (IceCube)'
    )
    ax1.plot(
        performance_data.performances_dict[metric]['opponent_centers'],
        performance_data.performances_dict[metric]['opponent_lows'],
        linestyle='dashed',
        color='green',
        alpha=0.5,
        label=r'16 \% (IceCube)'
    )
    ax1.plot(
        performance_data.performances_dict[metric]['opponent_centers'],
        performance_data.performances_dict[metric]['opponent_highs'],
        linestyle='dotted',
        color='green',
        alpha=0.5,
        label=r'84 \% (IceCube)'
    )
    cb_own = fig.colorbar(im_own, ax=ax1)
    cb_own.set_label('Frequency')
    ax1.set(
        title=r'{} reconstruction results, CubeFlow'.format(
            metric.title()
        ),
        xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$'
    )
    if metric == 'energy':
        ax1.set(
            ylabel=r'$\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}} \; [\%]$'
        )
    elif metric == 'azimuth':
        ax1.set(
            ylabel=r'$\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}} \; [\mathrm{Rad}]$'
        )
    elif metric == 'zenith':
        ax1.set(
            ylabel=r'$\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}} \; [\mathrm{Rad}]$'
        )
    elif metric == 'time':
        ax1.set(
            ylabel=r'$t_{\mathrm{reco}} - t_{\mathrm{true}} \; [\mathrm{ns}]$'
        )
    if legends:
        ax1.legend()
    counts_opponent, xedges_opponent, yedges_opponent, im_opponent = ax2.hist2d(
        performance_data.df.energy.values,
        performance_data.df['opponent_' + metric + '_error'].values,
        bins=[widths1_opponent, widths2_opponent],
        cmap='Oranges',
        density=True
    )
    ax2.plot(
        performance_data.performances_dict[metric]['own_centers'],
        performance_data.performances_dict[metric]['own_medians'],
        linestyle='solid',
        color='red',
        alpha=0.5,
        label=r'50 \% (CubeFlow)'
    )
    ax2.plot(
        performance_data.performances_dict[metric]['own_centers'],
        performance_data.performances_dict[metric]['own_lows'],
        linestyle='dashed',
        color='red',
        alpha=0.5,
        label=r'16 \% (CubeFlow)'
    )
    ax2.plot(
        performance_data.performances_dict[metric]['own_centers'],
        performance_data.performances_dict[metric]['own_highs'],
        linestyle='dotted',
        color='red',
        alpha=0.5,
        label=r'84 \% (CubeFlow)'
    )
    ax2.plot(
        performance_data.performances_dict[metric]['opponent_centers'],
        performance_data.performances_dict[metric]['opponent_medians'],
        linestyle='solid',
        alpha=0.5,
        color='green',
        label=r'50 \% (IceCube)'
    )
    ax2.plot(
        performance_data.performances_dict[metric]['opponent_centers'],
        performance_data.performances_dict[metric]['opponent_lows'],
        linestyle='dashed',
        alpha=0.5,
        color='green',
        label=r'16 \% (IceCube)'
    )
    ax2.plot(
        performance_data.performances_dict[metric]['opponent_centers'],
        performance_data.performances_dict[metric]['opponent_highs'],
        linestyle='dotted',
        alpha=0.5,
        color='green',
        label=r'84 \% (IceCube)'
    )
    cb_opponent = fig.colorbar(im_opponent, ax=ax2)
    cb_opponent.set_label('Frequency')
    ax2.set(
        title=r'{} reconstruction results, IceCube'.format(
            metric.title()
        ),
        xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$')
    if metric == 'energy':
        ax2.set(
            ylabel=r'$\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}} \; [\%]$'
        )
    elif metric == 'azimuth':
        ax2.set(
            ylabel=r'$\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}} \; [\mathrm{Rad}]$'
        )
    elif metric == 'zenith':
        ax2.set(
            ylabel=r'$\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}} \; [\mathrm{Rad}]$'
        )
    elif metric == 'time':
        ax2.set(
            ylabel=r'$t_{\mathrm{reco}} - t_{\mathrm{true}} \; [\mathrm{ns}]$'
        )
    if legends:
        ax2.legend()
    fig.tight_layout()
    return fig
