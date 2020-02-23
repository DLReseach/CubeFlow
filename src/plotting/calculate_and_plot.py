import io
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils.performance_data import PerformanceData
from utils.utils import get_time
from utils.utils import get_project_root
from plotting.plotting import plot_error_in_bin
from plotting.plotting import comparison_plot
from plotting.plotting import icecube_2d_histogram


def calculate_energy_bins(comparison_df):
    no_of_bins = 24
    comparison_df['energy_binned'] = pd.cut(
        comparison_df['true_energy'],
        no_of_bins
    )
    bins = comparison_df.energy_binned.unique()
    bins.sort_values(inplace=True)
    return bins


def calculate_dom_bins(comparison_df):
    no_of_bins = 20
    comparison_df['doms_binned'] = pd.cut(
        comparison_df['event_length'],
        no_of_bins
    )
    bins = comparison_df.doms_binned.unique()
    bins.sort_values(inplace=True)
    return bins


def calculate_and_plot(
    RUN_ROOT,
    config=None,
    wandb=None,
    dom_plots=False,
    use_train_dists=False,
    only_use_metrics=None,
    legends=True,
    reso_hists=False
):
    # COMPARISON_DF_FILE = RUN_ROOT.joinpath('comparison_dataframe_parquet.gzip')
    # comparison_df = pd.read_parquet(COMPARISON_DF_FILE, engine='fastparquet')
    COMPARISON_DF_FILE = RUN_ROOT.joinpath('comparison_dataframe.gzip')
    with open(COMPARISON_DF_FILE, 'rb') as f:
        comparison_df = pickle.load(f)


    comparison_df = comparison_df[comparison_df.true_energy <= 3.0]
    comparison_df.true_energy = 10**comparison_df.true_energy.values

    if use_train_dists:
        TRAIN_DATA_DF_FILE = get_project_root().joinpath(
            'train_distributions/train_data_parquet.gzip'
        )
        train_data_df = pd.read_parquet(TRAIN_DATA_DF_FILE, engine='fastparquet')
        train_data_df = train_data_df[train_data_df.train_true_energy <= 3.0]

    if only_use_metrics is not None:
        comparison_df = comparison_df[comparison_df.metric.isin(only_use_metrics)]

    PLOTS_DIR = RUN_ROOT.joinpath('plots')
    PLOTS_DIR.mkdir(exist_ok=True)
    RESO_PLOTS_DIR = PLOTS_DIR.joinpath('resolution_plots')
    RESO_PLOTS_DIR.mkdir(exist_ok=True)

    comparison_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    comparison_df.dropna(inplace=True)

    energy_bins = calculate_energy_bins(comparison_df)
    dom_bins = calculate_dom_bins(comparison_df)

    metrics = comparison_df.metric.unique()

    for metric in metrics:
        print(
            '{}: Plotting {} metric, binned in energy'
            .format(
                get_time(),
                metric
            )
        )
        performance_data = PerformanceData(
            comparison_df=comparison_df,
            bins=energy_bins,
            metric=metric,
            bin_type='energy',
            percentiles=[0.16, 0.84]
        )
        fig, markers_own = comparison_plot(
            performance_data,
            train_data_df.train_true_energy.values if use_train_dists else None,
            legends
        )
        file_name = PLOTS_DIR.joinpath(
            '{}_{}_reso_comparison.pdf'.format(
                'energy_bins',
                metric
            )
        )
        fig.savefig(file_name)
        if config is not None:
            if config.wandb:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                im = Image.open(buf)
                wandb.log(
                    {'{} resolution plot'.format(metric.title()): [wandb.Image(im)]}
                )
                buf.close()
                wandb.save(str(file_name))
                for x, y in zip(markers_own.get_data()[0], markers_own.get_data()[1]):
                    wandb.log(
                        {
                            '{} resolution comparison'.format(metric.title()): y,
                            'global_step': x
                        }
                    )
        plt.close(fig)
        fig = icecube_2d_histogram(performance_data, legends)
        file_name = PLOTS_DIR.joinpath(
            '{}_{}_ic_comparison.pdf'.format(
                'energy_bins',
                metric
            )
        )
        fig.savefig(file_name)
        if config is not None:
            if config.wandb:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                im = Image.open(buf)
                wandb.log(
                    {'{} IceCube histogram'.format(metric.title()): [wandb.Image(im)]}
                )
                buf.close()
                wandb.save(str(file_name))
        plt.close(fig)
        if reso_hists:
            for i, ibin in enumerate(energy_bins):
                indexer = (comparison_df.metric == metric)\
                    & (comparison_df.energy_binned == ibin)
                fig = plot_error_in_bin(
                    comparison_df[indexer].own_error,
                    comparison_df[indexer].opponent_error,
                    metric,
                    ibin,
                    'energy',
                    legends
                )
                file_name = RESO_PLOTS_DIR.joinpath(
                    '{}_{}_resolution_in_bin_{:02d}.pdf'.format(
                        'energy_bins',
                        metric,
                        i
                    )
                )
                fig.savefig(file_name)
                if config is not None:
                    if config.wandb:
                        wandb.save(str(file_name))
                plt.close(fig)

    if dom_plots:
        for metric in metrics:
            print(
                '{}: Plotting {} metric, binned in DOMs'
                .format(
                    get_time(),
                    metric
                )
            )
            performance_data = PerformanceData(
                comparison_df=comparison_df,
                bins=dom_bins,
                metric=metric,
                bin_type='doms',
                percentiles=[0.16, 0.84]
            )
            fig, markers_own = comparison_plot(
                performance_data,
                train_data_df.train_event_length.values if use_train_dists else None,
                legends
            )
            file_name = PLOTS_DIR.joinpath(
                '{}_{}_reso_comparison.pdf'.format(
                    'dom_bins',
                    metric
                )
            )
            fig.savefig(file_name)
            if config is not None:
                if config.wandb:
                    wandb.save(str(file_name))
            plt.close(fig)
            if reso_hists:
                for i, ibin in enumerate(dom_bins):
                    indexer = (comparison_df.metric == metric)\
                        & (comparison_df.doms_binned == ibin)
                    fig = plot_error_in_bin(
                        comparison_df[indexer].own_error,
                        comparison_df[indexer].opponent_error,
                        metric,
                        ibin,
                        'dom',
                        legends
                    )
                    file_name = RESO_PLOTS_DIR.joinpath(
                        '{}_{}_resolution_in_bin_{:02d}.pdf'.format(
                            'dom_bins',
                            metric,
                            i
                        )
                    )
                    fig.savefig(file_name)
                    if config is not None:
                        if config.wandb:
                            wandb.save(str(file_name))
                    plt.close(fig)
