import io
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from src.modules.performance_data import PerformanceData
from src.modules.utils import get_time
from src.modules.utils import get_project_root
from src.modules.plotting import plot_error_in_bin
from src.modules.plotting import comparison_plot
from src.modules.plotting import icecube_2d_histogram


def calculate_energy_bins(comparison_df):
    no_of_bins = 18
    comparison_df['energy_binned'] = pd.cut(
        comparison_df['energy'],
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
    files_and_dirs,
    dom_plots=False,
    use_train_dists=False,
    only_use_metrics=None,
    legends=True,
    reso_hists=False,
    use_own=True,
    reporter=None,
    wandb=False
):

    first_metric_plot = True

    file_name = files_and_dirs['run_root'].joinpath('error_dataframe_parquet.gzip')
    errors_df = pd.read_parquet(file_name, engine='fastparquet')

    errors_df = errors_df[errors_df.energy <= 3.0]
    # comparison_df.energy = 10**comparison_df.energy.values

    if use_train_dists:
        TRAIN_DATA_DF_FILE = files_and_dirs['project_root'].joinpath(
            'train_distributions/train_dists_parquet.gzip'
        )
        train_data_df = pd.read_parquet(TRAIN_DATA_DF_FILE, engine='fastparquet')
        train_data_df = train_data_df[train_data_df.train_true_energy <= 3.0]

    if only_use_metrics is not None:
        errors_df = errors_df[errors_df.metric.isin(only_use_metrics)]

    PLOTS_DIR = files_and_dirs['run_root'].joinpath('plots')
    PLOTS_DIR.mkdir(exist_ok=True)
    RESO_PLOTS_DIR = PLOTS_DIR.joinpath('resolution_plots')
    RESO_PLOTS_DIR.mkdir(exist_ok=True)

    errors_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    errors_df.dropna(inplace=True)

    energy_bins = calculate_energy_bins(errors_df)
    dom_bins = calculate_dom_bins(errors_df)

    metrics = [metric.replace('own_', '').replace('_error', '') for metric in errors_df.keys() if not metric.find('own')]

    print('{}: Calculating performance data for energy bins'.format(get_time()))
    performance_data = PerformanceData(
        metrics,
        df=errors_df,
        bins=energy_bins,
        bin_type='energy',
        percentiles=[0.16, 0.84],
        use_own=use_own
    )

    print(performance_data)

    for metric in metrics:
        print(
            '{}: Plotting {} metric, binned in energy'
            .format(
                get_time(),
                metric
            )
        )
        fig, markers_own = comparison_plot(
            metric,
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
        if wandb:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            im = Image.open(buf)
            log_text = '{} resolution plot'.format(metric.title())
            reporter.add_plot_to_wandb(im, log_text)
            buf.close()
            log_text = '{} resolution comparison'.format(metric.title())
            reporter.add_metric_comparison_to_wandb(markers_own, log_text)
        plt.close(fig)
        fig = icecube_2d_histogram(metric, performance_data, legends)
        file_name = PLOTS_DIR.joinpath(
            '{}_{}_ic_comparison.pdf'.format(
                'energy_bins',
                metric
            )
        )
        fig.savefig(file_name)
        if wandb:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            im = Image.open(buf)
            log_text = '{} IceCube histogram'.format(metric.title())
            reporter.add_plot_to_wandb(im, log_text)
            buf.close()
        plt.close(fig)
        if reso_hists:
            for i, ibin in enumerate(energy_bins):
                indexer = errors_df.energy_binned == ibin
                fig = plot_error_in_bin(
                    errors_df[indexer]['own_' + metric + '_error'].values,
                    errors_df[indexer]['opponent_' + metric + '_error'].values,
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
                if wandb:
                    reporter.save_file_to_wandb(str(file_name))
                plt.close(fig)

    if dom_plots:
        print('{}: Calculating performance data for DOM bins'.format(get_time()))
        performance_data = PerformanceData(
            metrics,
            df=errors_df,
            bins=dom_bins,
            bin_type='doms',
            percentiles=[0.16, 0.84],
            use_own=use_own
        )
        for metric in metrics:
            print(
                '{}: Plotting {} metric, binned in DOMs'
                .format(
                    get_time(),
                    metric
                )
            )
            fig, markers_own = comparison_plot(
                metric,
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
            if wandb:
                reporter.save_file_to_wandb(str(file_name))
            plt.close(fig)
            if reso_hists:
                for i, ibin in enumerate(dom_bins):
                    indexer = errors_df.doms_binned == ibin
                    fig = plot_error_in_bin(
                        errors_df[indexer]['own_' + metric + '_error'].values,
                        errors_df[indexer]['opponent_' + metric + '_error'].values,
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
                    if wandb:
                        reporter.save_file_to_wandb(str(file_name))
                    plt.close(fig)
