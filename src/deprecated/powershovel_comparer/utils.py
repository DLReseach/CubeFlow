import pandas as pd
import numpy as np


def open_pandas_parquet_file(paths):
    df_list = []
    for i, path in enumerate(paths):
        df_list.append(pd.read_parquet(path, engine='fastparquet'))
    return df_list


def find_metrics(df, comparison_type):
    metrics = []
    for frame in df:
        metrics.append([s for s in frame.columns.values if comparison_type in s])
    if len(metrics) > 1:
        assert set(metrics[0]) - set(metrics[1]) == set(), 'Whoops, different metrics in frames!'
    metrics = metrics[0]
    return metrics


def calculate_bins(
    df,
    min_energy,
    max_energy,
    min_doms,
    max_doms,
    resolution
):
    df = df[df.energy <= max_energy]
    no_of_energy_bins = resolution
    no_of_dom_bins = max_doms
    energy_x_labels = np.linspace(min_energy, max_energy, resolution)
    dom_x_labels = np.arange(min_doms, max_doms)
    df['energy_binned'] = pd.cut(
        df['energy'],
        no_of_energy_bins,
        labels=energy_x_labels
    )
    df['doms_binned'] = pd.cut(
        df['event_length'],
        no_of_dom_bins,
        labels=dom_x_labels
    )
    df.energy_binned = df.energy_binned.astype('float')
    df.doms_binned = df.doms_binned.astype('float')
    energy_bins = np.sort(df.energy_binned.unique())
    dom_bins = np.sort(df.doms_binned.unique())
    out_dict = {
        'energy_bins': energy_bins,
        'energy_x_labels': energy_x_labels,
        'dom_bins': dom_bins,
        'dom_x_labels': dom_x_labels
    }
    return df, out_dict


def make_2d_histogram(x, y, clip_values, bins):
    H, xedges, yedges = np.histogram2d(
        x=np.clip(x.values, clip_values[0], clip_values[1]),
        y=y.values,
        bins=bins
    )
    return H


def find_min_max_percentiles(df, metric, percentiles):
    percentile_min = df[metric].quantile(percentiles[0])
    percentile_max = df[metric].quantile(percentiles[1])
    return [percentile_min, percentile_max]
