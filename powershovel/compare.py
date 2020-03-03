import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def open_pandas_parquet_file(path):
    df = pd.read_parquet(path, engine='fastparquet')
    return df


def calculate_energy_bins(df, min_energy, max_energy, resolution, labels):
    no_of_bins = resolution
    df['energy_binned'] = pd.cut(
        df['energy'],
        no_of_bins,
        labels=labels
    )
    df.energy_binned = df.energy_binned.astype('float')


def calculate_dom_bins(df, min_doms, max_doms, labels):
    no_of_bins = max_doms
    df['doms_binned'] = pd.cut(
        df['event_length'],
        no_of_bins,
        labels=labels
    )
    df.doms_binned = df.doms_binned.astype('float')


def make_2d_histogram(x, y, clip_values, bins):
    H, xedges, yedges = np.histogram2d(
        x=np.clip(x.values, clip_values[0], clip_values[1]),
        y=y.values,
        bins=bins
    )
    return H


def find_min_max_percentiles(df, metric):
    percentile_min = df[metric].quantile(0.03)
    percentile_max = df[metric].quantile(0.97)
    return [percentile_min, percentile_max]


runs_path = Path().home().joinpath('runs')
runs = [folder.name for folder in runs_path.iterdir() if folder.is_dir()]
boilerplate_columns = ['file_number', 'energy', 'event_length']

min_energy = 0.0
max_energy = 3.0
min_doms = 0
max_doms = 200
resolution_x = 200
resolution_y = 200

comparison_type = st.sidebar.radio(
    'Compare to...',
    ['other run', 'IceCube']
)

bin_in = st.sidebar.radio(
    'Bin in...',
    ['energy', 'DOMs']
)

if bin_in == 'energy':
    x_labels = np.linspace(min_energy, max_energy, resolution_x)
elif bin_in == 'DOMs':
    x_labels = np.arange(min_doms, max_doms)

if comparison_type == 'other run':
    run_1 = st.sidebar.selectbox(
        'Comparison 1',
        runs,
        index=0
    )
    df_1 = open_pandas_parquet_file(runs_path.joinpath(run_1 + '/error_dataframe_parquet.gzip'))

    run_2 = st.sidebar.selectbox(
        'Comparison 2',
        runs,
        index=1
    )
    df_2 = open_pandas_parquet_file(runs_path.joinpath(run_2 + '/error_dataframe_parquet.gzip'))

    metrics = [s for s in df_1.columns.values if 'own' in s]

    metric = st.sidebar.selectbox(
        'Metric',
        metrics,
        index=0,
        format_func=lambda x: x.split('_')[1]
    )

    clip_1 = find_min_max_percentiles(df_1, metric)
    clip_2 = find_min_max_percentiles(df_2, metric)
    clip_list = clip_1 + clip_2
    clip_y_min = min(clip_list)
    clip_y_max = max(clip_list)

    df_1 = df_1[df_1.energy <= max_energy]
    df_2 = df_2[df_2.energy <= max_energy]
    if bin_in == 'energy':
        calculate_energy_bins(df_1, min_energy, max_energy, resolution_x, x_labels)
        calculate_energy_bins(df_2, min_energy, max_energy, resolution_x, x_labels)
        H_1 = make_2d_histogram(df_1[metric], df_1.energy_binned, [clip_y_min, clip_y_max], [resolution_y, resolution_x])
        H_2 = make_2d_histogram(df_2[metric], df_2.energy_binned, [clip_y_min, clip_y_max], [resolution_y, resolution_x])
    elif bin_in == 'DOMs':
        calculate_dom_bins(df_1, min_doms, max_doms, x_labels)
        calculate_dom_bins(df_2, min_doms, max_doms, x_labels)
        H_1 = make_2d_histogram(df_1[metric], df_1.doms_binned, [clip_y_min, clip_y_max], [resolution_y, resolution_x])
        H_2 = make_2d_histogram(df_2[metric], df_2.doms_binned, [clip_y_min, clip_y_max], [resolution_y, resolution_x])

elif comparison_type == 'IceCube':
    run = st.sidebar.selectbox(
        'Run',
        runs,
        index=0
    )
    df = open_pandas_parquet_file(runs_path.joinpath(run + '/error_dataframe_parquet.gzip'))

    metrics = [s for s in df.columns.values if 'opponent' in s]

    metric = st.sidebar.selectbox(
        'Metric',
        metrics,
        index=0,
        format_func=lambda x: x.split('_')[1]
    )

    clip_1 = find_min_max_percentiles(df, metric.replace('opponent', 'own'))
    clip_2 = find_min_max_percentiles(df, metric)
    clip_list = clip_1 + clip_2
    clip_y_min = min(clip_list)
    clip_y_max = max(clip_list)

    df = df[df.energy <= max_energy]

    if bin_in == 'energy':
        calculate_energy_bins(df, min_energy, max_energy, resolution_x, x_labels)
        H_1 = make_2d_histogram(df[metric.replace('opponent', 'own')], df.energy_binned, [clip_y_min, clip_y_max], [resolution_y, resolution_x])
        H_2 = make_2d_histogram(df[metric], df.energy_binned, [clip_y_min, clip_y_max], [resolution_y, resolution_x])
    elif bin_in == 'DOMs':
        calculate_dom_bins(df, min_doms, max_doms, x_labels)
        H_1 = make_2d_histogram(df[metric.replace('opponent', 'own')], df.doms_binned, [clip_y_min, clip_y_max], [resolution_y, resolution_x])
        H_2 = make_2d_histogram(df[metric], df.doms_binned, [clip_y_min, clip_y_max], [resolution_y, resolution_x])


y_labels = np.linspace(clip_y_min, clip_y_max, resolution_y)

max_z_values = []
max_z_values.append(np.amax(H_1))
max_z_values.append(np.amax(H_2))
max_z_value = max(max_z_values)

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Heatmap(
        x=x_labels,
        y=y_labels,
        z=H_1,
        colorscale='Oranges',
        zmin=0,
        zmax=max_z_value,
        showscale=True
    ),
    row=1, col=1
)
fig.add_trace(
    go.Heatmap(
        x=x_labels,
        y=y_labels,
        z=H_2,
        colorscale='Oranges',
        zmin=0,
        zmax=max_z_value,
        showscale=False
    ),
    row=1, col=2
)
fig.update_layout(
    autosize=False,
    width=800,
    height=700
)


st.plotly_chart(fig, use_container_width=True)