import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from utils import open_pandas_parquet_file
from utils import find_min_max_percentiles
from utils import calculate_bins
from utils import find_metrics
from utils import make_2d_histogram
from compare_ic_plot import compare_ic_histogram

runs_path = Path().home().joinpath('runs')
runs = [folder.name for folder in runs_path.iterdir() if folder.is_dir()]
boilerplate_columns = ['file_number', 'energy', 'event_length']

min_energy = 0.0
max_energy = 3.0
min_doms = 0
max_doms = 200
resolution_x = 200
resolution_y = 200
percentiles = [0.00, 1]

comparison_type = st.sidebar.radio(
    'Compare to...',
    ['other run', 'IceCube']
)

bin_in = st.sidebar.radio(
    'Bin in...',
    ['energy', 'DOMs']
)

if comparison_type == 'other run':
    run_1 = st.sidebar.selectbox(
        'Comparison 1',
        runs,
        index=0
    )
    run_2 = st.sidebar.selectbox(
        'Comparison 2',
        runs,
        index=1
    )
    df_list = open_pandas_parquet_file(
        [
            runs_path.joinpath(run_1 + '/error_dataframe_parquet.gzip'),
            runs_path.joinpath(run_2 + '/error_dataframe_parquet.gzip')
        ]
    )
    metrics = find_metrics(df_list, 'own')
    metric = st.sidebar.selectbox(
        'Metric',
        metrics,
        index=0,
        format_func=lambda x: x.split('_')[1]
    )

    clip = []
    bins = []
    for i, df in enumerate(df_list):
        clip.extend(find_min_max_percentiles(df, metric, percentiles))
        df_list[i], bins_temp = calculate_bins(df, min_energy, max_energy, min_doms, max_doms, resolution_x)
        bins.append(bins_temp)

    clip_y = [min(clip), max(clip)]

    H = []

    if bin_in == 'energy':
        for i, df in enumerate(df_list):
            H.append(
                make_2d_histogram(
                    df[metric],
                    df.energy_binned,
                    clip_y,
                    [resolution_y, resolution_x]
                )
            )
        x_labels = bins[0]['energy_x_labels']
#         median = []
#         bins_iter = sorted(list(df_1.energy_binned.unique()))
#         for i in bins_iter:
#             median.append(df_1[df_1.energy_binned == i][metric].quantile(0.5))
    elif bin_in == 'DOMs':
        for i, df in enumerate(df_list):
            H.append(
                make_2d_histogram(
                    df[metric],
                    df.doms_binned,
                    clip_y,
                    [resolution_y, resolution_x]
                )
            )
        x_labels = bins[0]['dom_x_labels']

    meta = {'title_1': run_1, 'title_2': run_2, 'y_label': metric.replace('own_', ''), 'x_label': bin_in}

elif comparison_type == 'IceCube':
    run = st.sidebar.selectbox(
        'Run',
        runs,
        index=0
    )
    df_list = open_pandas_parquet_file([runs_path.joinpath(run + '/error_dataframe_parquet.gzip')])

    metrics = find_metrics(df_list, 'opponent')
    metric = st.sidebar.selectbox(
        'Metric',
        metrics,
        index=0,
        format_func=lambda x: x.split('_')[1]
    )

    clip = []
    bins = []
    for i, df in enumerate(df_list):
        clip.extend(find_min_max_percentiles(df, metric, percentiles))
        df_list[i], bins_temp = calculate_bins(df, min_energy, max_energy, min_doms, max_doms, resolution_x)
        bins.append(bins_temp)

    clip_y = [min(clip), max(clip)]

    H = []
    metrics = [metric.replace('opponent', 'own'), metric]
    print(metrics)

    if bin_in == 'energy':
        for i, metric in enumerate(metrics):
            H.append(
                make_2d_histogram(
                    df_list[0][metric],
                    df_list[0].energy_binned,
                    clip_y,
                    [resolution_y, resolution_x]
                )
            )
        x_labels = bins[0]['energy_x_labels']
#         median = []
#         bins_iter = sorted(list(df_1.energy_binned.unique()))
#         for i in bins_iter:
#             median.append(df_1[df_1.energy_binned == i][metric].quantile(0.5))
    elif bin_in == 'DOMs':
        for i, metric in enumerate(metrics):
            H.append(
                make_2d_histogram(
                    df_list[0][metric],
                    df_list[0].doms_binned,
                    clip_y,
                    [resolution_y, resolution_x]
                )
            )
        x_labels = bins[0]['dom_x_labels']
    meta = {'title_1': run, 'title_2': 'icecube', 'y_label': metric.replace('opponent_', ''), 'x_label': bin_in}

fig = compare_ic_histogram(
    clip_y,
    resolution_y,
    H,
    x_labels,
    meta
)


st.plotly_chart(fig, use_container_width=True)

performances_df_1 = pd.read_parquet(str(runs_path.joinpath(run_1).joinpath('performance_energy_binned_dataframe_parquet.gzip')), engine='fastparquet')
performances_df_2 = pd.read_parquet(str(runs_path.joinpath(run_2).joinpath('performance_energy_binned_dataframe_parquet.gzip')), engine='fastparquet')

only_metric_string = metric.replace('own', '').replace('error', '').replace('opponent', '').replace('_', '')


trace1 = go.Scatter(
    x=np.linspace(0, 3, 18),
    y=performances_df_1[performances_df_1.metric == only_metric_string].own_performances.values,
    mode='markers',
    name=run_1,
    error_y=dict(
        type='data',
        array=performances_df_1[performances_df_1.metric == only_metric_string].own_sigmas.values,
        visible=True
    ),
    xaxis='x1'
)

trace2 = go.Scatter(
    x=np.linspace(0, 3, 18),
    y=performances_df_2[performances_df_2.metric == only_metric_string].own_performances.values,
    mode='markers',
    name=run_2,
    error_y=dict(
        type='data',
        array=performances_df_2[performances_df_2.metric == only_metric_string].own_sigmas.values,
        visible=True
    ),
    xaxis='x1'
)

trace3 = go.Histogram(
    x=df_list[0].energy,
    xbins=dict(
        start=0,
        end=3,
        size=0.16666
    ),
    opacity=0.2,
    marker_color='grey',
    xaxis='x1',
    yaxis='y2',
    name='test data'
)

trace4 = go.Histogram(
    x=df_list[0].own_azimuth_error,
    xaxis='x2',
    yaxis='y3',
    name='error'
)


data = [trace1, trace2, trace3, trace4]

layout = go.Layout(
    xaxis=dict(
        title='Energy',
        domain=[0, 0.6]
    ),
    xaxis2=dict(
        title='blah',
        domain=[0.7, 1]
    ),
    yaxis=dict(
        title='yaxis title',
    ),
    yaxis2=dict(
        title='yaxis2 title',
        type='log',
        overlaying='y',
        side='right'
    ),
    yaxis3=dict(
        title='test',
        side='right',
        anchor='x2'
    )
)

fig = go.Figure(data=data, layout=layout)

fig.update_layout(
    legend=dict(
        x=-.1,
        y=1.2,
        orientation='h'
    )
)

st.plotly_chart(fig, use_container_width=True)

st.write(df_list[0].head())
