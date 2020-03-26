import streamlit as st
from pathlib import Path
import numpy as np
import numexpr as ne
import pandas as pd
import tables
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from powershovel_sql import get_event_in_energy_range

ne.set_num_threads(1)


def find_min_max_percentiles(df, metric, percentiles):
    percentile_min = df[metric].quantile(percentiles[0])
    percentile_max = df[metric].quantile(percentiles[1])
    return [percentile_min, percentile_max]


def read_geom_file(geom_file):
    drop_cols = ['pmt', 'om', 'string', 'area', 'omtype', 'ux', 'uy', 'uz']
    geom = pd.read_hdf(geom_file, key='pmt_geom')
    geom_clean = geom.loc[geom.omtype == 20].copy()
    geom_clean.drop(drop_cols, axis=1, inplace=True)
    return geom_clean


def make_2d_histogram(x, y, clip_values, bins):
    H, xedges, yedges = np.histogram2d(
        x=np.clip(x.values, clip_values[0], clip_values[1]),
        y=y.values,
        bins=bins
    )
    return H, xedges, yedges


def convert_to_cartesian(azimuth, zenith):
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.array([x, y, z])


def direction_vectors(direction, position, scale):
    x = [position[0, 0] - direction[0, 0] * scale, direction[0, 0] * scale + position[0, 0]]
    y = [position[0, 1] - direction[0, 1] * scale, direction[0, 1] * scale + position[0, 1]]
    z = [position[0, 2] - direction[0, 2] * scale, direction[0, 2] * scale + position[0, 2]]
    return np.array([x, y, z])


def calculate_energy_bins(comparison_df):
    no_of_bins = 18
    comparison_df['energy_binned'] = pd.cut(
        comparison_df.true_energy,
        no_of_bins
    )
    bin_counts = [len(comparison_df[comparison_df.energy_binned == ibin]) for ibin in np.sort(comparison_df.energy_binned.unique())]
    bin_mids = [ibin.mid for ibin in np.sort(comparison_df.energy_binned.unique())]
    bin_widths = [ibin.length for ibin in np.sort(comparison_df.energy_binned.unique())]
    bin_half_widths = [ibin.length / 2 for ibin in np.sort(comparison_df.energy_binned.unique())]
    comparison_df.energy_binned = comparison_df.energy_binned.astype(str)
    bins = comparison_df.energy_binned.unique()
    bins = np.sort(bins)
    return bins, bin_counts, bin_mids, bin_widths, bin_half_widths


resolution_x = 200
resolution_y = 200
percentiles = [0.00, 1]

DATABASE = Path().home().joinpath('sqlite/test_set.db')
RUNS_PATH = Path().home().joinpath('runs')
AVAILABLE_RUNS = [folder.name for folder in RUNS_PATH.iterdir() if folder.is_dir()]
OWN_ERROR_DF_NAME = 'own_error_dataframe_parquet.gzip'
OWN_PERFORMANCE_DF_NAME = 'own_performance_energy_binned_dataframe_parquet.gzip'
OWN_PREDICTION_DF_NAME = 'prediction_dataframe_parquet.gzip'
COMMON_COLUMNS = ['file_number', 'true_energy', 'event_length']
GEOM_FILE = Path().home().joinpath('repos/CubeFlow/powershovel/GeoCalibDetectorStatus_ICUpgrade.v53.mixed.V0.hdf5')
geom_df = read_geom_file(str(GEOM_FILE))

SELECTED_RUN = st.sidebar.selectbox(
    'Run',
    options=AVAILABLE_RUNS,
    index=0
)

AVAILABLE_RUNS.remove(SELECTED_RUN)

SELECTED_COMPARISON = st.sidebar.selectbox(
    'Comparison',
    options=['IceCube'] + AVAILABLE_RUNS,
    index=1
)

SELECTED_RUN_ERRORS = pd.read_parquet(
    str(RUNS_PATH.joinpath(SELECTED_RUN).joinpath(OWN_ERROR_DF_NAME))
)
SELECTED_RUN_PERFORMANCE = pd.read_parquet(
    str(RUNS_PATH.joinpath(SELECTED_RUN).joinpath(OWN_PERFORMANCE_DF_NAME))
)
SELECTED_RUN_PREDICTIONS = pd.read_parquet(
    str(RUNS_PATH.joinpath(SELECTED_RUN).joinpath(OWN_PREDICTION_DF_NAME))
)

AVAILABLE_METRICS = [x for x in list(SELECTED_RUN_ERRORS.columns.values) if x not in COMMON_COLUMNS]

if SELECTED_COMPARISON == 'IceCube':
    COMPARISON_ERROR_DF_NAME = 'opponent_error_dataframe_parquet.gzip'
    COMPARISON_PERFORMANCE_DF_NAME = 'opponent_performance_energy_binned_dataframe_parquet.gzip'
    COMPARISON_PREDICTION_DF_NAME = 'prediction_dataframe_parquet.gzip'

    COMPARISON_ERRORS = pd.read_parquet(
        str(RUNS_PATH.joinpath(SELECTED_RUN).joinpath(COMPARISON_ERROR_DF_NAME))
    )
    COMPARISON_PERFORMANCE = pd.read_parquet(
        str(RUNS_PATH.joinpath(SELECTED_RUN).joinpath(COMPARISON_PERFORMANCE_DF_NAME))
    )
    COMPARISON_PREDICTIONS = pd.read_parquet(
        str(RUNS_PATH.joinpath(SELECTED_RUN).joinpath(COMPARISON_PREDICTION_DF_NAME))
    )
else:
    COMPARISON_ERRORS = pd.read_parquet(
        str(RUNS_PATH.joinpath(SELECTED_COMPARISON).joinpath(OWN_ERROR_DF_NAME))
    )
    COMPARISON_PERFORMANCE = pd.read_parquet(
        str(RUNS_PATH.joinpath(SELECTED_COMPARISON).joinpath(OWN_PERFORMANCE_DF_NAME))
    )
    COMPARISON_PREDICTIONS = pd.read_parquet(
        str(RUNS_PATH.joinpath(SELECTED_COMPARISON).joinpath(OWN_PREDICTION_DF_NAME))
    )

VIEW_TYPE = st.sidebar.selectbox(
    'View...',
    options=['Events', 'Comparisons'],
    index=0
)

OWN_BINS, OWN_BIN_COUNTS, OWN_BIN_MIDS, OWN_BIN_WIDTHS, OWN_BIN_HALF_WIDTHS = calculate_energy_bins(SELECTED_RUN_ERRORS)
OPPONENT_BINS, OPPONENT_BIN_COUNTS, OPPONENT_BIN_MIDS, OPPONENT_BIN_WIDTHS, OPPONENT_BIN_HALF_WIDTHS = calculate_energy_bins(COMPARISON_ERRORS)
BIN = st.sidebar.selectbox(
    'Bin',
    options=OWN_BINS,
    index=0
)
BIN_INDEX = list(OWN_BINS).index(BIN)

HISTOGRAM_COLORS = ['grey'] * 18
HISTOGRAM_COLORS[BIN_INDEX] = 'red'

if VIEW_TYPE == 'Events':
    ENERGY_RANGE = re.findall('\d+\.\d+', BIN)
    ENERGY_RANGE = [float(number) for number in ENERGY_RANGE]
    sequential_df, scalar_df = get_event_in_energy_range(DATABASE, ENERGY_RANGE)
    # sequential_df.dom_charge = np.clip(sequential_df.dom_charge.values, 0, 20)
    # sequential_df.dom_charge = sequential_df.dom_charge.apply(lambda x: (x - min(sequential_df.dom_charge.values)) / (20) * 500)

    EVENT = scalar_df.event.values[0]
    # EVENT = 10773757

    OWN_DIRS = SELECTED_RUN_PREDICTIONS[SELECTED_RUN_PREDICTIONS.file_number == str(EVENT)][['own_primary_direction_x', 'own_primary_direction_y', 'own_primary_direction_z']].values
    OWN_POS = SELECTED_RUN_PREDICTIONS[SELECTED_RUN_PREDICTIONS.file_number == str(EVENT)][['own_primary_position_x', 'own_primary_position_y', 'own_primary_position_z']].values

    TRUE_DIRS = SELECTED_RUN_PREDICTIONS[SELECTED_RUN_PREDICTIONS.file_number == str(EVENT)][['true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z']].values
    TRUE_POS = SELECTED_RUN_PREDICTIONS[SELECTED_RUN_PREDICTIONS.file_number == str(EVENT)][['true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z']].values

    OWN_VECTOR = direction_vectors(OWN_DIRS, OWN_POS, 1000)
    TRUE_VECTOR = direction_vectors(TRUE_DIRS, TRUE_POS, 1000)

    if SELECTED_COMPARISON == 'IceCube':
        OPP_ANGLES = COMPARISON_PREDICTIONS[COMPARISON_PREDICTIONS.file_number == str(EVENT)][['opponent_azimuth', 'opponent_zenith']]
        OPP_DIRS = convert_to_cartesian(OPP_ANGLES[0], OPP_ANGLES[1])
        OPP_POS = COMPARISON_PREDICTIONS[COMPARISON_PREDICTIONS.file_number == str(EVENT)][['opponent_x', 'opponent_y', 'opponent_z']]
    else:
        OPP_DIRS = COMPARISON_PREDICTIONS[COMPARISON_PREDICTIONS.file_number == str(EVENT)][['own_primary_direction_x', 'own_primary_direction_y', 'own_primary_direction_z']].values
        OPP_POS = COMPARISON_PREDICTIONS[COMPARISON_PREDICTIONS.file_number == str(EVENT)][['own_primary_position_x', 'own_primary_position_y', 'own_primary_position_z']].values

    OPP_VECTOR = direction_vectors(OPP_DIRS, OPP_POS, 1000)

    trace1 = go.Scatter3d(
        x=sequential_df.dom_x.values,
        y=sequential_df.dom_y.values,
        z=sequential_df.dom_z.values,
        mode='markers',
        name='pulses',
        marker=dict(
            size=10,
            color=sequential_df.dom_time.values,
            colorscale='Oranges',
            opacity=0.8
        )
    )

    trace2 = go.Scatter3d(
        x=geom_df.x.values,
        y=geom_df.y.values,
        z=geom_df.z.values,
        mode='markers',
        name='DOMs',
        marker=dict(
            size=1,
            color='black',
            opacity=0.5
        )
    )

    trace3 = go.Scatter3d(
        x=OWN_VECTOR[0, :],
        y=OWN_VECTOR[1, :],
        z=OWN_VECTOR[2, :],
        mode='lines',
        name=SELECTED_RUN + ' vector'
    )

    trace4 = go.Scatter3d(
        x=OPP_VECTOR[0, :],
        y=OPP_VECTOR[1, :],
        z=OPP_VECTOR[2, :],
        mode='lines',
        name=SELECTED_COMPARISON + ' vector'
    )

    trace5 = go.Scatter3d(
        x=TRUE_VECTOR[0, :],
        y=TRUE_VECTOR[1, :],
        z=TRUE_VECTOR[2, :],
        mode='lines',
        name='true vector'
    )

    trace6 = go.Scatter3d(
        x=[OWN_POS[0, 0]],
        y=[OWN_POS[0, 1]],
        z=[OWN_POS[0, 2]],
        mode='markers',
        name=SELECTED_RUN + ' vertex'
    )

    trace7 = go.Scatter3d(
        x=[OPP_POS[0, 0]],
        y=[OPP_POS[0, 1]],
        z=[OPP_POS[0, 2]],
        mode='markers',
        name=SELECTED_COMPARISON + ' vertex'
    )

    trace8 = go.Scatter3d(
        x=[TRUE_POS[0, 0]],
        y=[TRUE_POS[0, 1]],
        z=[TRUE_POS[0, 2]],
        mode='markers',
        name='true vertex'
    )

    data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]
    fig = go.Figure(data=data)
    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[-800, 800]),
            yaxis = dict(range=[-800, 800]),
            zaxis = dict(range=[-800, 800])
        ),
        height=800,
        scene_aspectmode='cube',
        legend=dict(
            x=-.1,
            y=1.2,
            orientation='h'
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)

elif VIEW_TYPE == 'Comparisons':
    SELECTED_METRIC = st.sidebar.selectbox(
        'Metric',
        options=AVAILABLE_METRICS
    )

    clips_1 = find_min_max_percentiles(SELECTED_RUN_ERRORS, SELECTED_METRIC, percentiles)
    clips_2 = find_min_max_percentiles(COMPARISON_ERRORS, SELECTED_METRIC, percentiles)
    clips = clips_1 + clips_2
    clip_y = [min(clips), max(clips)]

    H1, xedges1, yedges1 = make_2d_histogram(
        SELECTED_RUN_ERRORS[SELECTED_METRIC],
        SELECTED_RUN_ERRORS.true_energy,
        clip_y,
        [resolution_y, resolution_x]
    )

    H2, xedges2, yedges2 = make_2d_histogram(
        COMPARISON_ERRORS[SELECTED_METRIC],
        COMPARISON_ERRORS.true_energy,
        clip_y,
        [resolution_y, resolution_x]
    )

    max_z_values = []
    max_z_values.append(np.amax(H1))
    max_z_values.append(np.amax(H2))
    max_z_value = max(max_z_values)

    trace1 = go.Heatmap(
        x=yedges1,
        y=xedges1,
        z=H1,
        zmin=0,
        zmax=max_z_value,
        showscale=True,
        xaxis='x1',
        yaxis='y1',
        coloraxis = 'coloraxis'
    )

    trace2 = go.Heatmap(
        x=yedges2,
        y=xedges2,
        z=H2,
        zmin=0,
        zmax=max_z_value,
        showscale=False,
        xaxis='x2',
        yaxis='y1',
        coloraxis = 'coloraxis'
    )

    trace3 = go.Scatter(
        x=OWN_BIN_MIDS,
        y=[SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.16) for ibin in OWN_BINS],
        mode='lines',
        line=dict(
            color='Red',
            dash='dot'
        ),
        xaxis='x1',
        yaxis='y1',
        showlegend=False
    )
    trace4 = go.Scatter(
        x=OWN_BIN_MIDS,
        y=[SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.5) for ibin in OWN_BINS],
        mode='lines',
        line=dict(
            color='Red'
        ),
        xaxis='x1',
        yaxis='y1',
        name=SELECTED_RUN,
        showlegend=True
    )
    trace5 = go.Scatter(
        x=OWN_BIN_MIDS,
        y=[SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.84) for ibin in OWN_BINS],
        mode='lines',
        line=dict(
            color='Red',
            dash='dash'
        ),
        xaxis='x1',
        yaxis='y1',
        showlegend=False
    )
    trace6 = go.Scatter(
        x=OPPONENT_BIN_MIDS,
        y=[COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.16) for ibin in OPPONENT_BINS],
        mode='lines',
        line=dict(
            color='Green',
            dash='dot'
        ),
        xaxis='x1',
        yaxis='y1',
        showlegend=False
    )
    trace7 = go.Scatter(
        x=OPPONENT_BIN_MIDS,
        y=[COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.5) for ibin in OPPONENT_BINS],
        mode='lines',
        line=dict(
            color='Green'
        ),
        xaxis='x1',
        yaxis='y1',
        showlegend=True,
        name=SELECTED_COMPARISON
    )
    trace8 = go.Scatter(
        x=OPPONENT_BIN_MIDS,
        y=[COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.84) for ibin in OPPONENT_BINS],
        mode='lines',
        line=dict(
            color='Green',
            dash='dash'
        ),
        xaxis='x1',
        yaxis='y1',
        showlegend=False
    )

    trace9 = go.Scatter(
        x=OWN_BIN_MIDS,
        y=[SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.16) for ibin in OWN_BINS],
        mode='lines',
        line=dict(
            color='Red',
            dash='dot'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace10 = go.Scatter(
        x=OWN_BIN_MIDS,
        y=[SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.5) for ibin in OWN_BINS],
        mode='lines',
        line=dict(
            color='Red'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace11 = go.Scatter(
        x=OWN_BIN_MIDS,
        y=[SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.84) for ibin in OWN_BINS],
        mode='lines',
        line=dict(
            color='Red',
            dash='dash'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace12 = go.Scatter(
        x=OPPONENT_BIN_MIDS,
        y=[COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.16) for ibin in OPPONENT_BINS],
        mode='lines',
        line=dict(
            color='Green',
            dash='dot'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace13 = go.Scatter(
        x=OPPONENT_BIN_MIDS,
        y=[COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.5) for ibin in OPPONENT_BINS],
        mode='lines',
        line=dict(
            color='Green'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace14 = go.Scatter(
        x=OPPONENT_BIN_MIDS,
        y=[COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == ibin][SELECTED_METRIC].quantile(0.84) for ibin in OPPONENT_BINS],
        mode='lines',
        line=dict(
            color='Green',
            dash='dash'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )

    data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13, trace14]

    layout = go.Layout(
        title='Test',
        coloraxis=dict(
            colorscale='Oranges'
        ),
        xaxis=dict(
            title='Energy',
            domain=[0, 0.48]
        ),
        xaxis2=dict(
            title='Energy',
            domain=[0.52, 1]
        ),
        yaxis=dict(
            title='Error',
            anchor='x1'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        legend=dict(
            x=-.1,
            y=1.2,
            orientation='h'
        ),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

    trace1 = go.Bar(
        x=OWN_BIN_MIDS,
        y=OWN_BIN_COUNTS,
        width=OWN_BIN_WIDTHS,
        opacity=0.2,
        marker_color=HISTOGRAM_COLORS,
        xaxis='x1',
        yaxis='y2',
        name='test data',
        showlegend=False
    )

    trace2 = go.Scatter(
        x=OWN_BIN_MIDS,
        y=SELECTED_RUN_PERFORMANCE[SELECTED_METRIC].values,
        mode='markers',
        name=SELECTED_RUN,
        showlegend=False,
        marker=dict(
            color='Red'
        ),
        error_y=dict(
            type='data',
            array=SELECTED_RUN_PERFORMANCE[SELECTED_METRIC + '_sigma'].values,
            visible=True
        ),
        error_x=dict(
            type='data',
            array=OWN_BIN_HALF_WIDTHS,
            visible=True
        ),
        xaxis='x1'
    )

    trace3 = go.Scatter(
        x=OPPONENT_BIN_MIDS,
        y=COMPARISON_PERFORMANCE[SELECTED_METRIC].values,
        mode='markers',
        name=SELECTED_COMPARISON,
        showlegend=False,
        marker=dict(
            color='Green'
        ),
        error_y=dict(
            type='data',
            array=COMPARISON_PERFORMANCE[SELECTED_METRIC + '_sigma'].values,
            visible=True
        ),
        error_x=dict(
            type='data',
            array=OPPONENT_BIN_HALF_WIDTHS,
            visible=True
        ),
        xaxis='x1'
        )

    trace4 = go.Histogram(
        x=SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == BIN][SELECTED_METRIC],
        xaxis='x2',
        yaxis='y3',
        name=SELECTED_RUN + ' error',
        histnorm='probability',
        opacity=0.5,
        marker_color='Red',
        showlegend=False
    )
    trace5 = go.Histogram(
        x=COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == BIN][SELECTED_METRIC],
        xaxis='x2',
        yaxis='y3',
        name=SELECTED_COMPARISON + ' error',
        histnorm='probability',
        opacity=0.5,
        marker_color='Green',
        showlegend=False
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        xaxis=dict(
            title='Energy',
            domain=[0, 0.6]
        ),
        xaxis2=dict(
            title='Error',
            domain=[0.7, 1]
        ),
        yaxis=dict(
            title='Resolution',
        ),
        yaxis2=dict(
            title='Events',
            type='log',
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='Density',
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
        ),
        height=500
    )

    fig.add_shape(
                type='line',
                xref='x2',
                yref='paper',
                x0=SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == BIN][SELECTED_METRIC].quantile(0.16),
                y0=0,
                x1=SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == BIN][SELECTED_METRIC].quantile(0.16),
                y1=1,
                name=SELECTED_RUN + ' 16\%',
                line=dict(
                    color='Red',
                    width=1,
                    dash='dot'
                ),
            )
    fig.add_shape(
                type='line',
                xref='x2',
                yref='paper',
                x0=SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == BIN][SELECTED_METRIC].quantile(0.84),
                y0=0,
                x1=SELECTED_RUN_ERRORS[SELECTED_RUN_ERRORS.energy_binned == BIN][SELECTED_METRIC].quantile(0.84),
                y1=1,
                name=SELECTED_RUN + ' 84\%',
                line=dict(
                    color='Red',
                    width=1,
                    dash='dot'
                ),
            )

    fig.add_shape(
                type='line',
                xref='x2',
                yref='paper',
                x0=COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == BIN][SELECTED_METRIC].quantile(0.16),
                y0=0,
                x1=COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == BIN][SELECTED_METRIC].quantile(0.16),
                y1=1,
                name=SELECTED_COMPARISON + ' 16\%',
                line=dict(
                    color='Green',
                    width=1,
                    dash='dot'
                ),
            )
    fig.add_shape(
                type='line',
                xref='x2',
                yref='paper',
                x0=COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == BIN][SELECTED_METRIC].quantile(0.84),
                y0=0,
                x1=COMPARISON_ERRORS[COMPARISON_ERRORS.energy_binned == BIN][SELECTED_METRIC].quantile(0.84),
                y1=1,
                name=SELECTED_COMPARISON + ' 84\%',
                line=dict(
                    color='Green',
                    width=1,
                    dash='dot'
                ),
            )

    st.plotly_chart(fig, use_container_width=True)
