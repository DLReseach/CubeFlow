import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from powershovel_sql import DbHelper
from powershovel_utils import make_2d_histograms
from powershovel_utils import calculate_resolution_widths
from powershovel_plotting import plotly_2d_histograms
from powershovel_plotting import plotly_error_comparison
from powershovel_plotting import plotly_event


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()


cubeflow_data_path = Path().home().joinpath('CubeFlowData')
geom_path = cubeflow_data_path.joinpath('geom')
dbs_path = cubeflow_data_path.joinpath('dbs')
test_set_file = dbs_path.joinpath('test_set.db')

geom = pd.read_parquet(
    geom_path.joinpath('GeoCalibDetectorStatus_ICUpgrade.v53.mixed.V0.gzip')
)
drop_cols = ['pmt', 'om', 'string', 'area', 'omtype', 'ux', 'uy', 'uz']
geom_clean = geom.loc[geom.omtype == 20].copy()
geom_clean.drop(drop_cols, axis=1, inplace=True)

mask_names = [folder for folder in dbs_path.iterdir() if folder.is_dir()]

st.sidebar.title('Powershovel 2.0')

selected_masks = st.sidebar.selectbox(
    'Masks',
    mask_names,
    index=0,
    format_func=lambda x: x.name
)

histograms = dbs_path.joinpath(selected_masks.name).joinpath('histograms.pkl')

with open(histograms, 'rb') as f:
    histogram_data = pickle.load(f)
    runs_data = histogram_data['runs']
    meta_data = histogram_data['meta']

db = DbHelper(
    dbs_path.joinpath(selected_masks).joinpath('predictions.db'),
    test_set_file
)

runs_list = list(runs_data.keys())
runs_list.remove('retro_crs_prefit')

selected_run = st.sidebar.selectbox(
    'Run',
    runs_list,
    index=0
)

selected_run_data = runs_data[selected_run]

runs_list.remove(selected_run)

selected_comparison = st.sidebar.selectbox(
    'Comparison',
    ['retro_crs_prefit'] + runs_list
)

selected_comparison_data = runs_data[selected_comparison]

bins = list(range(len(meta_data['1d_histogram']['bins'])))

selected_bin = st.sidebar.slider(
    'Bin',
    min_value=bins[0],
    max_value=bins[-1],
    step=1
)

selected_bin_index = bins.index(selected_bin)

random_event_from_bin = np.random.choice(meta_data['events'][selected_bin_index])
predictions, scalars, sequential, comparison, meta = db.get_predictions(selected_run, selected_comparison, random_event_from_bin)

selected_metric = st.sidebar.selectbox(
    'Metric',
    list(selected_run_data.keys())
)

show_cleaned_pulses = st.sidebar.radio(
    'Show cleaned pulses',
    ['Yes', 'No'],
    index=1
)

if show_cleaned_pulses == 'Yes':
    sequential = sequential[sequential['SRTInIcePulses'] == 1]

# st.markdown('## 2D histograms')
# 
# fig = plotly_2d_histograms(
#     selected_run_data,
#     selected_comparison_data,
#     selected_metric,
#     selected_bin_type,
#     selected_run,
#     selected_comparison,
#     meta_data
# )
# st.plotly_chart(fig, use_container_width=True)
# 
st.markdown('## Comparisons and errors')

fig = plotly_error_comparison(
    selected_run_data,
    selected_comparison_data,
    selected_metric,
    selected_run,
    selected_comparison,
    meta_data,
    selected_bin
)

st.plotly_chart(fig, use_container_width=True)
# 
# st.markdown('## Random event from bin')
# 
# st.write(comparison['predicted_primary_energy'])
# 
# df = pd.DataFrame(
#     [
#         [
#             10**scalars['true_primary_energy'].values[0],
#             2,
#             3,
#             scalars['true_primary_time'].values[0],
#             scalars['true_primary_position_x'].values[0],
#             scalars['true_primary_position_y'].values[0],
#             scalars['true_primary_position_z'].values[0],
#             len(sequential)
#         ],
#         [
#             10**predictions['predicted_primary_energy'].values[0],
#             2,
#             3,
#             predictions['predicted_primary_time'].values[0],
#             predictions['predicted_primary_position_x'].values[0],
#             predictions['predicted_primary_position_y'].values[0],
#             predictions['predicted_primary_position_z'].values[0],
#             'N/A'
#         ],
#         [
#             scalars['retro_crs_prefit_energy'].values[0] if selected_comparison == 'retro_crs_prefit' else 10**comparison['predicted_primary_energy'].values[0],
#             2,
#             3,
#             scalars['retro_crs_prefit_time'].values[0] if selected_comparison == 'retro_crs_prefit' else comparison['predicted_primary_time'],
#             scalars['retro_crs_prefit_x'].values[0] if selected_comparison == 'retro_crs_prefit' else comparison['predicted_primary_position_x'],
#             scalars['retro_crs_prefit_y'].values[0] if selected_comparison == 'retro_crs_prefit' else comparison['predicted_primary_position_y'],
#             scalars['retro_crs_prefit_z'].values[0] if selected_comparison == 'retro_crs_prefit' else comparison['predicted_primary_position_z'],
#             'N/A'
#         ],
#     ],
#     columns=[
#         'Energy',
#         'Azimuth',
#         'Zenith',
#         'Time',
#         'Vertex x',
#         'Vertex y',
#         'Vertex z',
#         'Event length'
#     ],
#     index=[
#         'Truth',
#         selected_run,
#         selected_comparison
#     ]
# )
# 
# st.table(df)
# 
# fig = plotly_event(predictions, scalars, sequential, comparison, geom_clean, selected_run, selected_comparison)
# st.plotly_chart(fig, use_container_width=True)
