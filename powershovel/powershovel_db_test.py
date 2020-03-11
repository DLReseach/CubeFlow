import streamlit as st
from pathlib import Path
import numpy as np
import shelve
from datetime import datetime

from powershovel_sql import DbHelper
from powershovel_utils import make_2d_histograms
from powershovel_utils import calculate_resolution_widths
from powershovel_plotting import plotly_2d_histograms
from powershovel_plotting import plotly_error_comparison


# @st.cache
def get_errors(db, selected_metric, selected_run, selected_comparison, selected_bin_type, selected_bin_type_bin_name):
    own_errors, opponent_errors = db.get_metric_errors(selected_metric, selected_run, selected_comparison, selected_bin_type, selected_bin_type_bin_name)
    return own_errors, opponent_errors


sql_file = Path('/mnt/c/Users/MadsEhrhorn/Downloads/test_set_backup.db')
pickle_file = Path().home().joinpath('runs').joinpath('powershovel')
db = DbHelper(sql_file)

with shelve.open(str(pickle_file), 'r') as f:
    runs_data = f['datasets']
    meta_data = f['meta']

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

selected_bin_type = st.sidebar.radio(
    'Bin in',
    ['energy', 'event_length']
)

bins = list(range(len(meta_data[selected_bin_type]['bins'])))

selected_bin = st.sidebar.slider(
    'Bin',
    min_value=bins[0],
    max_value=bins[-1],
    step=1
)

selected_bin_index = bins.index(selected_bin)

random_event_from_bin = np.random.choice(meta_data['events'][selected_bin_type][selected_bin_index])
# st.write(random_event_from_bin)
predictions = db.get_predictions(selected_run, random_event_from_bin)
# st.write(predictions.head())

selected_metric = st.sidebar.selectbox(
    'Metric',
    list(selected_run_data.keys())
)

fig = plotly_2d_histograms(
    selected_run_data,
    selected_comparison_data,
    selected_metric,
    selected_bin_type,
    selected_run,
    selected_comparison,
    meta_data
)
st.plotly_chart(fig, use_container_width=True)

fig = plotly_error_comparison(
    selected_run_data,
    selected_comparison_data,
    selected_metric,
    selected_bin_type,
    selected_run,
    selected_comparison,
    meta_data,
    selected_bin
)

st.plotly_chart(fig, use_container_width=True)
