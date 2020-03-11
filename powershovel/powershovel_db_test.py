import streamlit as st
from pathlib import Path
import numpy as np

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
db = DbHelper(sql_file)

clip_x_values = [0, 4]
clip_y_quantiles = [0.01, 0.99]
resolution = [200, 200]
comparison_quantiles = [0.16, 0.84]

selected_run = st.sidebar.selectbox(
    'Run',
    db.runs,
    index=1
)

db.runs.remove(selected_run)
selected_run_metadata = db.get_run_metadata(selected_run)

# db.drop_table('tangerine-wildcat')

selected_comparison = st.sidebar.selectbox(
    'Comparison',
    ['IceCube'] + db.runs
)

selected_bin_type = st.sidebar.radio(
    'Bin in',
    ['true_primary_energy', 'event_length']
)
selected_bin_type_bin_name = 'true_primary_energy_bin' if selected_bin_type == 'true_primary_energy' else 'event_length_bin'

selected_bin = st.sidebar.selectbox(
    'Bin',
    np.sort(selected_run_metadata[selected_bin_type_bin_name])
)
selected_bin_index = selected_run_metadata[selected_bin_type_bin_name].index(selected_bin)

selected_metric = st.sidebar.selectbox(
    'Metric',
    selected_run_metadata['metric_names']
)

own_errors, opponent_errors = get_errors(db, selected_metric, selected_run, selected_comparison, selected_bin_type, selected_bin_type_bin_name)

own_widths, opponent_widths = calculate_resolution_widths(
    own_errors,
    opponent_errors,
    selected_metric,
    selected_run_metadata[selected_bin_type_bin_name],
    selected_bin_type_bin_name,
    comparison_quantiles
)

H1, H2, max_z_value = make_2d_histograms(
    [own_errors, opponent_errors],
    selected_metric,
    selected_bin_type,
    resolution,
    clip_y_quantiles
)

fig = plotly_2d_histograms(
    H1,
    H2,
    own_errors,
    opponent_errors,
    max_z_value,
    selected_metric,
    selected_bin_type_bin_name,
    selected_bin_type,
    selected_run,
    selected_comparison
)
st.plotly_chart(fig, use_container_width=True)

fig = plotly_error_comparison(
    own_widths,
    opponent_widths,
    own_errors,
    opponent_errors,
    selected_metric,
    selected_bin_type_bin_name,
    selected_bin_type,
    selected_bin,
    selected_bin_index,
    selected_run,
    selected_comparison
)
st.plotly_chart(fig, use_container_width=True)