"""streamlit dashboard for IceCube event viewing.

Install streamlit with `pip install streamlit`.

How to run: `streamlit run powershovel.py`

Created by Mads Ehrhorn 19/10/2019.
"""
import streamlit as st
import pandas as pd
import h5py as h5
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from operator import itemgetter

from utils.utils import get_project_root

# Columns to drop from geom frame


# @st.cache
def read_geom_file(geom_file):
    """Read the geometry file, output as dictionary of DataFrames.

    Function also deletes columns that aren't used.

    Args:
        geom_file (str): HDF5 geometry file path.

    Returns:
        geom (pandas.DataFrame): Dictionary containing geometry info.

    """
    drop_cols = ['pmt', 'om', 'string', 'area', 'omtype', 'ux', 'uy', 'uz']
    geom = pd.read_hdf(geom_file, key='pmt_geom')
    geom_clean = geom.loc[geom.omtype == 20].copy()
    geom_clean.drop(drop_cols, axis=1, inplace=True)
    return geom_clean


# @st.cache
def meta_data(events_file):
    """Calculate meta data from event file.

    Args:
        None

    Returns:
        no_of_doms (list): Number of DOMs involved in each event.
        energy (pandas.Series): Energy for each event.
        events (list): Events to include.
        integrated_charge (list): Total charge involved in each event.

    """
    with h5.File(events_file, 'r') as f:
        all_integrated_charge = f['raw/dom_charge'][:]
        integrated_charge = list(
            map(
                sum,
                all_integrated_charge
            )
        )
        max_charge = max(np.concatenate(all_integrated_charge).ravel())
        min_charge = min(np.concatenate(all_integrated_charge).ravel())
        no_of_doms = [len(x) for x in all_integrated_charge]
        all_energy = f['raw/true_primary_energy'][:]
        toi_eval_ratios = f['raw/toi_evalratio'][:]
        toi_ratios = pd.Series(toi_eval_ratios)
        energy = pd.Series(all_energy)
        out = pd.DataFrame(
            {
                'toi_ratio': toi_ratios,
                'energy': energy,
                'integrated_charge': integrated_charge,
                'no_of_doms': no_of_doms,
                'max_charge': max_charge
            }
        )
    return out


def read_event(events_file, event_no, clean_mask):
    """Read an hdf5 file and output activations and truth as DataFrame.

    Args:
        events_file (str): HDF5 events file path.
        event_no (int): Event number.
        truth_cols (list): List of truth variable columns names.

    Returns:
        activations (pandas.DataFrame): DataFrame containing event activations.
        truth (pandas.Series): Series containing truth variables.

    """
    with h5.File(events_file, 'r') as f:
        activations = {}
        truth = {}
        mask = f['masks/' + clean_mask][event_no]
        for array in f['raw'].__iter__():
            data = f['raw/' + array][event_no]
            if array == 'dom_time':
                activations[array] = f['transform1/' + array][event_no][mask]
            elif type(data) == np.ndarray:
                activations[array] = f['raw/' + array][event_no][mask]
            else:
                truth[array] = []
                truth[array].append(f['raw/' + array][event_no])
        activations = pd.DataFrame.from_dict(activations)
        truth = pd.DataFrame.from_dict(truth)
    return activations, truth


def direction_vectors(truth, scale):
    """Calculate direction vectors.

    Args:
        truth (pandas.Series): Series containing truth variables.
        key1 (str): String with name of first key.
        key2 (str): String with name of second key.

        Returns:
            x (list): List with two x points.
            y (list): List with two y points.
            z (list): List with two z points.

    """
    output = {'true': [], 'toi': []}
    keys = []
    keys.append(['true_primary_position', 'true_primary_direction'])
    keys.append(['toi_point_on_line', 'toi_direction'])
    keys_iter = iter(keys)
    for output_type in output:
        key = next(keys_iter)
        pol_x = truth[key[0] + '_x'].values[0]
        pol_y = truth[key[0] + '_y'].values[0]
        pol_z = truth[key[0] + '_z'].values[0]
        dir_x = truth[key[1] + '_x'].values[0] * scale
        dir_y = truth[key[1] + '_y'].values[0] * scale
        dir_z = truth[key[1] + '_z'].values[0] * scale
        x = [pol_x - dir_x, dir_x + pol_x]
        y = [pol_y - dir_y, dir_y + pol_y]
        z = [pol_z - dir_z, dir_z + pol_z]
        output[output_type] = (x, y, z)
    output['entry'] = (
        truth['true_primary_position_x'].values[0],
        truth['true_primary_position_y'].values[0],
        truth['true_primary_position_z'].values[0]
    )
    return output


def direction_vectors2(truth, scale):
    """Calculate direction vectors.

    Args:
        truth (pandas.Series): Series containing truth variables.
        key1 (str): String with name of first key.
        key2 (str): String with name of second key.

        Returns:
            x (list): List with two x points.
            y (list): List with two y points.
            z (list): List with two z points.

    """
    output = {'true': []}
    keys = []
    keys.append(['true_primary_entry_position', 'true_primary_direction'])
    keys_iter = iter(keys)
    for output_type in output:
        key = next(keys_iter)
        pol_x = truth[key[0] + '_x'].values[0]
        pol_y = truth[key[0] + '_y'].values[0]
        pol_z = truth[key[0] + '_z'].values[0]
        dir_x = truth[key[1] + '_x'].values[0] * scale
        dir_y = truth[key[1] + '_y'].values[0] * scale
        dir_z = truth[key[1] + '_z'].values[0] * scale
        x = [pol_x - dir_x, dir_x + pol_x]
        y = [pol_y - dir_y, dir_y + pol_y]
        z = [pol_z - dir_z, dir_z + pol_z]
        output[output_type] = (x, y, z)
    return output


def create_animation(data, template):
    """Create plotly express object with animation.

    Args:
        data (pandas.DataFrame): DataFrame to plot.

    Returns:
        fig (plotly.ExpressFigure): Plotly Express object.

    """
    fig = px.scatter_3d(
        data,
        x='dom_x',
        y='dom_y',
        z='dom_z',
        animation_frame='bin',
        size='dom_charge',
        template=template,
        range_x=[-axis_lims, axis_lims],
        range_y=[-axis_lims, axis_lims],
        range_z=[-650, 650]
    )
    return fig


def create_static(
    activations,
    geom,
    truth,
    predict,
    template,
    superpose,
    time_range
):
    """Create plotly express object without animation.

    Args:
        data (pandas.DataFrame): DataFrame to plot.

    Returns:
        fig (plotly.ExpressFigure): Plotly Express object.

    """
    # Create plot of activations
    vectors = direction_vectors(truth, 2000)
    fig = px.scatter_3d(
        activations,
        x='dom_x',
        y='dom_y',
        z='dom_z',
        size='dom_charge',       
        color='dom_time',
        template=template,
        range_x=[-axis_lims, axis_lims],
        range_y=[-axis_lims, axis_lims],
        range_z=[-axis_lims, axis_lims],
        color_continuous_scale=px.colors.diverging.RdYlBu,
        range_color=time_range,
        size_max=20
    )
    # Add DOM geometry
    fig.add_scatter3d(
        x=geom.x,
        y=geom.y,
        z=geom.z,
        mode='markers',
        marker={'size': 0.8, 'color': 'black', 'opacity': 0.2},
        name='DOM'
    )
    if 'Truth' in superpose:
        fig.add_scatter3d(
            x=vectors['true'][0],
            y=vectors['true'][1],
            z=vectors['true'][2],
            mode='lines',
            name='Truth',
            marker={'size': 6.0, 'color': 'red'}
        )
        fig.add_scatter3d(
            x=[vectors['entry'][0]],
            y=[vectors['entry'][1]],
            z=[vectors['entry'][2]],
            marker={'color': 'red'},
            mode='markers',
            name='Entry'
        )
        fig.add_trace(
            go.Cone(
                x=[vectors['entry'][0]],
                y=[vectors['entry'][1]],
                z=[vectors['entry'][2]],
                u=100 * truth.true_primary_direction_x,
                v=100 * truth.true_primary_direction_y,
                w=100 * truth.true_primary_direction_z,
                showscale=False
            )
        )

    if 'ToI' in superpose:
        fig.add_scatter3d(
            x=vectors['toi'][0],
            y=vectors['toi'][1],
            z=vectors['toi'][2],
            mode='lines',
            name='ToI',
            marker={'size': 6.0, 'color': 'orange'}
        )
    if 'Predict' in superpose:
        fig.add_scatter3d(
            x=predict['true'][0],
            y=predict['true'][1],
            z=predict['true'][2],
            mode='lines',
            name='Prediction',
            marker={'size': 6.0, 'color': 'orange'}
        )
    return fig


def selection_func(choice):
    """Label selections with energy instead of event number.

    Args:
        input (int): Event number.

    Returns:
        energy_label (str): Stringified energy value.

    """
    label = str(round(10**select_meta.loc[choice], 2))
    return label


# @st.cache
def create_template():
    """Create a template for a plotly plot.

    Args:
        None

    Returns:
        icecube_template (plotly.graph_objects.layout.Template): Template.

    """
    icecube_template = go.layout.Template(
        layout=go.Layout(
            {
                'scene': {
                    'xaxis': {
                        'range': [-axis_lims, axis_lims],
                        'backgroundcolor': 'grey',
                        'showbackground': True,
                        'showgrid': False,
                        'showticklabels': False,
                        'title': 'x'
                    },
                    'yaxis': {
                        'range': [-axis_lims, axis_lims],
                        'backgroundcolor': 'grey',
                        'showbackground': True,
                        'showgrid': False,
                        'showticklabels': False,
                        'title': 'y'
                    },
                    'zaxis': {
                        'range': [-axis_lims, axis_lims],
                        'backgroundcolor': 'grey',
                        'showbackground': True,
                        'showgrid': False,
                        'showticklabels': False,
                        'title': 'z'
                    },
                    'aspectmode': 'cube',
                },
                'legend_orientation': 'h'
            }
        )
    )
    return icecube_template


def file_finder(data_set, particle_type, index=None):
    events_path = files_path.joinpath(data_set)
    events_path = events_path.absolute()
    data_set_files = sorted(
        [f for f in events_path.glob('*' + particle_type + '*.h5')
        if f.is_file() and f.suffix == '.h5']
    )
    if index == None:
        return data_set_files
    else:
        return data_set_files[index]


def h5_groups_reader(data_file, group):
    with h5.File(data_file, 'r') as f:
        groups = list(f[group].keys())
    return groups


def h5_data_reader(data_file, group, idx):
    with h5.File(data_file, 'r') as f:
        if idx == 'all':
            data = f[group][:]
        else:
            data = f[group][idx]
    return data


root = get_project_root()
# Path to HDF5 event file
files_path = root.joinpath('data/')
histograms_path = root.joinpath('powershovel/histograms')

# Path to HDF5 geometry file
geom_file = root.joinpath('powershovel/geom.h5')

# Sidebar title
st.sidebar.markdown('# *Powershovel^{TM}*')
st.sidebar.markdown('## IceCube event viewer')

show_dists = st.sidebar.selectbox(
    'View type',
    options=['Events', 'Distributions'],
    index=0
)

if show_dists == 'Events':
    data_sets = [d.name for d in files_path.iterdir() if d.is_dir()]
    data_set = st.sidebar.selectbox(
        'Select dataset',
        options=data_sets,
        index=0
    )

    particle_types = ['120000', '140000', '160000']
    particle_type = st.sidebar.selectbox(
        'Select particle type',
        options=particle_types,
        index=0
    )

    events_files = file_finder(data_set, particle_type)

    clean_masks = h5_groups_reader(events_files[0], 'masks')
    clean_mask = st.sidebar.selectbox(
        'Select cleaning mask',
        options=clean_masks,
        index=1
    )

    events_file = st.sidebar.selectbox(
        'Select file',
        options=events_files,
        format_func=lambda x: x.stem.split('.')[-1].split('__')[0],
        index=0
    )
    meta = meta_data(str(events_file))

    # Read HDF5 geometry file
    geom = read_geom_file(geom_file)
    axis_lims = geom.x.max() + 0.4 * geom.x.max()
    template = create_template()

    manual = st.sidebar.radio(
        'Input event or browse?',
        options=['Input', 'Browse'],
        index=0
    )

    if manual == 'Input':
        event_no = st.sidebar.text_input(
            'Enter event number:',
            0
        )
        event_no = int(event_no)
    else:
        browse_type = st.sidebar.selectbox(
            'Browsing type',
            [
                'energy',
                'toi_ratio',
                'no_of_doms',
                'integrated_charge'
            ],
            index=0
        )
        sort = st.sidebar.radio(
            'Sort by',
            ['High', 'Low'],
            0
        )
        if sort == 'High':
            meta = meta.sort_values(browse_type, ascending=False)
        elif sort == 'Low':
            meta = meta.sort_values(browse_type, ascending=True)
        meta.energy = meta.energy.apply(lambda x: 10**x)
        select_meta = meta[browse_type].iloc[0:100]
        event_no = st.sidebar.selectbox(
            'Select {} value of event'.format(browse_type),
            options=select_meta.index,
            format_func=lambda x: round(select_meta[select_meta.index == x].values[0], 2)
        )

    # Get activations and truth from selected event number
    activations, truth = read_event(str(events_file), event_no, clean_mask)
    true_muon_entry_pos = truth[
        [
            'true_primary_position_x',
            'true_primary_position_y',
            'true_primary_position_z',
        ]
    ]

    activations = activations.sort_values('dom_time')
    hist_file = root.joinpath('powershovel/histograms/' + data_set + '/files/')
    hist_file = hist_file.joinpath(particle_type + '.h5')
    time = h5_data_reader(hist_file, 'histograms/transform1/edges/dom_time', 'all')
    time_range = [min(time), max(time)]
    st.write(
        'Selected event no. {}, length of event {}, ToI ratio {}, energy {}'
        .format(
            event_no,
            len(activations),
            round(truth.toi_evalratio.values[0], 3),
            float(round(10**truth['true_primary_energy'], 3))
        )
    )
    superpose = st.multiselect(
        'Choose wisely',
        options=['Truth', 'ToI'],
    )
    event_prediction_vector = None
    # Create plotly figure
    fig = create_static(
        activations,
        geom,
        truth,
        event_prediction_vector,
        template,
        superpose,
        time_range
    )
    # Plot plotly figure
    st.plotly_chart(fig, width=0, height=700)
    st.text('Event dataframe:')
    st.write(activations.sort_values('dom_time'))
    doms_fig = px.histogram(meta, x='no_of_doms', log_y=False)
    st.plotly_chart(doms_fig)
    st.write('Mean number of DOMs:', meta.no_of_doms.mean())
elif show_dists == 'Distributions':
    hist_type = st.sidebar.selectbox(
        'All files or sets?',
        ['All files', 'Sets'],
        index=0
    )
    if hist_type == 'All files':
        data_sets = [d.name for d in files_path.iterdir() if d.is_dir()]
        data_set = st.sidebar.selectbox(
            'Select dataset',
            options=data_sets,
            index=0
        )
        hist_path = root.joinpath('powershovel/histograms/' + data_set + '/files')
        particle_types = [f.stem for f in hist_path.glob('**/*.h5') if f.is_file()]
        particle_type = st.sidebar.selectbox(
            'Select particle type',
            options=particle_types,
            index=0
        )
        file = hist_path.joinpath(particle_type + '.h5')
        transforms = h5_groups_reader(file, 'histograms')
        transform = st.sidebar.selectbox(
            label='Choose transform',
            options=transforms
        )
        group = 'histograms/' + transform + '/edges'
        keys = h5_groups_reader(file, group)
        key = st.sidebar.selectbox(
            label='Choose key',
            options=keys
        )
        values_group = 'histograms/' + transform + '/values/' + key
        edges_group = 'histograms/' + transform + '/edges/' + key
        values = h5_data_reader(file, values_group, 'all')
        edges = h5_data_reader(file, edges_group, 'all')
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]
        bins = len(edges) - 1
        st.write('No. of bins:', bins)
        st.write('Transform:', transform)
        st.write('Key:', key)

        fig = go.Figure(
            data=go.Bar(
                x=centers,
                y=values,
                width=width
            )
        )

        fig.update_layout(
            updatemenus=[
                go.layout.Updatemenu(
                    buttons=list(
                        [
                            dict(
                                label="Linear",
                                method="update",
                                args=[
                                    {
                                        'visible': [
                                            True,
                                            False
                                        ]
                                    },
                                    {
                                        'yaxis': {
                                            'type': 'linear'
                                        }
                                    }
                                ]
                            ),
                            dict(
                                label="Log",
                                method="update",
                                args=[
                                    {
                                        'visible': [
                                            True,
                                            True
                                        ]
                                    },
                                    {
                                        'yaxis': {
                                            'type': 'log'
                                        }
                                    }
                                ]
                            )
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                )
            ]
        )

        fig.update_traces(
            marker_color='black',
            marker_line_color='black',
        )

        st.plotly_chart(fig)
