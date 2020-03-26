import pickle
import shelve
import numpy as np
import pandas as pd
from pathlib import Path
import shelve


def find_clip_y_values(data, percentiles):
    percentile_min = np.nanpercentile(data, percentiles[0])
    percentile_max = np.nanpercentile(data, percentiles[1])
    return [percentile_min, percentile_max]


def calculate_bins(df):
    no_of_energy_bins = 18
    no_of_dom_bins = 20
    df['energy_binned'] = pd.cut(
        df['true_primary_energy'],
        no_of_energy_bins
    )
    df['dom_binned'] = pd.cut(
        df['event_length'],
        np.arange(0, 160, 5)
    )
    output_energy_bin_mids = [ibin.mid for ibin in np.sort(df['energy_binned'].unique())]
    output_dom_bins_mids = [ibin.mid for ibin in np.sort(df['dom_binned'].unique())]
    output_energy_bin_widths = [ibin.length / 2 for ibin in np.sort(df['energy_binned'].unique())]
    output_dom_bin_widths = [ibin.length / 2 for ibin in np.sort(df['dom_binned'].unique())]
    output_energy_bins = np.sort(df['energy_binned'].unique()).astype(str)
    output_dom_bins = np.sort(df['dom_binned'].unique()).astype(str)
    return output_energy_bins, output_energy_bin_mids, output_energy_bin_widths, output_dom_bins, output_dom_bins_mids, output_dom_bin_widths


def calculate_2d_bin_edges(data_x, data_y, clip_y_values, bin_type):
    bin_edges_x = np.histogram_bin_edges(np.clip(data_x, clip_y_values[0], clip_y_values[1]), bins=200)
    if bin_type == 'true_primary_energy':
        bin_edges_y = np.histogram_bin_edges(data_y, bins=200)
    elif bin_type == 'event_length':
        bin_edges_y = np.histogram_bin_edges(data_y, bins=np.arange(0, 160, 5))
    return bin_edges_x, bin_edges_y


def calculate_2d_histograms(df, bin_type, key, two_d_bins, other_key):
    if bin_type == 'true_primary_energy':
        x_name = 'true_primary_energy'
        bin_name = 'energy_binned'
        bins = np.sort(df['energy_binned'].unique())
    elif bin_type == 'event_length':
        x_name = 'event_length'
        bin_name = 'dom_binned'
        bins = np.sort(df['dom_binned'].unique())
    output_dict = {}
    percentile_dict = {}
    clip_y_values = find_clip_y_values(df[other_key].values, [1, 99])
    H, xedges, yedges = np.histogram2d(
        x=np.clip(df[key].values, clip_y_values[0], clip_y_values[1]),
        y=df[x_name].values,
        bins=two_d_bins
    )
    output_dict['hist'] = H
    percentile_dict['low_percentile'] = [np.nanpercentile(df[df[bin_name] == ibin][key].values, 16) for ibin in bins]
    percentile_dict['median'] = [np.nanpercentile(df[df[bin_name] == ibin][key].values, 50) for ibin in bins]
    percentile_dict['high_percentile'] = [np.nanpercentile(df[df[bin_name] == ibin][key].values, 84) for ibin in bins]
    percentile_dict['low_cut_percentile'] = [np.nanpercentile(df[df[bin_name] == ibin][key].values, 0.5) for ibin in bins]
    percentile_dict['high_cut_percentile'] = [np.nanpercentile(df[df[bin_name] == ibin][key].values, 99.5) for ibin in bins]
    return output_dict, percentile_dict


def calculate_error_histograms(df, bin_type, key):
    if bin_type == 'true_primary_energy':
        x_name = 'true_primary_energy'
        bin_name = 'energy_binned'
        bins = np.sort(df['energy_binned'].unique())
    elif bin_type == 'event_length':
        x_name = 'event_length'
        bin_name = 'dom_binned'
        bins = np.sort(df['dom_binned'].unique())
    output_dict = {}
    for i, ibin in enumerate(bins):
        output_dict[i] = {}
        hist, bin_edges = np.histogram(df[df[bin_name] == ibin][key].values, bins=100)
        output_dict[i]['hist'] = hist
        output_dict[i]['bin_edges'] = bin_edges
    return output_dict


def calculate_performance(data):
    performance = [(data['high_percentile'][i] - data['low_percentile'][i]) / 2 for i in range(len(data['high_percentile']))]
    return performance
# %%
own_errors = [
    'predicted_azimuth_error',
    'predicted_zenith_error',
    'predicted_energy_error',
    'predicted_time_error',
    'predicted_x_error',
    'predicted_y_error',
    'predicted_z_error'
]
icecube_errors = [
    'opponent_azimuth_error',
    'opponent_zenith_error',
    'opponent_energy_error',
    'opponent_time_error',
    'opponent_x_error',
    'opponent_y_error',
    'opponent_z_error'
]
# %%
run_names = ['ingenious-poodle', 'lively-vulture']
file_name = 'run_dataframe.gzip'
runs_dir = Path().home().joinpath('runs')
output_dict = {}
output_dict['retro_crs_prefit'] = {}
# %%
base_run = runs_dir.joinpath(run_names[0]).joinpath(file_name)
base_run_df = pd.read_parquet(base_run)
base_run_df = base_run_df[base_run_df['true_primary_energy'] <= 3.0]
base_run_df = base_run_df[base_run_df['event_length'] <= 150]
no_of_energy_bins = 18
no_of_dom_bins = 20
base_run_df['energy_binned'] = pd.cut(
    base_run_df['true_primary_energy'],
    no_of_energy_bins
)
print(base_run_df['energy_binned'].unique())
base_run_df['dom_binned'] = pd.cut(
    base_run_df['event_length'],
    np.arange(0, 160, 5)
)
output_dict['meta'] = {}
output_dict['meta']['events'] = {}
output_dict['meta']['bin_edges'] = {}
for i, (bin_type, pretty_print_bin_type, binning_bin_type) in enumerate(zip(['true_primary_energy', 'event_length'], ['energy', 'event_length'], ['energy_binned', 'dom_binned'])):
    output_dict['meta']['events'][pretty_print_bin_type] = {}
    for j, ibin in enumerate(np.sort(base_run_df[binning_bin_type].unique())):
        output_dict['meta']['events'][pretty_print_bin_type][j] = base_run_df[base_run_df[binning_bin_type] == ibin]['event'].values
    output_dict['meta']['bin_edges'][pretty_print_bin_type] = {}
    for key in icecube_errors:
        clip_y_values = find_clip_y_values(base_run_df[key].values, [1, 99])
        pretty_print_key = key.replace('opponent_', '').replace('_error', '')
        temp = calculate_2d_bin_edges(base_run_df[key].values, base_run_df[bin_type].values, clip_y_values, bin_type)
        output_dict['meta']['bin_edges'][pretty_print_bin_type]['yedges'] = temp[1]
        output_dict['meta']['bin_edges'][pretty_print_bin_type][pretty_print_key] = {}
        output_dict['meta']['bin_edges'][pretty_print_bin_type][pretty_print_key]['xedges'] = temp[0]
# %%
for i, run_name in enumerate(run_names):
    print(run_name)
    runs_file = runs_dir.joinpath(run_name).joinpath(file_name)
    runs_df = pd.read_parquet(runs_file)
    runs_df = runs_df[runs_df['true_primary_energy'] <= 3.0]
    runs_df = runs_df[runs_df['event_length'] <= 150]
    output_dict[run_name] = {}
    if i == 0:
        print_bins = calculate_bins(runs_df)
    else:
        _ = calculate_bins(runs_df)
    for key, opponent_key in zip(own_errors, icecube_errors):
        pretty_print_key = key.replace('predicted_', '').replace('_error', '')
        output_dict[run_name][pretty_print_key] = {}
        print(key)
        for bin_type, pretty_print_bin_type in zip(['true_primary_energy', 'event_length'], ['energy', 'event_length']):
            output_dict[run_name][pretty_print_key][pretty_print_bin_type] = {}
            output_dict[run_name][pretty_print_key][pretty_print_bin_type]['percentiles'] = {}
            output_dict[run_name][pretty_print_key][pretty_print_bin_type]['2d_histogram'] = {}
            output_dict[run_name][pretty_print_key][pretty_print_bin_type]['error_histograms'] = {}
            output_dict[run_name][pretty_print_key][pretty_print_bin_type]['2d_histogram'], output_dict[run_name][pretty_print_key][pretty_print_bin_type]['percentiles'] = calculate_2d_histograms(runs_df, bin_type, key, [output_dict['meta']['bin_edges'][pretty_print_bin_type][pretty_print_key]['xedges'], output_dict['meta']['bin_edges'][pretty_print_bin_type]['yedges']], opponent_key)
            output_dict[run_name][pretty_print_key][pretty_print_bin_type]['error_histograms'] = calculate_error_histograms(runs_df, bin_type, key)
            output_dict[run_name][pretty_print_key][pretty_print_bin_type]['performance'] = calculate_performance(output_dict[run_name][pretty_print_key][pretty_print_bin_type]['percentiles'])
# %%
run_name = run_names[0]
runs_file = runs_dir.joinpath(run_name).joinpath(file_name)
runs_df = pd.read_parquet(runs_file)
runs_df = runs_df[runs_df['true_primary_energy'] <= 3.0]
runs_df = runs_df[runs_df['event_length'] <= 150]
_ = calculate_bins(runs_df)
print('icecube')
for key in icecube_errors:
    pretty_print_key = key.replace('opponent_', '').replace('_error', '')
    print(key)
    output_dict['retro_crs_prefit'][pretty_print_key] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['energy'] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['energy']['percentiles'] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['energy']['2d_histogram'] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['energy']['error_histograms'] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['event_length'] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['event_length']['percentiles'] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['event_length']['2d_histogram'] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['event_length']['error_histograms'] = {}
    output_dict['retro_crs_prefit'][pretty_print_key]['energy']['2d_histogram'], output_dict['retro_crs_prefit'][pretty_print_key]['energy']['percentiles'] = calculate_2d_histograms(runs_df, 'true_primary_energy', key, [output_dict['meta']['bin_edges']['energy'][pretty_print_key]['xedges'], output_dict['meta']['bin_edges']['energy']['yedges']], key)
    output_dict['retro_crs_prefit'][pretty_print_key]['event_length']['2d_histogram'], output_dict['retro_crs_prefit'][pretty_print_key]['event_length']['percentiles'] = calculate_2d_histograms(runs_df, 'event_length', key, [output_dict['meta']['bin_edges']['event_length'][pretty_print_key]['xedges'], output_dict['meta']['bin_edges']['event_length']['yedges']], key)
    output_dict['retro_crs_prefit'][pretty_print_key]['energy']['error_histograms'] = calculate_error_histograms(runs_df, 'true_primary_energy', key)
    output_dict['retro_crs_prefit'][pretty_print_key]['event_length']['error_histograms'] = calculate_error_histograms(runs_df, 'event_length', key)
    output_dict['retro_crs_prefit'][pretty_print_key]['energy']['performance'] = calculate_performance(output_dict['retro_crs_prefit'][pretty_print_key]['energy']['percentiles'])
    output_dict['retro_crs_prefit'][pretty_print_key]['event_length']['performance'] = calculate_performance(output_dict['retro_crs_prefit'][pretty_print_key]['event_length']['percentiles'])

output_dict['meta']['energy'] = {}
output_dict['meta']['event_length'] = {}
output_dict['meta']['energy']['bins'] = print_bins[0]
output_dict['meta']['energy']['bin_mids'] = print_bins[1]
output_dict['meta']['energy']['bin_widths'] = print_bins[2]
output_dict['meta']['event_length']['bins'] = print_bins[3]
output_dict['meta']['event_length']['bin_mids'] = print_bins[4]
output_dict['meta']['event_length']['bin_widths'] = print_bins[5]
output_dict['meta']['train_dists'] = {}

train_distribution_df = pd.read_parquet(Path().home().joinpath('train_dists_parquet.gzip'))
train_distribution_df = train_distribution_df[train_distribution_df['train_true_energy'] <= 3.0]
train_distribution_df = train_distribution_df[train_distribution_df['train_event_length'] <= 150]
no_of_energy_bins = 18
no_of_dom_bins = 20
train_distribution_df['energy_binned'] = pd.cut(
    train_distribution_df['train_true_energy'],
    no_of_energy_bins
)
train_distribution_df['dom_binned'] = pd.cut(
    train_distribution_df['train_event_length'],
    np.arange(0, 160, 5)
)
for bin_type, pretty_print_bin_type in zip(['energy_binned', 'dom_binned'], ['energy', 'event_length']):
    counts = []
    for ibin in np.sort(train_distribution_df[bin_type].unique()):
        counts.append(len(train_distribution_df[train_distribution_df[bin_type] == ibin]['train_true_energy'].values))
    output_dict['meta']['train_dists'][pretty_print_bin_type] = counts

test_dict = {}
test_dict['datasets'] = {}
for run_name in run_names:
    test_dict['datasets'][run_name] = output_dict[run_name]
test_dict['datasets']['retro_crs_prefit'] = output_dict['retro_crs_prefit']

with shelve.open(str(runs_dir.joinpath('powershovel')), 'n') as db:
    db['datasets'] = test_dict['datasets']
    db['meta'] = output_dict['meta']

# with open(str(runs_dir.joinpath('powershovel.pickle')), 'wb') as f:
#     pickle.dump(output_dict, f)
