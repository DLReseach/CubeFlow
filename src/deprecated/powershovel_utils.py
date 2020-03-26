import numpy as np


def find_clip_y_values(dfs, percentiles):
    percentile_min = []
    percentile_max = []
    for df in dfs:
        percentile_min.append(df.quantile(percentiles[0]))
        percentile_max.append(df.quantile(percentiles[1]))
    clip_values = [min(percentile_min), max(percentile_max)]
    return clip_values


def make_2d_histograms(dfs, metric, bin_in, resolution, percentiles):
    clip_y_values = find_clip_y_values((dfs[0][metric], dfs[1][metric]), percentiles)

    H1, xedges1, yedges1 = np.histogram2d(
        x=np.clip(dfs[0][metric].values, clip_y_values[0], clip_y_values[1]),
        y=dfs[0][bin_in].values,
        bins=resolution
    )

    H2, xedges2, yedges2 = np.histogram2d(
        x=np.clip(dfs[1][metric].values, clip_y_values[0], clip_y_values[1]),
        y=dfs[1][bin_in].values,
        bins=resolution
    )

    max_z_values = []
    max_z_values.append(np.amax(H1))
    max_z_values.append(np.amax(H2))
    max_z_value = max(max_z_values)

    return (H1, xedges1, yedges1), (H2, xedges2, yedges2), max_z_value


def calculate_resolution_width(df, quantiles):
    low = df.quantile(quantiles[0])
    high = df.quantile(quantiles[1])
    width = (high - low) / 2
    return width


def calculate_resolution_widths(own_errors, opponent_errors, metric, bins, bin_name, quantiles):
    own_widths = []
    opponent_widths = []
    for ibin in bins:
        own_widths.append(calculate_resolution_width(own_errors[own_errors[bin_name] == ibin][metric], quantiles))
        opponent_widths.append(calculate_resolution_width(opponent_errors[opponent_errors[bin_name] == ibin][metric], quantiles))
    return own_widths, opponent_widths
