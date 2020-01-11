import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def histogram(data, title, xlabel, ylabel, width_scale=0.7, bins='fd'):
    values, edges = np.histogram(data, bins=bins)
    width = width_scale * (edges[1] - edges[0])
    center = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots()
    ax.bar(center, values, align='center', width=width)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def matplotlib_histogram(data, title, xlabel, ylabel, bins='fd'):
    fig = go.Figure()
    fig.add_trace(
        go.histogram(
            x=data
        ),
        marker_color='black'
    )
    return fig
