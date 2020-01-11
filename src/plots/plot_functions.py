import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax
