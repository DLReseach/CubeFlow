import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compare_ic_histogram(
    clip_y,
    resolution_y,
    H,
    x_labels,
    meta    
):
    y_labels = np.linspace(clip_y[0], clip_y[1], resolution_y)

    max_z_values = []
    max_z_values.append(np.amax(H[0]))
    max_z_values.append(np.amax(H[1]))
    max_z_value = max(max_z_values)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(meta['title_1'], meta['title_2']))

    fig.add_trace(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=H[0],
            colorscale='Oranges',
            zmin=0,
            zmax=max_z_value,
            showscale=True
        ),
        row=1, col=1
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=x_labels,
    #         y=median
    #     ),
    #     row=1, col=1
    # )
    fig.add_trace(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=H[1],
            colorscale='Oranges',
            zmin=0,
            zmax=max_z_value,
            showscale=False
        ),
        row=1, col=2
    )
    fig.update_layout(
        font=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
    fig.update_xaxes(title_text=meta['x_label'], row=1, col=1)
    fig.update_xaxes(title_text=meta['x_label'], row=1, col=2)
    fig.update_yaxes(title_text=meta['y_label'], row=1, col=1)
    fig.update_yaxes(title_text=meta['y_label'], row=1, col=2)
    fig.update_layout(
        autosize=False,
        width=800,
        height=700
    )
    return fig