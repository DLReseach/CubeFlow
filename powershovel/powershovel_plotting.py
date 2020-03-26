import plotly.graph_objects as go
from plotly import tools
import numpy as np


def y_axis_print(metric):
    if metric == 'energy':
        y_axis_text = '(x<sub>reco</sub> - x<sub>true</sub>) / x<sub>true</sub>'
    elif metric == 'azimuth' or metric == 'zenith':
        y_axis_text = 'x<sub>reco</sub> - x<sub>true</sub> (rad)'
    elif metric == 'time':
        y_axis_text = 'x<sub>reco</sub> - x<sub>true</sub> (ns)'
    elif metric == 'direction_x' or metric == 'direction_y' or metric == 'direction_z':
        y_axis_text = 'x<sub>reco</sub> - x<sub>true</sub> (m)'
    elif metric == 'position_x' or metric == 'position_y' or metric == 'position_z':
        y_axis_text = 'x<sub>reco</sub> - x<sub>true</sub> (m)'
    elif metric == 'angle':
        y_axis_text = 'x<sub>reco</sub> - x<sub>true</sub> (rad)'
    elif metric == 'vertex':
        y_axis_text = 'x<sub>reco</sub> - x<sub>true</sub> (rad)'
    return y_axis_text


def plotly_2d_histograms(selected_run_data, selected_comparison_data, metric, bin_type, own_name, opponent_name, meta_data):
    x_axis_text = 'Event length' if bin_type == 'event_length' else 'log E (E / GeV)'
    y_axis_text = y_axis_print(metric)
    H1 = selected_run_data[metric][bin_type]['2d_histogram']['hist']
    H2 = selected_comparison_data[metric][bin_type]['2d_histogram']['hist']
    y_edges = meta_data['bin_edges'][bin_type]['yedges']
    x_edges = meta_data['bin_edges'][bin_type][metric]['xedges']
    percentile_x_mids = meta_data[bin_type]['bin_mids']
    max_z_values = []
    max_z_values.append(np.amax(H1))
    max_z_values.append(np.amax(H2))
    max_z_value = max(max_z_values)
    x_bin_mids = (y_edges[:-1] + y_edges[1:]) / 2
    y_bin_mids = (x_edges[:-1] + x_edges[1:]) / 2
    x_bin_widths = 1.0 * (x_edges[1] - x_edges[0])
    y_bin_widths = 1.0 * (y_edges[1] - y_edges[0])
    trace1 = go.Heatmap(
        x=x_bin_mids,
        y=y_bin_mids,
        z=H1,
        zmin=0,
        zmax=max_z_value,
        showscale=True,
        xaxis='x1',
        yaxis='y1',
        coloraxis = 'coloraxis',
    )

    trace2 = go.Heatmap(
        x=x_bin_mids,
        y=y_bin_mids,
        z=H2,
        zmin=0,
        zmax=max_z_value,
        showscale=False,
        xaxis='x2',
        yaxis='y1',
        coloraxis = 'coloraxis'
    )

    trace3 = go.Scatter(
        x=percentile_x_mids,
        y=selected_run_data[metric][bin_type]['percentiles']['low_percentile'],
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
        x=percentile_x_mids,
        y=selected_run_data[metric][bin_type]['percentiles']['median'],
        mode='lines',
        line=dict(
            color='Red'
        ),
        xaxis='x1',
        yaxis='y1',
        name=own_name,
        showlegend=True
    )
    trace5 = go.Scatter(
        x=percentile_x_mids,
        y=selected_run_data[metric][bin_type]['percentiles']['high_percentile'],
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
        x=percentile_x_mids,
        y=selected_comparison_data[metric][bin_type]['percentiles']['low_percentile'],
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
        x=percentile_x_mids,
        y=selected_comparison_data[metric][bin_type]['percentiles']['median'],
        mode='lines',
        line=dict(
            color='Green'
        ),
        xaxis='x1',
        yaxis='y1',
        showlegend=True,
        name=opponent_name
    )
    trace8 = go.Scatter(
        x=percentile_x_mids,
        y=selected_comparison_data[metric][bin_type]['percentiles']['high_percentile'],
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
        x=percentile_x_mids,
        y=selected_run_data[metric][bin_type]['percentiles']['low_percentile'],
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
        x=percentile_x_mids,
        y=selected_run_data[metric][bin_type]['percentiles']['median'],
        mode='lines',
        line=dict(
            color='Red'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace11 = go.Scatter(
        x=percentile_x_mids,
        y=selected_run_data[metric][bin_type]['percentiles']['high_percentile'],
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
        x=percentile_x_mids,
        y=selected_comparison_data[metric][bin_type]['percentiles']['low_percentile'],
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
        x=percentile_x_mids,
        y=selected_comparison_data[metric][bin_type]['percentiles']['median'],
        mode='lines',
        line=dict(
            color='Green'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace14 = go.Scatter(
        x=percentile_x_mids,
        y=selected_comparison_data[metric][bin_type]['percentiles']['high_percentile'],
        mode='lines',
        line=dict(
            color='Green',
            dash='dash'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=(own_name, opponent_name))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 1, 1)
    fig.append_trace(trace5, 1, 1)
    fig.append_trace(trace6, 1, 1)
    fig.append_trace(trace7, 1, 1)
    fig.append_trace(trace8, 1, 1)
    fig.append_trace(trace9, 1, 2)
    fig.append_trace(trace10, 1, 2)
    fig.append_trace(trace11, 1, 2)
    fig.append_trace(trace12, 1, 2)
    fig.append_trace(trace13, 1, 2)
    fig.append_trace(trace14, 1, 2)

    fig['layout']['xaxis1'].update(title=x_axis_text)
    fig['layout']['xaxis2'].update(title=x_axis_text)
    fig['layout']['yaxis1'].update(title='Error')
    fig['layout']['yaxis1'].update(range=[min(x_edges), max(x_edges)])
    fig['layout']['yaxis1'].update(title=y_axis_text)
    fig['layout']['yaxis2'].update(title='Error')
    fig['layout']['yaxis2'].update(title=y_axis_text)
    fig['layout']['yaxis2'].update(range=[min(x_edges), max(x_edges)])
    fig['layout']['coloraxis'].update(colorscale='Oranges')

    # fig.add_trace(go.Scatter(
    #     x=[2],
    #     y=[2],
    #     mode='text',
    #     xaxis='x1',
    #     yaxis='y1',
    #     text=['Text I'],
    #     textposition='bottom center',
    #     textfont=dict(
    #         family='sans serif',
    #         size=18,
    #         color='LightSeaGreen'
    #         )
    #     )
    # )

    fig.update_layout(
        annotations=[
            dict(
                x=0.2,
                y=1.05,
                showarrow=False,
                text=own_name,
                xref='paper',
                yref='paper'
            ),
            dict(
                x=0.1,
                y=1.05,
                showarrow=False,
                text='opponent_name',
                xref='paper',
                yref='paper',
            ),
            dict(
                x=0.01,
                y=0.99,
                showarrow=False,
                text='Events: {}'.format(int(np.sum(H1))),
                xref='paper',
                yref='paper',
                bgcolor='white'
            ),
            dict(
                x=0.99,
                y=0.99,
                showarrow=False,
                text='Events: {}'.format(int(np.sum(H2))),
                xref='paper',
                yref='paper',
                bgcolor='white'
            ),
        ]
    )


    # fig.add_annotation(
    #     x=2,
    #     y=1,
    #     text='dict Text 2',
    #     showarrow=True,
    # )

    fig.update_layout(
        legend=dict(
            x=-.1,
            y=1.2,
            orientation='h'
        ),
        height=700
    )
    return fig


def plotly_error_comparison(selected_run_data, selected_comparison_data, metric, own_name, opponent_name, meta_data, bin_no):
    x_axis_text = 'log E (E / GeV)'
    y_axis_text = y_axis_print(metric)
    bin_mids = meta_data['1d_histogram']['bin_mids']
    bin_widths = meta_data['1d_histogram']['bin_widths']
    bin_lengths = [ibin * 2 for ibin in bin_widths]
    bin_counts = meta_data['1d_histogram']['counts']
    own_error_bin_edges = selected_run_data[metric]['error_histograms'][bin_no]['x_edges']
    own_error_x_bin_mids = (own_error_bin_edges[:-1] + own_error_bin_edges[1:]) / 2
    own_error_x_bin_widths = 1.0 * (own_error_bin_edges[1] - own_error_bin_edges[0])
    own_percentiles = selected_run_data[metric]['error_histograms'][bin_no]['percentiles']
    own_performance = selected_run_data[metric]['1d_histogram']['performance']
    opponent_error_bin_edges = selected_comparison_data[metric]['error_histograms'][bin_no]['x_edges']
    opponent_error_x_bin_mids = (opponent_error_bin_edges[:-1] + opponent_error_bin_edges[1:]) / 2
    opponent_error_x_bin_widths = 1.0 * (opponent_error_bin_edges[1] - opponent_error_bin_edges[0])
    opponent_percentiles = selected_comparison_data[metric]['error_histograms'][bin_no]['percentiles']
    opponent_performance = selected_comparison_data[metric]['1d_histogram']['performance']
    histogram_colors = ['grey'] * len(bin_mids)
    histogram_colors[bin_no] = 'red'
    trace1 = go.Bar(
        x=bin_mids,
        y=bin_counts,
        width=bin_lengths,
        opacity=0.2,
        marker_color=histogram_colors,
        xaxis='x1',
        yaxis='y2',
        name='test data',
        showlegend=True
    )

    trace2 = go.Scatter(
        x=bin_mids,
        y=own_performance,
        mode='markers',
        name=own_name,
        showlegend=True,
        marker=dict(
            color='Red'
        ),
        error_x=dict(
            type='data',
            array=bin_widths,
            visible=True
        ),
        xaxis='x1'
    )

    trace3 = go.Scatter(
        x=bin_mids,
        y=opponent_performance,
        mode='markers',
        name=opponent_name,
        showlegend=True,
        marker=dict(
            color='Green'
        ),
        error_x=dict(
            type='data',
            array=bin_widths,
            visible=True
        ),
        xaxis='x1'
        )

    trace4 = go.Bar(
        x=own_error_x_bin_mids,
        y=selected_run_data[metric]['error_histograms'][bin_no]['counts'],
        width=own_error_x_bin_widths,
        xaxis='x2',
        yaxis='y3',
        name=own_name + ' error',
        opacity=0.5,
        marker_color='Red',
        showlegend=False
    )
    trace5 = go.Bar(
        x=opponent_error_x_bin_mids,
        y=selected_comparison_data[metric]['error_histograms'][bin_no]['counts'],
        width=opponent_error_x_bin_widths,
        xaxis='x2',
        yaxis='y3',
        name=opponent_name + ' error',
        opacity=0.5,
        marker_color='Green',
        showlegend=False
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        xaxis=dict(
            rangemode='tozero',
            title=x_axis_text,
            domain=[0, 0.6]
        ),
        xaxis2=dict(
            title=y_axis_text,
            domain=[0.7, 1],
            # range=[
            #     selected_run_data[metric][bin_type]['percentiles']['low_cut_percentile'][bin_no],
            #     selected_run_data[metric][bin_type]['percentiles']['high_cut_percentile'][bin_no]
            # ]
        ),
        yaxis=dict(
            rangemode='tozero',
            title='(84th percentile - 16th percentile) / 2',
            # range=[0,]
        ),
        yaxis2=dict(
            title='Events',
            type='log',
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='Frequency',
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

    fig.update_layout(
        annotations=[
            dict(
                x=0.01,
                y=0.99,
                showarrow=False,
                text='Events: {}'.format(int(sum(bin_counts))),
                xref='paper',
                yref='paper',
                bgcolor='white'
            ),
            dict(
                x=0.99,
                y=0.99,
                showarrow=False,
                text='Events: {}'.format(int(sum(selected_run_data[metric]['error_histograms'][bin_no]['counts']))),
                xref='paper',
                yref='paper',
                bgcolor='white'
            ),
        ]
    )

    fig.add_shape(
                type='line',
                xref='x2',
                yref='paper',
                x0=own_percentiles[0],
                y0=0,
                x1=own_percentiles[0],
                y1=1,
                name=own_name + ' 16\%',
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
                x0=own_percentiles[2],
                y0=0,
                x1=own_percentiles[2],
                y1=1,
                name=own_name + ' 84\%',
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
                x0=opponent_percentiles[0],
                y0=0,
                x1=opponent_percentiles[0],
                y1=1,
                name=opponent_name + ' 16\%',
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
                x0=opponent_percentiles[2],
                y0=0,
                x1=opponent_percentiles[2],
                y1=1,
                name=opponent_name + ' 84\%',
                line=dict(
                    color='Green',
                    width=1,
                    dash='dot'
                ),
            )
    return fig


def convert_to_cartesian(azimuth, zenith):
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.array([[x, y, z]])


def direction_vectors(direction, position, scale):
    x = [position[0, 0] - direction[0, 0] * scale, direction[0, 0] * scale + position[0, 0]]
    y = [position[0, 1] - direction[0, 1] * scale, direction[0, 1] * scale + position[0, 1]]
    z = [position[0, 2] - direction[0, 2] * scale, direction[0, 2] * scale + position[0, 2]]
    return np.array([x, y, z])


def plotly_event(predictions, truth, sequential, comparison, geom, selected_run, selected_comparison):
    own_dirs = predictions[['direction_x', 'direction_y', 'direction_z']].values
    own_pos = predictions[['position_x', 'position_y', 'position_z']].values

    true_dirs = truth[['direction_x', 'direction_y', 'direction_z']].values
    true_pos = truth[['position_x', 'position_y', 'position_z']].values

    opp_dirs = comparison[['direction_x', 'direction_y', 'direction_y']].values
    opp_pos = comparison[['position_x', 'position_y', 'position_z']].values

    own_vector = direction_vectors(own_dirs, own_pos, 100000)
    true_vector = direction_vectors(true_dirs, true_pos, 100000)
    opp_vector = direction_vectors(opp_dirs, opp_pos, 100000)

    trace1 = go.Scatter3d(
        x=sequential.dom_x.values,
        y=sequential.dom_y.values,
        z=sequential.dom_z.values,
        mode='markers',
        name='pulses',
        marker=dict(
            size=10,
            color=sequential.dom_time.values,
            colorscale='Viridis',
            opacity=0.5
        )
    )

    trace2 = go.Scatter3d(
        x=geom.x.values,
        y=geom.y.values,
        z=geom.z.values,
        mode='markers',
        name='DOMs',
        marker=dict(
            size=1,
            color='black',
            opacity=0.5
        )
    )

    trace3 = go.Scatter3d(
        x=own_vector[0, :],
        y=own_vector[1, :],
        z=own_vector[2, :],
        mode='lines',
        name=selected_run,
        marker=dict(
            color='red',
            size=0.1
        )
    )

    trace4 = go.Scatter3d(
        x=opp_vector[0, :],
        y=opp_vector[1, :],
        z=opp_vector[2, :],
        mode='lines',
        name=selected_comparison,
        marker=dict(
            color='green',
            size=0.1
        )
    )

    trace5 = go.Scatter3d(
        x=true_vector[0, :],
        y=true_vector[1, :],
        z=true_vector[2, :],
        mode='lines',
        name='truth',
        marker=dict(
            color='blue',
            size=0.1
        )
    )

    trace6 = go.Scatter3d(
        x=[own_pos[0, 0]],
        y=[own_pos[0, 1]],
        z=[own_pos[0, 2]],
        mode='markers',
        name=selected_run + ' vertex',
        showlegend=False,
        marker=dict(
            color='red'
        )
    )

    trace7 = go.Scatter3d(
        x=[opp_pos[0, 0]],
        y=[opp_pos[0, 1]],
        z=[opp_pos[0, 2]],
        mode='markers',
        name=selected_comparison + ' vertex',
        showlegend=False,
        marker=dict(
            color='green'
        )
    )

    trace8 = go.Scatter3d(
        x=[true_pos[0, 0]],
        y=[true_pos[0, 1]],
        z=[true_pos[0, 2]],
        mode='markers',
        name='true vertex',
        showlegend=False,
        marker=dict(
            color='blue'
        )
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
    return fig

def plotly_loss(selected_run_data):
    train_y = selected_run_data['meta']['train_loss']
    loss_x = np.arange(len(train_y))
    val_y = selected_run_data['meta']['val_loss']
    learning_rate_y = selected_run_data['meta']['learning_rate']
    learning_rate_x = np.arange(len(learning_rate_y))

    trace1 = go.Scatter(
        x=loss_x,
        y=train_y,
        name='train',
        xaxis='x1',
        yaxis='y1'
    )
    trace2 = go.Scatter(
        x=loss_x,
        y=val_y,
        name='val',
        xaxis='x1',
        yaxis='y1'
    )
    trace3 = go.Scatter(
        x=learning_rate_x,
        y=learning_rate_y,
        xaxis='x2',
        yaxis='y2'
        showlegend=False
    )
    data = [trace1, trace2]
    layout = go.Layout(
        xaxis=dict(
            title='iteration',
            domain=[0, 0.45]
        ),
        xaxis2=dict(
            title='iteration',
            domain=[0.55, 1]
        ),
        yaxis=dict(
            title='loss',
        ),
        yaxis2=dict(
            title='learning rate',
            side='right'
        )
    )
    fig = go.Figure(data, layout=layout)
    return fig
