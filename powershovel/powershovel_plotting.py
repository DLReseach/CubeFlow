import plotly.graph_objects as go
from plotly import tools
import numpy as np

def plotly_2d_histograms(H1, H2, own_errors, opponent_errors, max_z_value, metric, bin_name, bin_print_name, own_name, opponent_name):
    x_axis_title = 'log E (E / GeV)' if bin_print_name == 'true_primary_energy' else 'Event length'
    bin_mids = np.sort(own_errors[bin_name + '_mid'].unique())
    bins = np.sort(own_errors[bin_name].unique())
    trace1 = go.Heatmap(
        x=H1[2],
        y=H1[1],
        z=H1[0],
        zmin=0,
        zmax=max_z_value,
        showscale=True,
        xaxis='x1',
        yaxis='y1',
        coloraxis = 'coloraxis',
        text = ['A', 'B', 'C']
    )

    trace2 = go.Heatmap(
        x=H2[2],
        y=H2[1],
        z=H2[0],
        zmin=0,
        zmax=max_z_value,
        showscale=False,
        xaxis='x2',
        yaxis='y1',
        coloraxis = 'coloraxis'
    )

    trace3 = go.Scatter(
        x=bin_mids,
        y=[own_errors[own_errors[bin_name] == ibin][metric].quantile(0.16) for ibin in bins],
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
        x=bin_mids,
        y=[own_errors[own_errors[bin_name] == ibin][metric].quantile(0.5) for ibin in bins],
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
        x=bin_mids,
        y=[own_errors[own_errors[bin_name] == ibin][metric].quantile(0.84) for ibin in bins],
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
        x=bin_mids,
        y=[opponent_errors[opponent_errors[bin_name] == ibin][metric].quantile(0.16) for ibin in bins],
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
        x=bin_mids,
        y=[opponent_errors[opponent_errors[bin_name] == ibin][metric].quantile(0.5) for ibin in bins],
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
        x=bin_mids,
        y=[opponent_errors[opponent_errors[bin_name] == ibin][metric].quantile(0.84) for ibin in bins],
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
        x=bin_mids,
        y=[own_errors[own_errors[bin_name] == ibin][metric].quantile(0.16) for ibin in bins],
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
        x=bin_mids,
        y=[own_errors[own_errors[bin_name] == ibin][metric].quantile(0.5) for ibin in bins],
        mode='lines',
        line=dict(
            color='Red'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace11 = go.Scatter(
        x=bin_mids,
        y=[own_errors[own_errors[bin_name] == ibin][metric].quantile(0.84) for ibin in bins],
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
        x=bin_mids,
        y=[opponent_errors[opponent_errors[bin_name] == ibin][metric].quantile(0.16) for ibin in bins],
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
        x=bin_mids,
        y=[opponent_errors[opponent_errors[bin_name] == ibin][metric].quantile(0.5) for ibin in bins],
        mode='lines',
        line=dict(
            color='Green'
        ),
        xaxis='x2',
        yaxis='y1',
        showlegend=False
    )
    trace14 = go.Scatter(
        x=bin_mids,
        y=[opponent_errors[opponent_errors[bin_name] == ibin][metric].quantile(0.84) for ibin in bins],
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

    fig['layout']['xaxis1'].update(title=x_axis_title, range=[0.5, 3.0])
    fig['layout']['xaxis2'].update(title=x_axis_title, range=[0.5, 3.0])
    fig['layout']['yaxis1'].update(title='Error')
    fig['layout']['yaxis2'].update(title='Error')
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
                text='Events: {}'.format(int(np.sum(H1[0]))),
                xref='paper',
                yref='paper',
                bgcolor='white'
            ),
            dict(
                x=0.99,
                y=0.99,
                showarrow=False,
                text='Events: {}'.format(int(np.sum(H2[0]))),
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


def plotly_error_comparison(own_widths, opponent_widths, own_errors, opponent_errors, metric, bin_name, bin_print_name, bin_no, bin_index, own_name, opponent_name):
    bin_mids = np.sort(own_errors[bin_name + '_mid'].unique())
    bin_widths = np.sort(own_errors[bin_name + '_width'])
    bin_lengths = [ibin * 2 for ibin in bin_widths]
    bin_counts = own_errors.groupby([bin_name]).count()[metric].values
    histogram_colors = ['grey'] * len(bin_mids)
    histogram_colors[bin_index] = 'red'
    trace1 = go.Bar(
        x=bin_mids,
        y=bin_counts,
        width=bin_lengths,
        opacity=0.2,
        marker_color=histogram_colors,
        xaxis='x1',
        yaxis='y2',
        name='test data',
        showlegend=False
    )

    trace2 = go.Scatter(
        x=bin_mids,
        y=own_widths,
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
        y=opponent_widths,
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

    trace4 = go.Histogram(
        x=own_errors[own_errors[bin_name] == bin_no][metric].values,
        xaxis='x2',
        yaxis='y3',
        name=own_name + ' error',
        histnorm='probability',
        opacity=0.5,
        marker_color='Red',
        showlegend=False
    )
    trace5 = go.Histogram(
        x=opponent_errors[opponent_errors[bin_name] == bin_no][metric].values,
        xaxis='x2',
        yaxis='y3',
        name=opponent_name + ' error',
        histnorm='probability',
        opacity=0.5,
        marker_color='Green',
        showlegend=False
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        xaxis=dict(
            title=bin_print_name,
            domain=[0, 0.6],
            range=[0, 3.105]
        ),
        xaxis2=dict(
            title='Error',
            domain=[0.7, 1]
        ),
        yaxis=dict(
            title='Resolution',
            range=[0, 4]
        ),
        yaxis2=dict(
            title='Events',
            type='log',
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='Density',
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
                text='Events: {}'.format(np.sum(bin_counts)),
                xref='paper',
                yref='paper',
                bgcolor='white'
            ),
            dict(
                x=0.99,
                y=0.99,
                showarrow=False,
                text='Events: {}'.format(len(own_errors[own_errors[bin_name] == bin_no][metric].values)),
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
                x0=own_errors[own_errors[bin_name] == bin_no][metric].quantile(0.16),
                y0=0,
                x1=own_errors[own_errors[bin_name] == bin_no][metric].quantile(0.16),
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
                x0=own_errors[own_errors[bin_name] == bin_no][metric].quantile(0.84),
                y0=0,
                x1=own_errors[own_errors[bin_name] == bin_no][metric].quantile(0.84),
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
                x0=opponent_errors[opponent_errors[bin_name] == bin_no][metric].quantile(0.16),
                y0=0,
                x1=opponent_errors[opponent_errors[bin_name] == bin_no][metric].quantile(0.16),
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
                x0=opponent_errors[opponent_errors[bin_name] == bin_no][metric].quantile(0.84),
                y0=0,
                x1=opponent_errors[opponent_errors[bin_name] == bin_no][metric].quantile(0.84),
                y1=1,
                name=opponent_name + ' 84\%',
                line=dict(
                    color='Green',
                    width=1,
                    dash='dot'
                ),
            )
    return fig
