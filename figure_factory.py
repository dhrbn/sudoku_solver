import os
import base64
import logging
import sys
import logging
import datetime
import numpy as np
import pandas as pd
# import plotly.graph_objects as go
import plotly.graph_objs as go


def get_classic_layout():
    return dict(
        template="plotly_white",
        annotations=[],
        showlegend=False,
        title_x=0.5,
        images=[],
        font=dict(
            family='Montserrat, Century gothic',
        )
    )


def get_rotated_image_figure(si):
    layout = get_classic_layout()
    gos = []

    im_orig_go = go.Image(
        z=si.im_orig,
        xaxis='x',
        yaxis='y',
    )

    gos.append(im_orig_go)

    layout['annotations'].append(
        dict(
            text=f'<b>Original image</b>',
            font=dict(
                size=16,
                family='century gothic',
                color='black',
            ),
            x=0.25,
            y=1,
            showarrow=False,
            xref='paper',
            yref='paper',
            align='center',
            xanchor='center',
            yanchor='bottom',
        )
    )
    im_rotated_go = go.Image(
        z=si.im,
        xaxis='x2',
        yaxis='y2',
    )

    gos.append(im_rotated_go)

    layout['annotations'].append(
        dict(
            text=f'<b>Corrected image</b>',
            font=dict(
                size=16,
                family='century gothic',
                color='black',
            ),
            x=0.75,
            y=1,
            showarrow=False,
            xref='paper',
            yref='paper',
            align='center',
            xanchor='center',
            yanchor='bottom',
        )
    )

    layout[f'xaxis'] = dict(
        anchor=f'y',
        domain=[0.1, 0.4],
        showticklabels=False,
    )
    layout[f'yaxis'] = dict(
        anchor=f'x',
        domain=[0.2, 1],
        showticklabels=False,
    )
    layout[f'xaxis2'] = dict(
        anchor=f'y2',
        domain=[0.6, 0.9],
        showticklabels=False,
    )
    layout[f'yaxis2'] = dict(
        anchor=f'x2',
        domain=[0.2, 1],
        showticklabels=False,
    )
    layout['title'] = dict(
        font=dict(size=18),
        text='Image orientation correction',
    )

    return go.Figure(data=gos, layout=layout)


def get_grid_lines_figure(si):
    layout = get_classic_layout()
    gos = []

    im_rotated_go = go.Image(
        z=si.im,
        xaxis='x',
        yaxis='y',
    )

    gos.append(im_rotated_go)

    for v_line in si.v_lines:
        line_go = go.Scatter(
            x=[v_line[0], v_line[1]],
            y=[v_line[2], v_line[3]],
            xaxis='x',
            yaxis='y',
            line=dict(
                color='red',
            )
        )
        gos.append(line_go)

    for h_line in si.h_lines:
        line_go = go.Scatter(
            x=[h_line[0], h_line[1]],
            y=[h_line[2], h_line[3]],
            xaxis='x',
            yaxis='y',
            line=dict(
                color='#ddb0f0',
            )
        )
        gos.append(line_go)

    layout[f'xaxis'] = dict(
        anchor=f'y',
        domain=[0.1, 0.9],
        range=[0, si.width],
        showticklabels=False,
    )
    layout[f'yaxis'] = dict(
        anchor=f'x',
        domain=[0.2, 1],
        range=[si.height, 0],
        showticklabels=False,
    )
    layout['title'] = dict(
        font=dict(size=18),
        text='Grid lines detection',
    )

    return go.Figure(data=gos, layout=layout)


def get_numbers_detection(si):
    layout = get_classic_layout()
    gos = []

    im_rotated_go = go.Image(
        z=si.im,
        xaxis='x',
        yaxis='y',
    )

    gos.append(im_rotated_go)

    layout['annotations'].append(
        dict(
            text=f'<b>Corrected image</b>',
            font=dict(
                size=16,
                family='century gothic',
                color='black',
            ),
            x=0.25,
            y=1,
            showarrow=False,
            xref='paper',
            yref='paper',
            align='center',
            xanchor='center',
            yanchor='bottom',
        )
    )

    layout[f'xaxis'] = dict(
        anchor=f'y',
        domain=[0.1, 0.4],
        showticklabels=False,
    )
    layout[f'yaxis'] = dict(
        anchor=f'x',
        domain=[0.2, 1],
        showticklabels=False,
    )

    for i in range(0, 10):

        width = 6 if i in [0, 3, 6, 9] else 2

        line_go = go.Scatter(
            x=[i, i],
            y=[0, 9],
            xaxis='x2',
            yaxis='y2',
            line=dict(
                color='black',
                width=width,
            )
        )
        gos.append(line_go)

        line_go = go.Scatter(
            x=[0, 9],
            y=[i, i],
            xaxis='x2',
            yaxis='y2',
            line=dict(
                color='black',
                width=width,
            )
        )
        gos.append(line_go)

    for idx, i_row in enumerate(si.sudoku_df.iterrows()):
        i, row = i_row
        for c in si.sudoku_df.columns:
            if si.sudoku_df.at[i, c] is not None and not np.isnan(si.sudoku_df.at[i, c]):
                layout['annotations'].append(
                    dict(
                        text=f'<b>{si.sudoku_df.at[i, c]:.0f}</b>',
                        font=dict(
                            size=20,
                            family='century gothic',
                            color='blue',
                        ),
                        x=c + 0.5,
                        y=8 - idx + 0.5,
                        showarrow=False,
                        xref='x2',
                        yref='y2',
                        align='center',
                        xanchor='center',
                        yanchor='middle',
                    )
                )

    layout['annotations'].append(
        dict(
            text=f'<b>Detected grid</b>',
            font=dict(
                size=16,
                family='century gothic',
                color='black',
            ),
            x=0.75,
            y=1,
            showarrow=False,
            xref='paper',
            yref='paper',
            align='center',
            xanchor='center',
            yanchor='bottom',
        )
    )
    layout[f'xaxis2'] = dict(
        anchor=f'y2',
        domain=[0.6, 0.9],
        range=[-0.04, 9.04],
        showticklabels=False,
    )
    layout[f'yaxis2'] = dict(
        anchor=f'x2',
        domain=[0.2, 1],
        range=[-0.04, 9.04],
        showticklabels=False,
    )
    layout['title'] = dict(
        font=dict(size=18),
        text='Numbers detection',
    )

    return go.Figure(data=gos, layout=layout)


def get_grid_filling(sdk):
    layout = get_classic_layout()
    gos = []

    for i in range(0, 10):
        width = 6 if i in [0, 3, 6, 9] else 2

        line_go = go.Scatter(
            x=[i, i],
            y=[0, 9],
            xaxis='x',
            yaxis='y',
            line=dict(
                color='black',
                width=width,
            )
        )
        gos.append(line_go)

        line_go = go.Scatter(
            x=[0, 9],
            y=[i, i],
            xaxis='x',
            yaxis='y',
            line=dict(
                color='black',
                width=width,
            )
        )
        gos.append(line_go)

    for idx, i_row in enumerate(sdk.df.iterrows()):
        i, row = i_row
        for c in sdk.df.columns:
            if sdk.gold_df.at[i, c] and sdk.df.at[i, c] is not None and not np.isnan(sdk.df.at[i, c]):
                layout['annotations'].append(
                    dict(
                        text=f'<b>{sdk.df.at[i, c]:.0f}</b>',
                        font=dict(
                            size=20,
                            family='century gothic',
                            color='blue',
                        ),
                        x=c + 0.5,
                        y=8 - idx + 0.5,
                        showarrow=False,
                        xref='x',
                        yref='y',
                        align='center',
                        xanchor='center',
                        yanchor='middle',
                    )
                )

    layout['annotations'].append(
        dict(
            text=f'<b>Detected grid</b>',
            font=dict(
                size=16,
                family='century gothic',
                color='black',
            ),
            x=0.25,
            y=1,
            showarrow=False,
            xref='paper',
            yref='paper',
            align='center',
            xanchor='center',
            yanchor='bottom',
        )
    )

    layout[f'xaxis'] = dict(
        anchor=f'y',
        domain=[0.1, 0.4],
        range=[-0.04, 9.04],
        showticklabels=False,
    )
    layout[f'yaxis'] = dict(
        anchor=f'x',
        domain=[0.2, 1],
        range=[-0.04, 9.04],
        showticklabels=False,
    )

    for i in range(0, 10):
        width = 6 if i in [0, 3, 6, 9] else 2

        line_go = go.Scatter(
            x=[i, i],
            y=[0, 9],
            xaxis='x2',
            yaxis='y2',
            line=dict(
                color='black',
                width=width,
            )
        )
        gos.append(line_go)

        line_go = go.Scatter(
            x=[0, 9],
            y=[i, i],
            xaxis='x2',
            yaxis='y2',
            line=dict(
                color='black',
                width=width,
            )
        )
        gos.append(line_go)

    for idx, i_row in enumerate(sdk.df.iterrows()):
        i, row = i_row
        for c in sdk.df.columns:
            if sdk.df.at[i, c] is not None and not np.isnan(sdk.df.at[i, c]):
                val = sdk.df.at[i, c]
                if sdk.gold_df.at[i, c]:
                    val = f"<b>{val:.0f}</b>"
                    color = 'blue'
                else:
                    val = f"{val:.0f}"
                    color = 'black'

                layout['annotations'].append(
                    dict(
                        text=val,
                        font=dict(
                            size=20,
                            family='century gothic',
                            color=color,
                        ),
                        x=c + 0.5,
                        y=8 - idx + 0.5,
                        showarrow=False,
                        xref='x2',
                        yref='y2',
                        align='center',
                        xanchor='center',
                        yanchor='middle',
                    )
                )

    layout['annotations'].append(
        dict(
            text=f'<b>Filled grid</b>',
            font=dict(
                size=16,
                family='century gothic',
                color='black',
            ),
            x=0.75,
            y=1,
            showarrow=False,
            xref='paper',
            yref='paper',
            align='center',
            xanchor='center',
            yanchor='bottom',
        )
    )
    layout[f'xaxis2'] = dict(
        anchor=f'y2',
        domain=[0.6, 0.9],
        range=[-0.04, 9.04],
        showticklabels=False,
    )
    layout[f'yaxis2'] = dict(
        anchor=f'x2',
        domain=[0.2, 1],
        range=[-0.04, 9.04],
        showticklabels=False,
    )
    layout['title'] = dict(
        font=dict(size=18),
        text='Grid filling',
    )

    return go.Figure(data=gos, layout=layout)
