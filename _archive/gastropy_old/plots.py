# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import List, Union

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Panel, RangeTool, Tabs
from bokeh.plotting import figure


def plot_raw(
    signal: Union[np.array, List[np.array]], sfreq: int = 1000, figsize: int = 400
) -> Tabs:
    """Plot raw EGG recording.

    Parameters
    ----------
    signal : list or np.ndarray
        The EGG signal as a Numpy array. If a list is provided, it should
        contain *n* Numpy array of equal length.
    sfreq : int
        Signal sampling frequency. Default is set to 1000 Hz.
    figsize : int
        Figure height. Default is `300`.

    Returns
    -------
    fig : :class:`bokeh.models.Tabs`
        The bokeh figure containing the plot(s).

    """

    if isinstance(signal, list):
        for i, sig in enumerate(signal):
            if i == 0:
                # Create the time vector
                time = pd.to_datetime(
                    np.arange(0, len(sig)) / sfreq, unit="s", origin="unix"
                )
                data = {"time": time}
            data[f"EGG_{i+1}"] = sig
    elif isinstance(signal, np.ndarray):
        # Create the time vector
        time = pd.to_datetime(
            np.arange(0, len(signal)) / sfreq, unit="s", origin="unix"
        )
        data = {"time": time}
        data["EGG_1"] = signal
    else:
        raise ValueError(
            "Invalid data format provided. Should be list or 1d Numpy array"
        )

    source = ColumnDataSource(data=data)
    tabs = []
    for i in range(1, len(data)):
        raw = figure(
            title="Raw data",
            x_axis_type="datetime",
            sizing_mode="stretch_width",
            plot_height=figsize,
            x_axis_label="Time",
            y_axis_label="EGG",
            output_backend="webgl",
            x_range=(time[0], time[-1]),
        )

        raw.line("time", f"EGG_{i}", source=source)

        select = figure(
            title="Drag the middle and edges of the selection box to change the range above",
            plot_height=130,
            plot_width=800,
            y_range=raw.y_range,
            x_axis_type="datetime",
            y_axis_type=None,
            tools="",
            toolbar_location=None,
            background_fill_color="#efefef",
        )
        range_tool = RangeTool(x_range=raw.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2

        select.line("time", f"EGG_{i}", source=source)
        select.ygrid.grid_line_color = None
        select.add_tools(range_tool)
        select.toolbar.active_multi = range_tool

        tabs.append(
            Panel(
                child=column(*(raw, select), sizing_mode="stretch_width"),
                title=f"EGG_{i}",
            )
        )

    return Tabs(tabs=tabs)
