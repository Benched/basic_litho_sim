"""Plotly-based surface plotting helpers used in the notebooks."""

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def complex_surface(da: xr.DataArray, title_1="Real", title_2="Imaginary", xlim=None, ylim=None):
    """Render the real and imaginary parts of a complex surface side by side.

    Parameters
    ----------
    da : xarray.DataArray
        Two-dimensional complex-valued surface to visualize.
    title_1, title_2 : str, optional
        Titles for the real and imaginary subplots.
    xlim, ylim : tuple[float, float], optional
        Reserved for future axis limiting. The current implementation accepts
        these arguments for API compatibility but does not yet apply them.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing two surface traces.
    """
    X, Y = np.meshgrid(da.x, da.y, indexing="ij")
    Z1 = np.real(da)
    Z2 = np.imag(da)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=(title_1, title_2),
    )

    fig.add_trace(go.Surface(z=Z1, x=X, y=Y, colorscale="Viridis"), row=1, col=1)
    fig.add_trace(go.Surface(z=Z2, x=X, y=Y, colorscale="Cividis"), row=1, col=2)

    fig.update_layout(
        width=1000,
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def real_surface(da: xr.DataArray, title="", cmap="Viridis"):
    """Render a single real-valued surface with Plotly.

    Parameters
    ----------
    da : xarray.DataArray
        Two-dimensional real-valued surface to visualize.
    title : str, optional
        Plot title.
    cmap : str, optional
        Plotly colorscale name to apply to the surface.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing one surface trace.
    """
    X, Y = np.meshgrid(da.x, da.y, indexing="ij")
    Z = da.values

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "surface"}]],
        subplot_titles=(title),
    )

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=cmap), row=1, col=1)

    fig.update_layout(
        width=1000,
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig
