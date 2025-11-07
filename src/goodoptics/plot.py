import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def complex_surface(da:xr.DataArray, title_1 = "Real", title_2 = "Imaginary", xlim=None, ylim=None):

    X, Y = np.meshgrid(da.x, da.y, indexing="ij")
    Z1 = np.real(da)
    Z2 = np.imag(da)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=(title_1, title_2)
    )

    fig.add_trace(go.Surface(z=Z1, x=X, y=Y, colorscale="Viridis"), row=1, col=1)
    fig.add_trace(go.Surface(z=Z2, x=X, y=Y, colorscale="Cividis"), row=1, col=2)

    fig.update_layout(
        width=1000, height=500,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig

def real_surface(da:xr.DataArray, title = "", cmap="Viridis"):
    X, Y = np.meshgrid(da.x, da.y, indexing="ij")
    Z = da.values

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "surface"}]],
        subplot_titles=(title)
    )

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=cmap), row=1, col=1)

    fig.update_layout(
        width=1000, height=500,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig