"""Zernike polynomials, aberration composition, and visualization helpers."""

import math
from functools import cached_property
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import xarray as xr
from plotly.subplots import make_subplots

from . import plot


class zernike:
    """Represent a single Zernike mode with cached numeric samples.

    Instances can be constructed either from a pair `(n, m)` or from a single
    Fringe index `j` by passing that index as the first argument and leaving
    `m=None`.
    """

    def __init__(self, n: int, m: Optional[int] = None):
        if m is None:
            j = int(n)
            n_val, m_val = fringe_to_nm(j)
            self.n = n_val
            self.m = m_val
            self.j = j
        else:
            self.n = n
            self.m = m
            self.j = nm_to_fringe(self.n, self.m)
        self.expression = zernike_expression(self.n, self.m)
        self._N = 400
        self._endpoint = False
        self._sample_cache = None
        self._sample_da_cache = None
        self._sample_da_cache_key = None

    @cached_property
    def _lambdified(self):
        """Return a NumPy-callable version of the symbolic Zernike expression."""
        rho, theta = sp.symbols("rho theta", real=True)
        return sp.lambdify((rho, theta), self.expression, "numpy")

    def _compute_sample(self):
        """Evaluate the mode on a square Cartesian grid over the unit disk."""
        x = np.linspace(-1, 1, self._N)
        y = np.linspace(-1, 1, self._N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        mask = R <= 1
        Z = np.zeros_like(R)
        Z[mask] = self._lambdified(R[mask], Theta[mask])
        return Z

    def sample(self, recalc=False):
        """Return the cached NumPy sample of the Zernike mode."""
        if recalc or self._sample_cache is None:
            self._sample_cache = self._compute_sample()
        return self._sample_cache

    def sample_da(self, N=None, radius=1, recalc=False):
        """Return the cached sample wrapped in an `xarray.DataArray`.

        Parameters
        ----------
        N : int, optional
            Override for the per-axis sample count. When provided, this also
            updates the instance's cached resolution.
        radius : float, optional
            Radius to use for the returned coordinate system.
        recalc : bool, optional
            Force recomputation of the underlying sample and wrapped DataArray.

        Returns
        -------
        xarray.DataArray
            Sampled Zernike surface with `x`, `y`, `dx`, `dy`, `nx`, and `ny`
            metadata.
        """
        if N is not None and N != self._N:
            self.N = N
        cache_key = radius
        if recalc or self._sample_da_cache is None or self._sample_da_cache_key != cache_key:
            x = np.linspace(-radius, radius, self._N, endpoint=self._endpoint)
            dx = 2 * radius / (self._N - 1 if self._endpoint else self._N)
            data = self.sample(recalc=recalc)
            da = xr.DataArray(
                data,
                coords={"x": x, "y": x},
                dims=("x", "y"),
                attrs={"dx": dx, "dy": dx, "nx": self._N, "ny": self._N},
            )
            self._sample_da_cache = da
            self._sample_da_cache_key = cache_key
        return self._sample_da_cache

    def plot(self, cmap="RdBu"):
        """Plot the sampled mode with Matplotlib's `imshow`."""
        img = plt.imshow(self.sample(), cmap=cmap)
        return img

    def plot_surface(self, cmap="RdBu"):
        """Plot the sampled mode as an interactive Plotly surface."""
        return plot.real_surface(self.sample_da(), cmap=cmap)

    def _invalidate_cache(self):
        """Clear cached numeric samples after a configuration change."""
        self._sample_cache = None
        self._sample_da_cache = None
        self._sample_da_cache_key = None

    @property
    def N(self):
        """Number of sample points per Cartesian axis."""
        return self._N

    @N.setter
    def N(self, value):
        if value != self._N:
            self._N = value
            self._invalidate_cache()

    @property
    def endpoint(self):
        """Whether returned coordinates include the outer endpoint."""
        return self._endpoint

    @endpoint.setter
    def endpoint(self, value):
        if value != self._endpoint:
            self._endpoint = value
            self._invalidate_cache()

    def __eq__(self, other):
        return (self.n == other.n) and (self.m == other.m)

    def __gt__(self, other):
        return self.j > other.j

    def __lt__(self, other):
        return self.j < other.j


class zernike_aberration:
    """Represent a linear combination of Zernike modes.

    The coefficients are stored as an `xarray.DataArray` indexed by Fringe
    number, with companion `n` and `m` coordinates for convenient inspection.
    """

    _empty_coefficients = xr.DataArray(
        [],
        coords={"zernike": [], "n": ("zernike", []), "m": ("zernike", [])},
        dims=("zernike",),
    )

    def __init__(self, zernike_coefficients: dict, N=400):
        """Create an aberration model from Zernike coefficients."""
        self.coefficients = self._map_to_xr_coefficients(zernike_coefficients).sortby("zernike")
        self.zernikes = [zernike(i) for i in self.coefficients.zernike]
        self._N = N
        self._set_N_on_zernikes()
        self._cached_sample = None
        self._sample_da_cache = None
        self._sample_da_cache_key = None

    def _set_N_on_zernikes(self):
        """Propagate the current resolution to each constituent mode."""
        for z in self.zernikes:
            z.N = self._N

    def sample(self, recalc_full_depth=False, *args, **kwargs):
        """Return the summed NumPy sample of the aberration surface."""
        if recalc_full_depth or self._cached_sample is None:
            samples = np.array(
                [z.sample(recalc=recalc_full_depth, *args, **kwargs) for z in self.zernikes]
            )
            self._cached_sample = np.tensordot(self.coefficients, samples, axes=1)
        return self._cached_sample

    def sample_da(self, recalc_full_depth=False, *args, **kwargs):
        """Return the summed aberration surface as an `xarray.DataArray`."""
        cache_key = (args, tuple(sorted(kwargs.items())))
        if recalc_full_depth or self._sample_da_cache is None or self._sample_da_cache_key != cache_key:
            das = [
                z.sample_da(recalc=recalc_full_depth, **kwargs).assign_coords({"zernike": z.j})
                for z in self.zernikes
            ]
            self._sample_da_cache = (self.coefficients * xr.concat(das, dim="zernike")).sum("zernike")
            self._sample_da_cache_key = cache_key
        return self._sample_da_cache

    def plot(self, cmap="RdBu"):
        """Plot the summed aberration with Matplotlib's `imshow`."""
        img = plt.imshow(self.sample(), cmap=cmap)
        return img

    def plot_surface(self, cmap="RdBu"):
        """Plot the summed aberration as an interactive Plotly surface."""
        return plot.real_surface(self.sample_da(), cmap=cmap)

    def _map_to_xr_coefficients(self, zernike_coefficients):
        """Normalize coefficient input into the internal `xarray` structure."""
        if isinstance(zernike_coefficients, xr.DataArray):
            return zernike_coefficients
        if len(zernike_coefficients) == 0:
            return self._empty_coefficients
        indices = list(zernike_coefficients.keys())
        if isinstance(indices[0], tuple):
            _indices = indices
            _fringe_indices = [nm_to_fringe(*i) for i in indices]
        else:
            _indices = [fringe_to_nm(i) for i in indices]
            _fringe_indices = indices
        n_lst = [i[0] for i in _indices]
        m_lst = [i[1] for i in _indices]
        return xr.DataArray(
            list(zernike_coefficients.values()),
            coords={"zernike": _fringe_indices, "n": ("zernike", n_lst), "m": ("zernike", m_lst)},
            dims=("zernike",),
        )

    def _determine_zernike_sum(self, zernike_coefficients):
        """Align and add another coefficient collection to the current one."""
        if isinstance(zernike_coefficients, dict):
            new_coefficients = self._map_to_xr_coefficients(zernike_coefficients)
        elif isinstance(zernike_coefficients, xr.DataArray):
            new_coefficients = zernike_coefficients
        old, new = xr.align(self.coefficients, new_coefficients, join="outer", fill_value=0)
        newly_added_indices = list(set(new.zernike.values).difference(set(self.coefficients.zernike.values)))
        return (old + new).sortby("zernike"), newly_added_indices

    def add_zernikes(self, zernike_coefficients):
        """Add more coefficients to the aberration in place."""
        self.coefficients, newly_added_indices = self._determine_zernike_sum(zernike_coefficients)
        self._invalidate_cache()
        self.zernikes += [zernike(i) for i in newly_added_indices]
        self._set_N_on_zernikes()
        self.zernikes.sort()

    @property
    def N(self):
        """Number of sample points per axis used for constituent Zernikes."""
        return self._N

    @N.setter
    def N(self, value):
        if value != self._N:
            self._N = value
            self._invalidate_cache()
            for z in self.zernikes:
                z.N = value

    def _invalidate_cache(self):
        """Clear cached aggregate samples after a configuration change."""
        self._cached_sample = None
        self._sample_da_cache = None
        self._sample_da_cache_key = None

    def _repr_html_(self):
        """Delegate rich HTML rendering to the coefficient DataArray."""
        return self.coefficients._repr_html_()

    def __add__(self, other):
        if not isinstance(other, zernike_aberration):
            return NotImplemented
        return zernike_aberration(self._determine_zernike_sum(other.coefficients)[0], N=self._N)

    def __sub__(self, other):
        if not isinstance(other, zernike_aberration):
            return NotImplemented
        return zernike_aberration(self._determine_zernike_sum(-other.coefficients)[0], N=self._N)


def fit_zernikes_to_da(
    da: xr.DataArray,
    max_j: Optional[int] = None,
    indeces: Optional[list] = None,
) -> zernike_aberration:
    """Fit Zernike coefficients to a square 2D sample.

    Parameters
    ----------
    da : xarray.DataArray
        Two-dimensional square sample to fit.
    max_j : int, optional
        Highest Fringe index to include. Used when `indeces` is omitted.
    indeces : list[int], optional
        Explicit Fringe indices to fit. The parameter name preserves the
        project's existing public API.

    Returns
    -------
    zernike_aberration
        Fitted aberration model with the same sampling resolution as `da`.
    """
    if indeces is None:
        if max_j is None:
            raise ValueError("Either max_j or indeces must be provided.")
        indeces = list(range(1, max_j + 1))

    if da.ndim != 2:
        raise ValueError("Input DataArray must be 2-dimensional.")
    N = da.sizes["x"]
    if da.sizes["y"] != N:
        raise ValueError("Input DataArray must be square (same size in x and y).")

    zernike_modes = [zernike(j) for j in indeces]
    if N is not None:
        for z in zernike_modes:
            z.N = N

    A = np.array([z.sample().flatten() for z in zernike_modes]).T
    b = da.values.flatten()

    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    zernike_coefficients = {j: coeffs[i] for i, j in enumerate(indeces)}
    return zernike_aberration(zernike_coefficients, N=N)


def zernike_expression(n, m):
    """Return the symbolic Zernike polynomial for radial order `n` and azimuth `m`."""
    rho = sp.symbols("rho", real=True)
    theta = sp.symbols("theta", real=True)
    if abs(m) > n or (n - abs(m)) % 2 != 0:
        return 0
    k = sp.symbols("k", integer=True, nonnegative=True)
    radial = sp.summation(
        (-1) ** k
        * sp.factorial(n - k)
        / (
            sp.factorial(k)
            * sp.factorial((n + abs(m)) // 2 - k)
            * sp.factorial((n - abs(m)) // 2 - k)
        )
        * rho ** (n - 2 * k),
        (k, 0, (n - abs(m)) // 2),
    )
    if m >= 0:
        return radial * sp.cos(m * theta)
    return radial * sp.sin(-m * theta)


def nm_to_fringe(n, m):
    """Convert Zernike `(n, m)` indices to the Fringe indexing convention."""
    if (n - m) % 2 == 0:
        return int((1 + (n + abs(m)) // 2) ** 2 - 2 * abs(m) + np.floor((1 - np.sign(m)) / 2))


def fringe_to_nm(j: int):
    """Convert a Fringe index into the corresponding `(n, m)` pair."""
    if j < 1:
        raise ValueError("Fringe index must be >= 1")

    a = math.floor(math.sqrt(j - 1)) + 1
    r = j - (a - 1) ** 2

    if r == 2 * a - 1:
        m = 0
        n = 2 * (a - 1)
        return n, m

    k = math.ceil(r / 2)
    m_abs = a - k

    if r % 2 == 1:
        m = +m_abs
    else:
        m = -m_abs

    n = 2 * (a - 1) - m_abs
    return n, m


def plot_zernike_triangle(n_max=5, N=300, cmap="RdBu"):
    """Create a grid of heatmaps showing Zernike modes up to order `n_max`."""
    nrows = n_max + 1
    ncols = n_max + 1

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        subplot_titles=[],
    )

    vmax = 1

    for n in range(n_max + 1):
        valid_ms = [m for m in range(-n, n + 1, 2) if (n - abs(m)) % 2 == 0]
        for m in valid_ms:
            row = n + 1
            col = (m + n) // 2 + 1

            z = zernike(n, m)
            da = z.sample_da(N)
            Z = np.array(da)

            fig.add_trace(
                go.Heatmap(
                    z=Z,
                    colorscale=cmap,
                    zmin=-vmax,
                    zmax=vmax,
                    showscale=False,
                    hovertemplate=f"n={n}, m={m}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

    for n in range(n_max + 1):
        for col in range(n_max + 1):
            m = 2 * col - n
            if m < -n or m > n or (n - abs(m)) % 2 != 0:
                fig.update_xaxes(visible=False, row=n + 1, col=col + 1)
                fig.update_yaxes(visible=False, row=n + 1, col=col + 1)

    fig.update_layout(
        title=f"Interactive Zernike Triangle up to n={n_max}",
        width=200 * (n_max + 1),
        height=200 * (n_max + 1),
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def plotly_zernike_pyramid(n_max=4, N=120, cmap="RdBu"):
    """Create a 3D arrangement of stacked Zernike surface plots."""
    surfaces = []
    offset_scale = 2.5
    z_scale = 0.8

    for n in range(n_max + 1):
        valid_ms = [m for m in range(-n, n + 1, 2) if (n - abs(m)) % 2 == 0]
        for m in valid_ms:
            z_mode = zernike(n, m)
            da = z_mode.sample_da(N)
            Z = np.array(da)
            x = da.x.values
            y = da.y.values
            X, Y = np.meshgrid(x, y)

            x_offset = offset_scale * (m + n) / 2
            y_offset = -offset_scale * n

            surfaces.append(
                go.Surface(
                    z=z_scale * Z + y_offset * 0,
                    x=X + x_offset,
                    y=Y + y_offset,
                    colorscale=cmap,
                    cmin=-1,
                    cmax=1,
                    showscale=False,
                    name=f"n={n}, m={m}",
                    hovertemplate=f"n={n}, m={m}<extra></extra>",
                )
            )

    fig = go.Figure(data=surfaces)

    fig.update_layout(
        title=f"3D Zernike Pyramid (up to n={n_max})",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye={"x": 0, "y": -1.5, "z": 3.1}),
        ),
        width=1000,
        height=900,
        template="plotly_white",
    )

    return fig
