import sympy as sp
import numpy as np
import xarray as xr
import math
from functools import cached_property
import plot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

class zernike:
    def __init__(self, n:int, m:Optional[int]=None):
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

    @cached_property
    def _lambdified(self):
        """Cached compiled version of the Sympy expression."""
        rho, theta = sp.symbols("rho theta", real=True)
        return sp.lambdify((rho, theta), self.expression, "numpy")

    def _compute_sample(self):
        """Compute the raw numeric sample (NumPy array)."""
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
        """Return raw numeric Zernike sample."""
        if recalc or self._sample_cache is None:
            self._sample_cache = self._compute_sample()
        return self._sample_cache

    def sample_da(self, N=None, radius=1, recalc=False):
        """Return sampled DataArray with coordinates and metadata."""
        if N is not None and N != self._N:
            self.N = N
        if recalc or self._sample_da_cache is None:
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
        return self._sample_da_cache

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot(self, cmap="RdBu"):
        img = plt.imshow(self.sample(), cmap=cmap)
        return img

    def plot_surface(self, cmap="RdBu"):
        return plot.real_surface(self.sample_da(), cmap=cmap)

    # ------------------------------------------------------------------
    # Property access and cache invalidation
    # ------------------------------------------------------------------

    def _invalidate_cache(self):
        self._sample_cache = None
        self._sample_da_cache = None

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        if value != self._N:
            self._N = value
            self._invalidate_cache()

    @property
    def endpoint(self):
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


class zernike_aberration():
    """
    Represents linear combinations of zernikes and corresponding tools.
    """
    _empty_coefficients = xr.DataArray(
            [],
            coords={"zernike":[], "n":("zernike", []), "m":("zernike", [])}, 
            dims=('zernike',)
    )

    def __init__(self, zernike_coefficients:dict, N=400):
        self.coefficients = self._map_to_xr_coefficients(zernike_coefficients).sortby("zernike")
        self.zernikes = [zernike(i) for i in self.coefficients.zernike]
        self._N = N # start using
        self._set_N_on_zernikes()
        self._cached_sample = None
        self._sample_da_cache = None

    def _set_N_on_zernikes(self):
        for z in self.zernikes:
            z.N = self._N

    def sample(self, recalc_full_depth=False, *args, **kwargs):
        if recalc_full_depth or self._cached_sample is None:
            samples = np.array([z.sample(recalc=recalc_full_depth, *args, **kwargs) for z in self.zernikes])
            self._cached_sample = np.tensordot(self.coefficients, samples, axes=1)
        return self._cached_sample

    def sample_da(self, recalc_full_depth=False, *args, **kwargs):
        if recalc_full_depth or self._sample_da_cache is None:
            das = [z.sample_da(recalc=recalc_full_depth, **kwargs).assign_coords({'zernike':z.j}) for z in self.zernikes]
            self._sample_da_cache = (self.coefficients * xr.concat(das, dim="zernike")).sum("zernike")
        return self._sample_da_cache

    def plot(self, cmap="RdBu"):
        img = plt.imshow(self.sample(), cmap=cmap)
        return img

    def plot_surface(self, cmap="RdBu"):
        return plot.real_surface(self.sample_da(), cmap=cmap)

    def _map_to_xr_coefficients(self, zernike_coefficients):
        if isinstance(zernike_coefficients, xr.DataArray):
            return zernike_coefficients
        if len(zernike_coefficients) == 0:
            return self._empty_coefficients
        indices = list(zernike_coefficients.keys())
        if isinstance(indices[0], tuple):
            _indices = indices
            _fringe_indices = [ nm_to_fringe(*i) for i in indices]
        else:
            _indices = [fringe_to_nm(i) for i in indices]
            _fringe_indices = indices
        n_lst = [i[0] for i in _indices]
        m_lst = [i[1] for i in _indices]
        return xr.DataArray(
            list(zernike_coefficients.values()),
            coords={"zernike":_fringe_indices, "n":("zernike", n_lst), "m":("zernike", m_lst)}, 
            dims=('zernike',))

    def _determine_zernike_sum(self, zernike_coefficients):
        if isinstance(zernike_coefficients, dict):
            new_coefficients = self._map_to_xr_coefficients(zernike_coefficients)
        elif isinstance(zernike_coefficients, xr.DataArray):
            new_coefficients = zernike_coefficients
        old, new = xr.align(self.coefficients, new_coefficients, join="outer", fill_value=0)
        newly_added_indices = list(set(new.zernike.values).difference(set(self.coefficients.zernike.values)))
        return (old + new).sortby('zernike'), newly_added_indices

    def add_zernikes(self, zernike_coefficients):
        self.coefficients, newly_added_indices = self._determine_zernike_sum(zernike_coefficients)
        self._invalidate_cache() # This leaves the cache per zernike intact
        self.zernikes += [zernike(i) for i in newly_added_indices]
        self._set_N_on_zernikes()
        self.zernikes.sort()

    @property
    def N(self):
        return self._N
    
    @N.setter
    def N(self, value):
        if value != self._N:
            self._N = value
            self._invalidate_cache()
            for z in self.zernikes:
                z.N = value

    def _invalidate_cache(self):
        self._cached_sample = None
        self._sample_da_cache = None

    def _repr_html_(self):
        return self.coefficients._repr_html_()

    def __add__(self, other):
        if not isinstance(other, zernike_aberration):
            return NotImplemented
        return zernike_aberration(self._determine_zernike_sum(other.coefficients)[0], N=self._N) # Don't use the cache?

    def __sub__(self, other):
        if not isinstance(other, zernike_aberration):
            return NotImplemented
        return zernike_aberration(self._determine_zernike_sum(-other.coefficients)[0], N=self._N) # Don't reuse cache?

def fit_zernikes_to_da(da: xr.DataArray, max_j: Optional[int]=None, indeces:Optional[list]=None) -> zernike_aberration:
    """
    Fit Zernike coefficients to a given DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray to fit.
    max_j : int, optional
        Max fringe index to include in the fit.
    indeces : list of int, optional
        Specific Zernike fringe indices to include in the fit.

    Returns
    -------
    zernike_aberration
        The fitted Zernike aberration instance.
    """
    if indeces is None:
        if max_j is None:
            raise ValueError("Either max_j or indeces must be provided.")
        indeces = list(range(1, max_j + 1))

    if da.ndim != 2:
        raise ValueError("Input DataArray must be 2-dimensional.")
    N = da.sizes['x']
    if da.sizes['y'] != N:
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
    rho = sp.symbols("rho", real=True) 
    theta = sp.symbols("theta", real=True)
    if abs(m) > n or (n - abs(m)) % 2 != 0:
        return 0  # undefined
    k = sp.symbols('k', integer=True, nonnegative=True)
    radial = sp.summation(
        (-1)**k * sp.factorial(n - k) /
        (sp.factorial(k) * sp.factorial((n + abs(m))//2 - k) * sp.factorial((n - abs(m))//2 - k)) *
        rho**(n - 2*k),
        (k, 0, (n - abs(m))//2)
    )
    if m >= 0:
        return radial * sp.cos(m * theta)
    else:
        return radial * sp.sin(-m * theta)
    
def nm_to_fringe(n, m):
    if (n-m) % 2 == 0:
        return int((1+(n+abs(m))//2)**2 - 2*abs(m) + np.floor((1-np.sign(m))/2))

def fringe_to_nm(j: int):

    if j < 1:
        raise ValueError("Fringe index must be >= 1")

    # Determine block index (a) and position within block (r)
    a = math.floor(math.sqrt(j - 1)) + 1
    r = j - (a - 1)**2  # position in the block

    # Case 1: last term in block → m = 0
    if r == 2 * a - 1:
        m = 0
        n = 2 * (a - 1)
        return n, m

    # Case 2: alternating signs, decreasing |m|
    k = math.ceil(r / 2)
    m_abs = a - k

    if r % 2 == 1:
        m = +m_abs
    else:
        m = -m_abs

    n = 2 * (a - 1) - m_abs
    return n, m




def plot_zernike_triangle(n_max=5, N=300, cmap="RdBu"):
    """
    Create an interactive Plotly figure showing a Zernike triangle up to n_max.

    Each subplot shows Z_n^m sampled on a Cartesian grid.

    Parameters
    ----------
    n_max : int
        Maximum radial order (n) to display.
    N : int
        Sampling grid resolution.
    cmap : str
        Colormap name (e.g., 'RdBu', 'Viridis', etc.)
    """
    # Estimate the number of valid m-values per n
    nrows = n_max + 1
    ncols = n_max + 1

    fig = make_subplots(
        rows=nrows, cols=ncols,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        subplot_titles=[],
    )

    vmax = 1  # fixed symmetric color scale for consistency

    for n in range(n_max + 1):
        valid_ms = [m for m in range(-n, n+1, 2) if (n - abs(m)) % 2 == 0]
        for m in valid_ms:
            # Compute subplot position
            row = n + 1
            col = (m + n) // 2 + 1  # shift to non-negative index range

            z = zernike(n, m)
            da = z.sample_da(N)
            Z = np.array(da)

            # Add image heatmap
            fig.add_trace(
                go.Heatmap(
                    z=Z,
                    colorscale=cmap,
                    zmin=-vmax, zmax=vmax,
                    showscale=False,
                    hovertemplate=f"n={n}, m={m}<extra></extra>",
                ),
                row=row, col=col
            )

            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

    # Hide empty subplots
    for n in range(n_max + 1):
        for col in range(n_max + 1):
            m = 2*col - n
            if m < -n or m > n or (n - abs(m)) % 2 != 0:
                fig.update_xaxes(visible=False, row=n+1, col=col+1)
                fig.update_yaxes(visible=False, row=n+1, col=col+1)

    fig.update_layout(
        title=f"Interactive Zernike Triangle up to n={n_max}",
        width=200*(n_max+1),
        height=200*(n_max+1),
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig

def plotly_zernike_pyramid(n_max=4, N=120, cmap="RdBu"):
    """
    Create a 3D 'Zernike pyramid' — stacked surface plots for Z_n^m up to n_max.
    Just because ChatGPT really liked the idea :).

    Parameters
    ----------
    n_max : int
        Maximum radial order (n).
    N : int
        Sampling grid resolution per Zernike.
    cmap : str
        Plotly colormap (e.g. 'RdBu', 'Viridis', 'Picnic', etc.)
    """
    surfaces = []
    offset_scale = 2.5  # spacing between surfaces in x/y
    z_scale = 0.8       # flatten vertical height for aesthetics

    for n in range(n_max + 1):
        valid_ms = [m for m in range(-n, n + 1, 2) if (n - abs(m)) % 2 == 0]
        for m in valid_ms:
            # Sample the Zernike mode
            z_mode = zernike(n, m)
            da = z_mode.sample_da(N)
            Z = np.array(da)
            x = da.x.values
            y = da.y.values
            X, Y = np.meshgrid(x, y)

            # Position this surface in the pyramid grid
            x_offset = offset_scale * (m + n) / 2
            y_offset = -offset_scale * n

            surfaces.append(
                go.Surface(
                    z=z_scale * Z + y_offset * 0,  # flatten individual height
                    x=X + x_offset,
                    y=Y + y_offset,
                    colorscale=cmap,
                    cmin=-1, cmax=1,
                    showscale=False,
                    name=f"n={n}, m={m}",
                    hovertemplate=f"n={n}, m={m}<extra></extra>",
                )
            )

    # Combine all into one figure
    fig = go.Figure(data=surfaces)

    fig.update_layout(
        title=f"3D Zernike Pyramid (up to n={n_max})",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye={'x': 0, 'y': -1.5, 'z': 3.1})
        ),
        width=1000,
        height=900,
        template="plotly_white",
    )

    return fig