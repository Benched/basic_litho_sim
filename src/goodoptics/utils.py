"""Sampling utilities shared by the optics helpers and notebooks."""

import numpy as np
import xarray as xr


def to_da(x, values, xmin=None, xmax=None, endpoint=False):
    """Wrap a 1D sample in an `xarray.DataArray` with spacing metadata.

    Parameters
    ----------
    x : array-like
        Sample coordinates.
    values : array-like
        Values evaluated on `x`.
    xmin, xmax : float, optional
        Explicit domain bounds. When omitted they are inferred from `x` and the
        half-open sampling convention.
    endpoint : bool, optional
        Whether `x` includes the upper endpoint of the sampled interval.

    Returns
    -------
    xarray.DataArray
        A one-dimensional array with `dx` and `n` attributes.
    """
    da = xr.DataArray(values, dims=("x",), coords={"x": x})

    if xmin is None:
        xmin = min(x)

    if xmax is None:
        dx = x[1] - x[0]
        xmax = x[-1] + dx if endpoint is False else x[-1]

    n = len(values)
    dx = (xmax - xmin) / n

    return da.assign_attrs({"dx": dx, "n": n})


def sample_fn(f, N, t0, t1, as_da=True):
    """Sample a scalar 1D function on a uniform half-open interval.

    Parameters
    ----------
    f : callable
        Function of one variable to sample.
    N : int
        Number of sample points.
    t0, t1 : float
        Lower and upper interval bounds.
    as_da : bool, optional
        When `True`, return an `xarray.DataArray`; otherwise return raw NumPy
        arrays.

    Returns
    -------
    xarray.DataArray or tuple[np.ndarray, np.ndarray]
        Either the wrapped sample or the coordinate/value arrays.
    """
    x = np.linspace(t0, t1, N, endpoint=False)
    sample = np.array(list(map(f, x)))
    if as_da:
        return to_da(x, sample)
    return x, sample


def sample_fn_2d(fn, nx, ny, xmin, xmax, ymin, ymax, as_da=True, endpoint=False):
    """Sample a scalar 2D function over a regular Cartesian grid.

    Parameters
    ----------
    fn : callable
        Function of two variables to sample.
    nx, ny : int
        Number of grid points in the x and y directions.
    xmin, xmax : float
        Bounds of the x-domain.
    ymin, ymax : float
        Bounds of the y-domain.
    as_da : bool, optional
        When `True`, return an `xarray.DataArray`; otherwise return raw meshgrid
        arrays.
    endpoint : bool, optional
        Whether to include the upper domain bounds in the generated coordinate
        grids.

    Returns
    -------
    xarray.DataArray or tuple[np.ndarray, np.ndarray, np.ndarray]
        Either a 2D DataArray with spacing metadata or the `X`, `Y`, and `Z`
        meshgrid arrays.
    """
    x = np.linspace(xmin, xmax, nx, endpoint=endpoint)
    y = np.linspace(ymin, ymax, ny, endpoint=endpoint)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.vectorize(fn)(X, Y)

    if as_da:
        da = xr.DataArray(Z, coords={"x": x, "y": y}, dims=["x", "y"])
        da.attrs.update({"dx": dx, "dy": dy, "nx": nx, "ny": ny})
        return da
    return X, Y, Z


def riemann_approximation_ft_of_fn_on_symmetric_interval(f, N, t0):
    """Approximate a continuous Fourier transform with a Riemann sum.

    Parameters
    ----------
    f : callable
        Function to sample on the interval `[-t0, t0)`.
    N : int
        Number of samples used for the approximation.
    t0 : float
        Half-width of the symmetric sampling interval.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The sampled frequency grid and the corresponding Fourier approximation.

    Notes
    -----
    This helper is mainly used in tests as a slower but explicit reference
    implementation.
    """
    Fs = N / (2 * t0)
    dt = 1 / Fs
    n = np.array(list(range(N)))
    x = np.array([-t0 + n / Fs for n in range(N)])
    l = np.linspace(-Fs / 2, Fs / 2, N, endpoint=False)
    samples = np.array(list(map(f, x)))
    fourier_transform = (
        np.exp(2j * np.pi * l * t0)
        * np.dot(samples, np.exp(-2j * np.pi * dt * np.tensordot(n, l, 0)))
        * dt
    )
    return l, fourier_transform
