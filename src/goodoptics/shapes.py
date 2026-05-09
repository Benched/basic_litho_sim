"""Analytical shapes and sampled reference patterns for optics experiments."""

import numpy as np
import xarray as xr

from .utils import sample_fn, sample_fn_2d


def tophat_fn(x0, x1):
    """Return a 1D top-hat function with open interval support.

    The returned function evaluates to `1` when `x0 < x < x1` and to `0`
    otherwise.
    """
    return lambda x: int(x0 < x < x1)


def tophat_fn_2d(x0, x1, y0, y1):
    """Return a rectangular 2D top-hat function on an open interior region."""
    return lambda x, y: int(x0 < x < x1 and y0 < y < y1)


def gaussian_fn_2d(x, y, sigma_x=1.0, sigma_y=1.0, normalization=1.0):
    """Evaluate a centered anisotropic 2D Gaussian.

    Parameters
    ----------
    x, y : array-like
        Coordinates at which to evaluate the Gaussian.
    sigma_x, sigma_y : float, optional
        Standard deviations along the x and y axes.
    normalization : float, optional
        Desired total integral of the Gaussian.

    Returns
    -------
    numpy.ndarray
        Gaussian values evaluated at the supplied coordinates.
    """
    prefactor = normalization / (2.0 * np.pi * sigma_x * sigma_y)
    exponent = -0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2)
    return prefactor * np.exp(exponent)


def circular_tophat_fn_2d(radius, center_x=0.0, center_y=0.0):
    """Return a circular 2D top-hat function with an open boundary."""
    return lambda x, y: int(((x - center_x) ** 2 + (y - center_y) ** 2) < (radius ** 2))


def quadratic_decay_fn():
    """Return the 1D reference function `f(x) = 1 / (1 + x**2)`."""
    return lambda x: 1 / (1 + x**2)


def top_hat_da(n, x0, x1, xmin, xmax):
    """Sample a 1D top-hat function on a regular grid."""
    return sample_fn(tophat_fn(x0, x1), n, xmin, xmax)


def quadratic_decay_da(n, xmin, xmax):
    """Sample the 1D quadratic-decay reference function on a regular grid."""
    return sample_fn(quadratic_decay_fn(), n, xmin, xmax)


def top_hat_2d_da(nx, ny, x0, x1, y0, y1, xmin, xmax, ymin, ymax):
    """Sample a rectangular 2D top-hat function over a Cartesian grid.

    Parameters
    ----------
    nx, ny : int
        Number of grid points along x and y.
    x0, x1 : float
        Horizontal bounds of the top-hat region.
    y0, y1 : float
        Vertical bounds of the top-hat region.
    xmin, xmax : float
        Bounds of the sampled x-domain.
    ymin, ymax : float
        Bounds of the sampled y-domain.

    Returns
    -------
    xarray.DataArray
        Binary 2D sample with `x`, `y`, `dx`, `dy`, `nx`, and `ny` metadata.
    """
    fn = tophat_fn_2d(x0, x1, y0, y1)
    return sample_fn_2d(fn, nx, ny, xmin, xmax, ymin, ymax, as_da=True)


def circular_top_hat_da(nx, ny, radius, xmin, xmax, ymin, ymax, center_x=0.0, center_y=0.0):
    """Sample a circular top-hat function over a Cartesian grid."""
    fn = circular_tophat_fn_2d(radius, center_x=center_x, center_y=center_y)
    return sample_fn_2d(fn, nx, ny, xmin, xmax, ymin, ymax, as_da=True)


def quadratic_decay_2d_da(nx, ny, xmin, xmax, ymin, ymax):
    """Sample the 2D reference function `f(x, y) = 1 / (1 + x**2 + y**2)`."""

    def f(x, y):
        return 1 / (1 + x**2 + y**2)

    return sample_fn_2d(f, nx, ny, xmin, xmax, ymin, ymax, as_da=True)


def gaussian_2d_da(nx, ny, xmin, xmax, ymin, ymax, sigma_x=1, sigma_y=1, normalization=1):
    """Sample a centered anisotropic Gaussian over a Cartesian grid."""

    def fn(x, y):
        return gaussian_fn_2d(
            x,
            y,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            normalization=normalization,
        )

    return sample_fn_2d(fn, nx, ny, xmin, xmax, ymin, ymax, as_da=True)


def ft_tophat(k0, epsilon0, epsilon1):
    """Evaluate the analytic Fourier transform of a 1D top-hat function.

    Parameters
    ----------
    k0 : array-like
        Frequency coordinates.
    epsilon0, epsilon1 : float
        Open-interval support bounds of the top-hat in the spatial domain.

    Returns
    -------
    numpy.ndarray
        Complex Fourier-transform values on `k0`.
    """
    ft_theory = np.empty_like(k0, dtype=complex)
    nonzero = k0 != 0
    zero = ~nonzero

    ft_theory[nonzero] = (
        1
        / (2 * np.pi * 1j * k0[nonzero])
        * (
            np.exp(-2 * np.pi * 1j * k0[nonzero] * epsilon0)
            - np.exp(-2 * np.pi * 1j * k0[nonzero] * epsilon1)
        )
    )
    ft_theory[zero] = epsilon1 - epsilon0
    return ft_theory


def ft_tophat_2d(k, l, epsilon0, epsilon1, delta0, delta1):
    """Evaluate the analytic Fourier transform of a rectangular 2D top-hat."""
    fx = ft_tophat(k, epsilon0, epsilon1)[:, None]
    fy = ft_tophat(l, delta0, delta1)[None, :]
    return xr.DataArray(fx * fy, dims=("x", "y"), coords={"x": k, "y": l})


def ft_gaussian_2d(kx, ky, sigma_x=1, sigma_y=1, normalization=1.0, x0=0.0, y0=0.0, as_da=True):
    """Evaluate the analytic Fourier transform of a shifted 2D Gaussian.

    Parameters
    ----------
    kx, ky : array-like
        Frequency coordinates along the x and y directions.
    sigma_x, sigma_y : float, optional
        Standard deviations of the spatial-domain Gaussian.
    normalization : float, optional
        Integral of the spatial-domain Gaussian.
    x0, y0 : float, optional
        Center of the spatial-domain Gaussian. Nonzero values introduce a
        complex phase shift in the Fourier domain.
    as_da : bool, optional
        When `True`, return an `xarray.DataArray`; otherwise return a NumPy
        array.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Analytic Fourier-transform values sampled on the supplied grid.
    """
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    phase = np.exp(-2j * np.pi * (KX * x0 + KY * y0))
    amplitude = normalization * np.exp(
        -2 * (np.pi**2) * (sigma_x**2 * KX**2 + sigma_y**2 * KY**2)
    )
    ft_values = amplitude * phase

    if as_da:
        return xr.DataArray(ft_values, coords={"x": kx, "y": ky}, dims=("x", "y"))
    return ft_values


def apply_xy_lims_to_da(da: xr.DataArray, xlim=None, ylim=None):
    """Return a view of a 2D DataArray cropped to optional x and y limits."""
    if xlim is not None:
        da = da.sel(x=slice(*xlim))
    if ylim is not None:
        da = da.sel(y=slice(*ylim))
    return da


def reduce_density(da: xr.DataArray, x_reduction=1, y_reduction=1):
    """Subsample a 2D DataArray by integer strides along each axis."""
    return da.isel(
        x=slice(None, None, x_reduction),
        y=slice(None, None, y_reduction),
    )


def zero_outside_radius(da: xr.DataArray, radius: float, center_x=0.0, center_y=0.0):
    """Mask values outside a circular aperture by replacing them with zero."""
    X, Y = np.meshgrid(da.coords["x"].values, da.coords["y"].values, indexing="ij")
    mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= radius**2
    da_masked = da.where(mask, other=0.0)
    return da_masked
