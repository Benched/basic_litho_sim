"""Discrete approximations of the continuous Fourier transform.

These helpers work with the `xarray.DataArray` objects produced by the sampling
utilities in this package. The sampling metadata stored on those arrays is used
to keep the Fourier scaling consistent with the notebook examples.
"""

import numpy as np
import xarray as xr

from .utils import to_da


def _ft(samples, Fs, t0):
    """Return a scaled DFT approximation of a 1D continuous Fourier transform.

    Parameters
    ----------
    samples : array-like
        Signal values sampled at positions `t0 + n / Fs`.
    Fs : float
        Sampling frequency in samples per unit distance.
    t0 : float
        Starting coordinate of the sampled signal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The centered frequency grid and the corresponding complex Fourier
        samples.

    Notes
    -----
    The scaling follows the convention used throughout the notebooks and is
    based on the derivation described at:
    https://dspillustrations.com/pages/posts/misc/approximating-the-fourier-transform-with-dft.html
    """
    f = np.fft.fftshift(np.fft.fftfreq(len(samples), 1 / Fs))
    ft_values = np.fft.fftshift(np.fft.fft(samples)) / Fs * np.exp(-2j * np.pi * f * t0)
    return f, ft_values


def ft(da: xr.DataArray) -> xr.DataArray:
    """Approximate the 1D continuous Fourier transform of a sampled signal.

    Parameters
    ----------
    da : xarray.DataArray
        One-dimensional sample with an `x` coordinate and `dx` / `n` metadata
        as produced by :func:`goodoptics.utils.to_da` or
        :func:`goodoptics.utils.sample_fn`.

    Returns
    -------
    xarray.DataArray
        Complex Fourier samples indexed by frequency.

    Notes
    -----
    The sampling frequency is computed from the stored metadata rather than
    from endpoint differences so that the result stays aligned with the
    repository's half-open sampling convention.
    """
    Fs = float(da.n / (da.dx * da.n))
    t0 = float(da.x[0])
    f, ft_val = _ft(da.values, Fs, t0)
    return to_da(f, ft_val)


def ft2d(da: xr.DataArray) -> xr.DataArray:
    """Approximate the 2D continuous Fourier transform of a sampled surface.

    Parameters
    ----------
    da : xarray.DataArray
        Two-dimensional sample with `x` and `y` coordinates.

    Returns
    -------
    xarray.DataArray
        Complex Fourier spectrum with centered frequency coordinates on the
        same `x` / `y` dimension names used by the input array.
    """
    x = da.coords["x"].values
    y = da.coords["y"].values
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    x0 = x[0]
    y0 = y[0]
    Nx, Ny = len(x), len(y)

    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))

    fft_vals = np.fft.fftshift(np.fft.fft2(da.values))
    phase_correction = np.exp(
        -2j
        * np.pi
        * (
            np.outer(kx, np.ones_like(ky)) * x0
            + np.outer(np.ones_like(kx), ky) * y0
        )
    )
    spectrum = fft_vals * dx * dy * phase_correction

    return xr.DataArray(
        spectrum,
        dims=["x", "y"],
        coords={"x": kx, "y": ky},
        attrs={
            "description": "2D Continuous Fourier Transform approximation",
            "dx": dx,
            "dy": dy,
            "x0": x0,
            "y0": y0,
        },
    )
