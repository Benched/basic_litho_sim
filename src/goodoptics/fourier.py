import numpy as np
import xarray as xr
from .utils import to_da


def _ft(samples, Fs, t0):
    """Approximate the Fourier Transform of a time-limited
    signal by means of the discrete Fourier Transform.

    samples: signal values sampled at the positions t0 + n/Fs
    Fs: Sampling frequency of the signal
    t0: starting time of the sampling of the signal

    source: https://dspillustrations.com/pages/posts/misc/approximating-the-fourier-transform-with-dft.html
    """
    f = np.fft.fftshift(
        np.fft.fftfreq(len(samples), 1 / Fs)
    )  # np.linspace(-Fs/2, Fs/2, len(samples), endpoint=False)
    return f, np.fft.fftshift(np.fft.fft(samples)) / Fs * np.exp(-2j * np.pi * f * t0)


def ft(da: xr.DataArray) -> xr.DataArray:
    Fs = float(
        da.n / (da.dx * da.n)
    )  # Note that x[-1] - x[0] would give a slightly different value - this would have a rather big impact on the result and is not correct, given the way we sample.
    t0 = float(da.x[0])
    f, ft_val = _ft(da.values, Fs, t0)
    return to_da(f, ft_val)


def ft2d(da: xr.DataArray) -> xr.DataArray:
    """
    Approximates the 2D Continuous Fourier Transform using the 2D FFT.

    Parameters:
        da (xr.DataArray): 2D signal with coordinates 'x' and 'y'

    Returns:
        xr.DataArray: Complex spectrum with coordinates 'kx', 'ky'
    """
    # Coordinates
    x = da.coords["x"].values
    y = da.coords["y"].values
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    x0 = x[0]
    y0 = y[0]
    Nx, Ny = len(x), len(y)

    # Frequencies
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))

    # 2D FFT and scaling
    fft_vals = np.fft.fftshift(np.fft.fft2(da.values))
    spectrum = fft_vals * dx * dy * np.exp(-2j * np.pi * (np.outer(kx, np.ones_like(ky)) * x0 + np.outer(np.ones_like(kx), ky) * y0))

    # Return xarray
    return xr.DataArray(
        spectrum,
        dims=["x", "y"],
        coords={"x": kx, "y": ky},
        attrs={
            "description": "2D Continuous Fourier Transform approximation",
            "dx": dx,
            "dy": dy,
            "x0": x0,
            "y0": y0
        }
    )