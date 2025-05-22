import numpy as np
import xarray as xr

def to_da(x, values, xmin=None, xmax=None, endpoint=False):
    da = xr.DataArray(values, dims=("x"), coords={"x": x})

    if xmin is None:
        xmin = min(x)

    if xmax is None:
        dx = x[1] - x[0]  # assume uniform spacing
        xmax = x[-1] + dx if endpoint is False else x[-1]

    n = len(values)
    dx = (xmax - xmin) / n  # compute over full domain

    return da.assign_attrs({"dx": dx, "n": n})


def sample_fn(f, N, t0, t1, as_da=True):
    x = np.linspace(t0, t1, N, endpoint=False)
    sample = np.array(list(map(f, x)))
    if as_da:
        return to_da(x, sample)
    else:
        return x, sample

def sample_fn_2d(fn, nx, ny, xmin, xmax, ymin, ymax, as_da=True, endpoint=False):
    """
    Samples a 2D scalar function f(x, y) over a regular grid.

    Parameters:
        fn: callable (x, y) -> float
        nx, ny: number of samples in x and y directions
        xmin, xmax: x range
        ymin, ymax: y range
        as_da: if True, return an xarray.DataArray; else return raw numpy arrays

    Returns:
        - If as_da is True: xarray.DataArray with dims ['x', 'y']
        - If as_da is False: (X, Y, Z) where:
            X: 2D x-coordinates (meshgrid)
            Y: 2D y-coordinates (meshgrid)
            Z: 2D function values
    """
    x = np.linspace(xmin, xmax, nx, endpoint=endpoint)
    y = np.linspace(ymin, ymax, ny, endpoint=endpoint)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    X, Y = np.meshgrid(x, y, indexing="ij")  # shape (nx, ny)
    Z = np.vectorize(fn)(X, Y)

    if as_da:
        da = xr.DataArray(Z, coords={"x": x, "y": y}, dims=["x", "y"])
        da.attrs.update({"dx": dx, "dy": dy, "nx": nx, "ny": ny})
        return da
    else:
        return X, Y, Z


def riemann_approximation_ft_of_fn_on_symmetric_interval(f, N, t0):
    """Determines the fourier transform for a function on -t0, t0 via the Riemann sum.

    Used for testing.

    Inputs:
        fn (function): a complex function on (-t0, t0)
        N (int): The number
        t0 (float): Defines the interval -t0, t0 on which to sample


    Returns
        Nummerical approximation of the fourier transform of fn.
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
