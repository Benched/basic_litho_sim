import numpy as np
import xarray as xr
import shapes
from utils import sample_fn,sample_fn_2d


def tophat_fn(x0, x1):
    """
    Returns a 1D top-hat function:
    f(x) = 1 if x0 < x < x1
         = 0 otherwise

    Parameters:
        x0, x1: bounds of the top-hat region (exclusive)
    """
    return lambda x: int(x0 < x < x1)

def tophat_fn_2d(x0, x1, y0, y1):
    """
    Returns a 2D top-hat function:
    f(x, y) = 1 if x0 < x < x1 and y0 < y < y1
           = 0 otherwise

    Parameters:
        x0, x1: horizontal bounds of the "hat" (exclusive)
        y0, y1: vertical bounds of the "hat" (exclusive)
    """
    return lambda x, y: int(x0 < x < x1 and y0 < y < y1)

def gaussian_fn_2d(x, y):
    return np.exp(-np.pi * (x**2 + y**2))

def circular_tophat_fn_2d(radius, center_x=0.0, center_y=0.0):
    """
    Returns a 2D circular top-hat function centered at (center_x, center_y).
    The function is 1 inside the circle of the given radius, 0 outside.

    Uses strict inequality: points on the edge are excluded.

    Parameters:
        radius: float
            Radius of the circular region.
        center_x, center_y: float
            Coordinates of the circle's center.

    Returns:
        Callable[[float, float], int]
    """
    return lambda x, y: int(((x - center_x) ** 2 + (y - center_y) ** 2) < (radius ** 2))
    
def quadratic_decay_fn():
    return lambda x: 1 / (1 + x**2)


def top_hat_da(n, x0, x1, xmin, xmax):
    return sample_fn(tophat_fn(x0, x1), n, xmin, xmax)

def quadratic_decay_da(n, xmin, xmax):
    return sample_fn(quadratic_decay_fn(), n, xmin, xmax)

def top_hat_2d_da(nx, ny, x0, x1, y0, y1, xmin, xmax, ymin, ymax):
    """
    Returns a 2D top-hat function sampled over a rectangular grid.

    Parameters:
        nx, ny: number of samples in x and y directions
        x0, x1: horizontal bounds of the top-hat region
        y0, y1: vertical bounds of the top-hat region
        xmin, xmax: full x-domain sampling range
        ymin, ymax: full y-domain sampling range

    Returns:
        xarray.DataArray with dimensions ['x', 'y']
    """
    fn = tophat_fn_2d(x0, x1, y0, y1)
    return sample_fn_2d(fn, nx, ny, xmin, xmax, ymin, ymax, as_da=True)
    
def circular_top_hat_da(nx, ny, radius, xmin, xmax, ymin, ymax, center_x=0.0, center_y=0.0):
    """
    Samples a circular top-hat function over a 2D grid.

    Parameters:
        nx, ny: int
            Number of grid points in x and y.
        radius: float
            Radius of the circular top-hat.
        xmin, xmax: float
            Bounds of the x-domain.
        ymin, ymax: float
            Bounds of the y-domain.
        center_x, center_y: float, optional
            Center of the circular region. Defaults to (0.0, 0.0).

    Returns:
        xarray.DataArray with dims ['x', 'y'] and binary values (0 or 1).
    """
    fn = circular_tophat_fn_2d(radius, center_x=center_x, center_y=center_y)
    return sample_fn_2d(fn, nx, ny, xmin, xmax, ymin, ymax, as_da=True)

def quadratic_decay_2d_da(nx, ny, xmin, xmax, ymin, ymax):
    def f(x, y):
        return 1 / (1 + x**2 + y**2)
    return sample_fn_2d(f, nx, ny, xmin, xmax, ymin, ymax, as_da=True)

def gaussian_2d_da(nx, ny, x0, x1, y0, y1, xmin, xmax, ymin, ymax):
    return sample_fn_2d(gaussian_fn_2d, nx, ny, xmin, xmax, ymin, ymax, as_da=True)


def ft_tophat(k0, epsilon0, epsilon1):
    # Compute the analytical Fourier transform of the top-hat:
    # FT(k) = (1 / 2πi k) * (e^{-2πi k ε₀} - e^{-2πi k ε₁})
    # But we need to handle the k = 0 case separately to avoid division by zero
    ft_theory = np.empty_like(k0, dtype=complex)
    nonzero = k0 != 0
    zero = ~nonzero
    
    # Safe computation only for nonzero frequencies
    ft_theory[nonzero] = (
        1 / (2 * np.pi * 1j * k0[nonzero])
        * (np.exp(-2 * np.pi * 1j * k0[nonzero] * epsilon0) - np.exp(-2 * np.pi * 1j * k0[nonzero] * epsilon1))
    )
    
    # Patch k = 0 manually (limit of the expression is the top-hat's area)
    ft_theory[zero] = epsilon1 - epsilon0
    return ft_theory

def ft_tophat_2d(k, l, epsilon0, epsilon1, delta0, delta1):
    """
    Compute the continuous 2D Fourier transform of a rectangular top-hat:
        1 in (epsilon0, epsilon1) × (delta0, delta1), 0 elsewhere.

    Parameters:
        k, l : 1D arrays of frequency coordinates
        epsilon0, epsilon1 : interval for x-axis
        delta0, delta1 : interval for y-axis

    Returns:
        2D complex array of FT values with shape (len(k), len(l))
    """
    fx = ft_tophat(k, epsilon0, epsilon1)[:, None]  # shape (Nx, 1)
    fy = ft_tophat(l, delta0, delta1)[None, :]      # shape (1, Ny)
    return xr.DataArray(fx * fy, dims=('x', 'y'), coords={'x':k, 'y':l})