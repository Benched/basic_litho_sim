import warnings
import pytest
import shapes
import fourier
import utils
import numpy as np
import xarray as xr

@pytest.mark.parametrize("N, epsilon0, epsilon1, xmin, xmax, tol", [
    (100000, -0.1, 0.2, -100, 100, 0.005),
])
def test_1d_tophat_transform(N, epsilon0, epsilon1, xmin, xmax, tol):
    # High-resolution Fourier transform test for a 1D top-hat function
    # This verifies that the numerical FT matches the known analytical expression
    
    N = 100000
    epsilon0 = -0.1
    epsilon1 = 0.2
    xmin = -100
    xmax = 100
    
    # Sample a narrow top-hat on a large domain to approximate continuous FT behavior
    tophat_transform = fourier.ft(shapes.top_hat_da(N, epsilon0, epsilon1, xmin, xmax))
    
    # Get frequency grid (Fourier domain) corresponding to the transform
    k0 = tophat_transform.x.values
    
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
    
    assert (
        len(ft_theory - tophat_transform) == N
    ), f"Output size of the fourier transform does not match expectations: expected {N} got {len(ft_theory - tophat_transform)}"
    max_error = np.max(np.abs(ft_theory - tophat_transform.values))
    assert max_error < tol, (
        f"Max deviation between analytical and numerical Fourier transform exceeds tolerance.\n"
        f"Max error: {max_error:.5f}"
    )


@pytest.mark.parametrize(
    "N, xmin, xmax, tolerance",
    [
        (100000, -500, 500, 0.005)
    ]
)
def test_1d_quadratic_decay_fourier(N, xmin, xmax, tolerance):
    """
    Test that the Fourier transform of the 1D quadratic decay function
    f(x) = 1 / (1 + x^2) matches its known analytical Fourier transform:
        FT(k) = π * exp(-2π * |k|)
    """
    # Sample the function and compute its FT
    da = shapes.quadratic_decay_da(N, xmin, xmax)
    ft_result = fourier.ft(da)

    # Evaluate the theoretical FT
    k = ft_result.x.values
    expected = np.pi * np.exp(-2 * np.pi * np.abs(k))

    # Compute error cleanly
    max_error = np.max(np.abs(ft_result.values - expected))

    # Assertions with clear message
    assert max_error < tolerance, (
        f"Max error {max_error:.5e} exceeds tolerance {tolerance}"
    )


@pytest.mark.parametrize("N, tol", [
    (50000, 1e-5),
])
def test_fourier_involution_assym_fn(N, tol):
    """
    Test the involution property using a smooth asymmetric function:
        F(F(f))(k) ≈ f(-t)
    """
    assym_fn = lambda x: np.sin(x) * (x + 1)**2 / ((x / 10)**6 + 1) / 63.67921267
    assym_fn_reversed = lambda x: assym_fn(-x)
    
    xmin, xmax = -np.sqrt(N), np.sqrt(N)
    sampled = utils.sample_fn(assym_fn, N, xmin, xmax)
    
    double_ft = fourier.ft(fourier.ft(sampled))
    
    domain = double_ft.x.values
    reverse_sample_image = list(map(assym_fn_reversed, domain))
    reverse_sample = utils.to_da(domain, reverse_sample_image)
    
    max_error = float(abs(reverse_sample - double_ft).max())
    
    assert max_error < tol, (
        f"Double Fourier transform failed involution property.\n"
        f"Max error: {max_error:.2e} (tolerance {tol})"
    )


@pytest.mark.parametrize("nx, ny, x0, x1, y0, y1, xmin, xmax, ymin, ymax, rel_tol", [
    (1024, 1024, 0, 2.2, -1, 1.8, -10, 10, -10, 10, 1e-2),
])
def test_2d_tophat_fourier_transform(nx, ny, x0, x1, y0, y1, xmin, xmax, ymin, ymax, rel_tol):
    """
    Verify that the 2D Fourier transform of a rectangular top-hat matches
    the known analytical result within a specified relative tolerance.
    """

    # Generate top-hat signal
    signal = shapes.top_hat_2d_da(nx, ny, x0, x1, y0, y1, xmin, xmax, ymin, ymax)

    # Compute numerical Fourier transform
    ft_numerical = fourier.ft2d(signal)

    # Frequency axes
    kx = ft_numerical.coords["x"].values
    ky = ft_numerical.coords["y"].values

    # Compute analytical Fourier transform
    ft_theory = shapes.ft_tophat_2d(kx, ky, x0, x1, y0, y1)

    # Compute maximum relative error
    error = np.abs(ft_numerical.values - ft_theory.values)
    max_relative_error = np.max(error) / np.abs(ft_theory.values).max()

    # Assertion
    assert max_relative_error < rel_tol, (
        f"2D top-hat FT failed.\n"
        f"Max relative error = {max_relative_error:.2e} exceeds tolerance {rel_tol}"
    )

#@pytest.mark.parametrize(
#    "nx, ny, xmin, xmax, ymin, ymax, tolerance",
#    [
#        (512, 512, -50, 50, -50, 50, 0.01),
#        # Add more cases if desired
#    ]
#)
#def test_2d_quadratic_decay_fourier(nx, ny, xmin, xmax, ymin, ymax, tolerance):
#    """
#    Test that the 2D Fourier transform of f(x, y) = 1 / (1 + x² + y²)
#    matches its known analytic FT: π * exp(-2π * sqrt(kx² + ky²))
#
#    The function is sampled over a large domain and compared pointwise
#    to the theoretical transform.
#    """
#    # Sample the function
#    da = shapes.quadratic_decay_2d_da(nx, ny, xmin, xmax, ymin, ymax)
#    ft_result = fourier.ft2d(da)  # Should return xarray.DataArray with coords "x", "y"
#
#    # Frequency grids
#    kx = ft_result.coords["x"].values
#    ky = ft_result.coords["y"].values
#    KX, KY = np.meshgrid(kx, ky, indexing="ij")
#    kr = np.sqrt(KX**2 + KY**2)
#
#    # Analytical FT
#    expected = np.pi * np.exp(-2 * np.pi * kr)
#
#    # Compute error
#    max_error = np.max(np.abs(ft_result.values - expected))
#
#    assert max_error < tolerance, (
#        f"2D Fourier transform mismatch: max error {max_error:.5e} exceeds tolerance {tolerance}"
#    )


# @pytest.mark.parametrize("n", [10, 11])
# def test_block_position_corresponds_symmetric_function(n, epsilon=0.3):
#    """Verify for both even and odd n that the Fourier transform has no imaginary component."""
#    f = block(n, epsilon)
#    ft, _ = fourier(f)
#    assert (
#        np.max(np.abs(ft - np.real(ft))) < 1e-20
#    ), f"(n = {n}): Fourier transform of what is supposed to be an even function has an imaginary component."
