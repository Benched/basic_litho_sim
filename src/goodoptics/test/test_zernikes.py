from zernikes import zernike, fit_zernikes_to_da, zernike_aberration
import zernikes
import pytest
import numpy as np
import xarray as xr
import random
import plotly
import matplotlib

@pytest.mark.parametrize(
    "m, n, N",
    [
        (3, 1, 400),
        (56, 6, 833),
        (24, 0, 650)
    ]
)
def test_zernike(m, n, N):
    """Helper to check that spacing inferred from x-coords matches stored dx."""
    def assert_spacing_matches(da, msg_prefix=""):
        spacing = float(da.x.diff("x").isel(x=0))
        np.testing.assert_almost_equal(
            spacing,
            da.dx,
            err_msg=f"{msg_prefix} dx spacing {float(da.dx)} != x-diff {spacing}",
        )

    """Ensure sampling grid, spacing, caching, and endpoint logic behave consistently."""
    z = zernike(m, n)
    z.N = N
    da = z.sample_da()

    # initial properties
    assert da.shape == (N, N), "Shape of the sample is set incorrectly"
    assert not z.endpoint

    # --- initial spacing check ---
    assert_spacing_matches(da, f"Initial {(z.n, z.m)}:")

    # Reset bookkeeping via N
    z.N = z.N -1
    assert z._sample_cache is None, "Sample did not reset correctly for N update"
    assert z._sample_da_cache is None, "da sample did not reset correctly  for N update"

    # --- plot() triggers sampling only ---
    r = z.plot()
    assert z._sample_cache is not None, "sample was not resampled correctly after plot"
    assert z._sample_da_cache is None, "da sample should not resample upon plot"

    # --- sample_da() creates DataArray and metadata ---
    da = z.sample_da()
    assert z._sample_da_cache is not None, "da sample should be resampled, it was not"
    assert isinstance(da, xr.DataArray), "sample da does not return an xarray.DataArray"
    assert "x" in da.coords and "y" in da.coords, "da does not have all coordinates"
    assert "dx" in da.attrs and "dy" in da.attrs, "da does not have all attributes"

    spacing = float(da.x.diff('x').isel(x=0))
    np.testing.assert_almost_equal(spacing , da.dx, err_msg= f"dx spacing {float(da.dx)} does not match spacing in x ({spacing}) anymore after updating N and resampling {(z.n, z.m)}")

    # --- endpoint change resets caches again ---
    z.endpoint = not z.endpoint
    assert z.endpoint, "endpoint setter or getter did not behave correctly"
    assert z._sample_cache is None, "Sample did not reset correctly after endpoint change"
    assert z._sample_da_cache is None, "da sample did not reset correctly after endpoint change"

    spacing = float(da.x.diff('x').isel(x=0))
    np.testing.assert_almost_equal(spacing , da.dx, err_msg= f"dx spacing {float(da.dx)} does not match spacing in x ({spacing}) anymore after changing endpoint and resampling {(z.n, z.m)}")

def test_zernike_comparison():
    z1 = zernike(3, 1)
    z2 = zernike(3, 1)
    z3 = zernike(4, 0)
    z4 = zernike(12, 2)

    assert z1 == z2, "Zernike equality comparison failed for identical (n,m)"
    assert z1 != z3, "Zernike inequality comparison failed for different (n,m)"
    assert z4 > z3, "Zernike greater-than comparison failed"
    assert z1 < z4, "Zernike less-than comparison failed"

@pytest.mark.parametrize(
    "m, n, N",
    [
        (3, 1, 100),
    ]
)
def test_zernike_surfaceplot(m, n, N):
    z = zernike(m, n)
    r = z.plot_surface()
    assert isinstance(r, plotly.graph_objects.Figure)


def test_fringe_roundtrip():
    base_indices = list(range(1, 11))
    random_indices = random.sample(range(11, 10001), k=50) 
    indices = sorted(base_indices + random_indices)
    results = [zernikes.nm_to_fringe(*zernikes.fringe_to_nm(i)) == i for i in indices]
    assert np.all(results), f"Round-trip failed for indices: {np.array(indices)[~np.array(results)]}"

def test_zernike_aberration_addition():
    za1 = zernikes.zernike_aberration({1:0.1, 3:0.2, 12:0.3}, N=200)
    za2 = zernikes.zernike_aberration({3:0.4, 4:0.5}, N=200)

    za3 = za1 + za2

    expected_indices = [1,3,4,12]
    actual_indices = [z.j for z in za3.zernikes]
    assert expected_indices == actual_indices, f"Expected zernike indices {expected_indices}, got {actual_indices}"

    expected_coefficients = {1:0.1, 3:0.6, 4:0.5, 12:0.3}
    actual_coefficients = {z.j: float(za3.coefficients.sel(zernike=z.j).values) for z in za3.zernikes}
    for idx in expected_coefficients:
        np.testing.assert_almost_equal(
            expected_coefficients[idx],
            actual_coefficients[idx],
            err_msg=f"Coefficient for zernike {idx} incorrect: expected {expected_coefficients[idx]}, got {actual_coefficients[idx]}"
        )

def test_zernike_aberration_add_zernikes():
    za = zernikes.zernike_aberration({1:0.1, 2:0.2}, N=200)
    za.add_zernikes({2:0.3, 3:0.4})

    expected_indices = [1,2,3]
    actual_indices = [z.j for z in za.zernikes]
    assert expected_indices == actual_indices, f"Expected zernike indices {expected_indices}, got {actual_indices}"

    expected_coefficients = {1:0.1, 2:0.5, 3:0.4}
    actual_coefficients = {z.j: float(za.coefficients.sel(zernike=z.j).values) for z in za.zernikes}
    for idx in expected_coefficients:
        np.testing.assert_almost_equal(
            expected_coefficients[idx],
            actual_coefficients[idx],
            err_msg=f"Coefficient for zernike {idx} incorrect: expected {expected_coefficients[idx]}, got {actual_coefficients[idx]}"
        )

def test_zernike_aberration_subtraction():
    za1 = zernikes.zernike_aberration({1:0.5, 2:0.7, 6:0.9}, N=200)
    za2 = zernikes.zernike_aberration({2:0.2, 3:0.4}, N=200)

    za3 = za1 - za2

    expected_indices = [1,2,3,6]
    actual_indices = [z.j for z in za3.zernikes] # Sort is part of the expected behavior
    assert expected_indices == actual_indices, f"Expected zernike indices {expected_indices}, got {actual_indices}"

    expected_coefficients = {1:0.5, 2:0.5, 3:-0.4, 6:0.9}
    actual_coefficients = {z.j: float(za3.coefficients.sel(zernike=z.j).values) for z in za3.zernikes}
    for idx in expected_coefficients:
        np.testing.assert_almost_equal(
            expected_coefficients[idx],
            actual_coefficients[idx],
            err_msg=f"Coefficient for zernike {idx} incorrect: expected {expected_coefficients[idx]}, got {actual_coefficients[idx]}"
        )

def test_zernike_aberration_cache_reset_on_N_change():
    za = zernikes.zernike_aberration({1:0.1, 2:0.2}, N=100)
    _ = za.sample_da()
    assert za._sample_da_cache is not None, "Wavefront cache should be populated after first computation"
    for z in za.zernikes:
        assert z._sample_cache is not None, f"Zernike {z.j} sample cache should be populated after first computation"
        assert z._sample_da_cache is not None, f"Zernike {z.j} sample_da cache should be populated after first computation"

    za.N = 50
    assert za._sample_da_cache is None, "Wavefront cache should be reset to None after N change"
    for z in za.zernikes:
        assert z._sample_cache is None, f"Zernike {z.j} sample cache should be reset to None after N change"
        assert z._sample_da_cache is None, f"Zernike {z.j} sample_da cache should be reset to None after N change"

def test_zernike_aberration_plots():
    za = zernikes.zernike_aberration({1:0.1, 2:0.2}, N=100)
    fig1 = za.plot()
    assert isinstance(fig1, matplotlib.artist.Artist), "Zernike plot did not return a Matplotlib AxesImage"

    fig2 = za.plot_surface()
    assert isinstance(fig2, plotly.graph_objects.Figure), "Zernike plot did not return a Plotly Figure"

def test_fit_zernikes_to_da():
    za_true = zernike_aberration({1:0.3, 2:0.5, 5:-0.2}, N=150)
    da = za_true.sample_da()

    za_fit = fit_zernikes_to_da(da, max_j=5)

    for j in za_true.coefficients.zernike.values:
        true_coeff = float(za_true.coefficients.sel(zernike=j).values)
        fit_coeff = float(za_fit.coefficients.sel(zernike=j).values)
        np.testing.assert_almost_equal(
            true_coeff,
            fit_coeff,
            decimal=5,
            err_msg=f"Fitted coefficient for zernike {j} incorrect: expected {true_coeff}, got {fit_coeff}"
        )

    za_true.N = 400
    da = za_true.sample_da()
    za_fit = fit_zernikes_to_da(da, max_j=4)

    assert 5 not in za_fit.coefficients.zernike.values, "Fitted zernike_aberration contains zernike beyond max_j limit"
    for j in za_true.coefficients.zernike.values:
        if j > 4:
            continue
        true_coeff = float(za_true.coefficients.sel(zernike=j).values)
        fit_coeff = float(za_fit.coefficients.sel(zernike=j).values)
        np.testing.assert_almost_equal(
            true_coeff,
            fit_coeff,
            decimal=5,
            err_msg=f"Fitted coefficient for zernike {j} incorrect: expected {true_coeff}, got {fit_coeff}" # This assertion is justified by zernike orthogonality and a sufficiently dense sampling
        )


def test_zernike_aberration_sample_caching_behavior():
    import zernikes
    import numpy as np

    # Construct a small aberration with two modes
    za = zernikes.zernike_aberration({1: 0.5, 2: -0.3}, N=80)

    # First sampling call – should populate the cache
    result1 = za.sample()
    assert hasattr(za, "_cached_sample"), "zernike_aberration missing _cached_sample attribute"
    assert za._cached_sample is not None, "Cache was not populated after first sample() call"

    # Replace the lambdified Zernike functions to detect recomputation
    original_lambdas = [z._lambdified for z in za.zernikes]
    recomputation_flags = {"called": False}
    def wrapped(*args, **kwargs):
        recomputation_flags["called"] = True
        return np.zeros_like(args[0])
    za.zernikes[0]._lambdified = wrapped  # override one zernike

    # Second call should use cache, not trigger recomputation
    _ = za.sample()
    assert not recomputation_flags["called"], (
        "sample() recomputed Zernikes despite cache being filled — caching broken"
    )

    # Restore
    za.zernikes[0]._lambdified = original_lambdas[0]