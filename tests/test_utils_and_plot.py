import numpy as np
import plotly.graph_objects as go
import xarray as xr

from goodoptics import plot, shapes, utils


def test_to_da_sets_expected_metadata_from_coordinates():
    x = np.array([-1.0, -0.5, 0.0, 0.5])
    values = np.array([3.0, 4.0, 5.0, 6.0])

    da = utils.to_da(x, values)

    assert da.dims == ("x",)
    assert da.n == len(values)
    np.testing.assert_allclose(da.x.values, x)
    np.testing.assert_allclose(da.values, values)
    np.testing.assert_allclose(da.dx, 0.5)


def test_sample_fn_2d_can_return_raw_meshes_and_values():
    X, Y, Z = utils.sample_fn_2d(
        lambda x, y: x + 2 * y,
        nx=3,
        ny=2,
        xmin=0.0,
        xmax=2.0,
        ymin=-1.0,
        ymax=1.0,
        as_da=False,
        endpoint=True,
    )

    assert X.shape == (3, 2)
    assert Y.shape == (3, 2)
    assert Z.shape == (3, 2)
    np.testing.assert_allclose(X[:, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(Y[0, :], np.array([-1.0, 1.0]))
    np.testing.assert_allclose(Z, X + 2 * Y)


def test_riemann_approximation_matches_tophat_fourier_transform():
    N = 4000
    t0 = 3.0

    frequencies, numerical_ft = utils.riemann_approximation_ft_of_fn_on_symmetric_interval(
        lambda x: 1.0,
        N=N,
        t0=t0,
    )
    expected_ft = shapes.ft_tophat(frequencies, -t0, t0)

    max_error = np.max(np.abs(numerical_ft - expected_ft))
    assert max_error < 1e-2, f"Riemann FT approximation drifted by {max_error:.3e}"


def test_complex_surface_separates_real_and_imaginary_parts():
    da = xr.DataArray(
        np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]),
        coords={"x": [0.0, 1.0], "y": [0.0, 2.0]},
        dims=("x", "y"),
    )

    fig = plot.complex_surface(da, title_1="Real", title_2="Imaginary")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    np.testing.assert_allclose(np.array(fig.data[0].z), np.real(da.values))
    np.testing.assert_allclose(np.array(fig.data[1].z), np.imag(da.values))


def test_real_surface_returns_single_surface_trace():
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        coords={"x": [-1.0, 1.0], "y": [-2.0, 2.0]},
        dims=("x", "y"),
    )

    fig = plot.real_surface(da, title="Example", cmap="Cividis")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    np.testing.assert_allclose(np.array(fig.data[0].z), da.values)
