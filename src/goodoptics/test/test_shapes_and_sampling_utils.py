import warnings
import pytest
import shapes
import numpy as np
import xarray as xr

"""
Test Suite Structure for Shape Sampling Utilities (1D and 2D)

This test suite provides layered and comprehensive coverage for shape-generating
functions, both in 1D (e.g., top-hat, quadratic decay) and 2D (e.g., rectangular
top-hat, circular top-hat).

We organize the tests into three main categories:

1. Exact Content Tests (Examples & Edge Cases)
   - Verify known outputs by comparing against hand-crafted expected arrays
   - Used in both 1D and 2D to confirm shape correctness at key grid points
   - Especially useful for checking edge alignment and clipping behavior
   - Examples: `test_block_1d`, `test_top_hat_2d_clipped`, `test_circular_top_hat_corner_overlap`

2. Structural Integrity Tests
   - Check that the sampled xarray.DataArray has correct shape, coordinate
     bounds, spacing (dx, dy), and metadata attributes (nx, ny, etc.)
   - Separate from content; ensures sampling is grid-consistent
   - Example: `assert_grid_metadata_correct`, `check_shapes_da_metadata`

3. Analytic Function Match Tests (Parameterized)
   - Evaluate the analytic shape function directly over the grid
   - Compare sampled output to expected values derived from the definition
   - Scalable using `@pytest.mark.parametrize`
   - Covers both 1D and 2D shapes for functional correctness
   - Example: `test_shape_matches_analytic`

This layered approach ensures:
- Precise validation of specific outcomes (via example-based tests)
- Consistency of sampling infrastructure (via structural checks)
- General correctness of shape logic (via functional comparison)

To add a new shape function:
  - Implement its analytical form (e.g., `triangular_fn`, `gaussian_fn_2d`)
  - Add parameterized tests comparing sampled and analytic results
  - Include at least one hardcoded content test if the shape involves edge or clipping behavior

This strategy balances confidence, readability, and scalability.
"""


def test_top_hat_1d_exact():
    """Test 1D top-hat matches expected binary array."""
    da = shapes.top_hat_da(10, -0.5, 0.5, -1, 1)
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    assert np.array_equal(da.values, expected), "Top-hat 1D does not match expected values."


def test_quadratic_decay_1d_exact():
    """Test 1D quadratic decay function samples match the formula."""
    da = shapes.quadratic_decay_da(5, -2, 2)
    expected = np.array([1 / (1 + x**2) for x in da.x.values])
    assert np.allclose(da.values, expected), "Quadratic decay 1D does not match expected values."


def test_top_hat_2d_exact():
    """
    Test a rectangular 2D top-hat that is partially inside the sampling domain.

    The top-hat region is defined over (0.1, 1.51) × (-0.51, 0.51) using strict inequalities.
    The sampling grid is 5×5 over [-1, 1) in both x and y, using endpoint=False.
    Only values strictly inside the region should be 1.
    """
    da = shapes.top_hat_2d_da(
                nx=5, ny=5,
                x0=0.1, x1=1.51,
                y0=-0.51, y1=0.51,
                xmin=-1, xmax=1,
                ymin=-1, ymax=1
            )
        
    expected = np.array(
       [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0],
       [0, 0, 1, 1, 0]])

    assert np.array_equal(da.values, expected)

def test_circular_top_hat_exact():
    """
    Test a circular top-hat that appears at the edge of the domain.

    The circle is centered at (1, -2) with radius 1, and is partially clipped by the grid boundary.
    """
    da = shapes.circular_top_hat_da(
        nx=7, ny=8,
        radius=1,
        xmin=0, xmax=2,
        ymin=-2, ymax=0,
        center_x=1,
        center_y=-2
    )
    
    expected = np.array(
      [[0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 1, 0, 0, 0, 0, 0]])

    assert np.array_equal(da.values, expected)

def assert_grid_metadata_correct(da, nx, ny, xmin, xmax, ymin, ymax, atol=1e-10):
    """
    Assert that a 2D xarray.DataArray has correct grid structure and metadata.

    Parameters:
        da : xarray.DataArray
            The data array to validate (must have 'x' and 'y' dimensions).
        nx, ny : int
            Expected number of samples in x and y.
        xmin, xmax : float
            Expected bounds of the x coordinate.
        ymin, ymax : float
            Expected bounds of the y coordinate.
        atol : float
            Absolute tolerance for floating-point comparisons.
    """
    assert da.shape == (nx, ny)
    x = da.coords["x"].values
    y = da.coords["y"].values

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    # Start of domain is still exact
    assert np.isclose(x[0], xmin)
    assert np.isclose(y[0], ymin)

    # End is one dx/dy short of xmax/ymax
    assert np.isclose(x[-1], xmax - dx)
    assert np.isclose(y[-1], ymax - dy)

    # Check spacing
    assert np.allclose(np.diff(x), dx)
    assert np.allclose(np.diff(y), dy)

    # Optional: check attrs if present
    if "nx" in da.attrs:
        assert da.attrs["nx"] == nx, f"nx attr mismatch: expected {nx}, got {da.attrs['nx']}"
    if "ny" in da.attrs:
        assert da.attrs["ny"] == ny, f"ny attr mismatch: expected {ny}, got {da.attrs['ny']}"

def check_2d_shapes_da_metadata( # Helper function
    shape_fn,
    nx, ny,
    xmin, xmax, ymin, ymax,
    *shape_args,
    **shape_kwargs
):
    da = shape_fn(nx, ny, *shape_args, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, **shape_kwargs)
    assert_grid_metadata_correct(da, nx, ny, xmin, xmax, ymin, ymax)

@pytest.mark.parametrize(
    "shape_fn, shape_args, shape_kwargs, nx, ny, xmin, xmax, ymin, ymax",
    [
        (shapes.top_hat_2d_da, (-1, 1, -0.5, 0.5), {}, 100, 80, -2, 2, -1, 1),
        (shapes.circular_top_hat_da, (), {"radius": 0.75}, 128, 128, -2, 2, -2, 2),
    ]
)
def test_2d_shape_metadata(
    shape_fn, shape_args, shape_kwargs, nx, ny, xmin, xmax, ymin, ymax
):
    check_2d_shapes_da_metadata(
        shape_fn,
        nx, ny,
        xmin, xmax, ymin, ymax,
        *shape_args,
        **shape_kwargs
    )

@pytest.mark.parametrize(
    "shape_fn, n, shape_kwargs, expected_bounds",
    [
        (
            shapes.top_hat_da,
            11,
            {"x0": -0.5, "x1": 0.5, "xmin": -1, "xmax": 1},
            (-1, 1)
        ),
        # Add more shape functions here in the future
    ]
)
def test_shape_1d_structure(shape_fn, n, shape_kwargs, expected_bounds):
    """
    Test that a 1D shape-generating function returns a well-formed DataArray
    with correct shape, spacing, and coordinate bounds.
    """
    da = shape_fn(n, **shape_kwargs)

    xmin, xmax = expected_bounds
    dx_expected = (xmax - xmin) / (n)

    assert da.dims == ("x",)
    assert da.shape == (n,)
    assert da.n == n
    assert np.isclose(da.dx, dx_expected)
    assert np.isclose(da.x[0], xmin)
    assert np.isclose(da.x[-1], xmax-dx_expected)
    
@pytest.mark.parametrize(
    "shape_fn, shape_args, shape_kwargs, analytic_fn",
    [
        (shapes.top_hat_2d_da, (100, 80), {"x0": -1, "x1": 1.5, "y0": 0, "y1": 1, "xmin": -0.7, "xmax": 2, "ymin": -0.5, "ymax": 3},
         shapes.tophat_fn_2d(-1, 1.5, 0, 1)),
        (shapes.circular_top_hat_da, (100, 80), {"radius": 1, "center_x": 0.5, "center_y": 0.5, "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1},
         shapes.circular_tophat_fn_2d(1, 0.5, 0.5)),
        # Add more shapes here
    ]
)
def test_shape_matches_analytic(shape_fn, shape_args, shape_kwargs, analytic_fn):
    """
    Test that the sampled shape function matches its analytical version
    over the defined 2D grid.
    """
    da = shape_fn(*shape_args, **shape_kwargs)

    x = da.coords["x"].values
    y = da.coords["y"].values
    X, Y = np.meshgrid(x, y, indexing="ij")

    expected = np.vectorize(analytic_fn)(X, Y)
    assert np.array_equal(da.values, expected), (
        f"Sampled values do not match the analytical function.\n"
        f"Function: {shape_fn.__name__}\n"
        f"Shape args: {shape_args}\n"
        f"Shape kwargs: {shape_kwargs}"
    )

@pytest.mark.parametrize(
    "nx, ny, x0, x1, xmin, xmax, ymin, ymax, y0, y1",
    [
        (56, 34, 0.1, 0.7, -1, 1, -3, 3, -2, 4),
    ],
)
def test_sample_2d_dx_dy_consistency(nx, ny, x0, x1, xmin, xmax, ymin, ymax, y0, y1):
    """Verify that dx, dy in attrs match actual coordinate spacing for a 2D top-hat sample."""
    sample_2d = shapes.top_hat_2d_da(nx, ny, x0, x1, xmin, xmax, ymin, ymax, y0, y1)

    dx_attr = sample_2d.attrs["dx"]
    dy_attr = sample_2d.attrs["dy"]

    # Check coordinate differences vs stored dx, dy
    assert np.allclose(np.diff(sample_2d.x), dx_attr, atol=1e-15), \
        f"dx mismatch: expected {dx_attr}, got {np.diff(sample_2d.x).mean()}"
    assert np.allclose(np.diff(sample_2d.y), dy_attr, atol=1e-15), \
        f"dy mismatch: expected {dy_attr}, got {np.diff(sample_2d.y).mean()}"

def test_reduce_density():
    da = xr.DataArray(
        [[0, 1, 2, 3],
         [4, 5, 6, 7],
         [8, 9, 10, 11]],
        coords={"x": [0, 1, 2, 3], "y": [-1, 0, 1]},
        dims=("y", "x")
    )
    expected = xr.DataArray(
        [[0, 2],
         [8, 10]],
        coords={"x": [0, 2], "y": [-1, 1]},
        dims=("y", "x")
    )
    result = shapes.reduce_density(da, 2, 2)
    assert result.equals(expected), "Reduced DataArray does not match expected downsampled result"
    assert shapes.reduce_density(da, 1, 1).equals(da), "Reduction by 1 should return the original DataArray"

def test_apply_xy_lims_to_da():
    da = xr.DataArray(
        [[0, 1, 2, 3],
         [4, 5, 6, 7],
         [8, 9, 10, 11]],
        coords={"x": [0, 1, 2, 3], "y": [-1, 0, 1]},
        dims=("y", "x")
    )
    expected_y = xr.DataArray(
        [[1, 2],
         [5, 6],
         [9, 10]],
        coords={"x": [1, 2], "y": [-1, 0, 1]},
        dims=("y", "x")
    )
    expected_x = xr.DataArray(
        [
         [4, 5, 6, 7],
        ],
        coords={"x": [0, 1, 2, 3], "y": [0]},
        dims=("y", "x")
    )
    assert shapes.apply_xy_lims_to_da(da, [1, 2], [-1, 1]).equals(expected_y), "apply_xy_lims_to_da() failed for simultaneous x/y limits"
    assert shapes.apply_xy_lims_to_da(da, [1, 2], None).equals(expected_y), "x-only limit should include all y and match combined x/y result"
    assert shapes.apply_xy_lims_to_da(da, None, [-0.5, 0.5]).equals(expected_x),  "y-only limit around zero should yield only the middle row (y=0)"