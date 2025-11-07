import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import sys
    return mo, np, sys


@app.cell
async def _(sys):
    wheel_name = "goodoptics-0.1.0-py3-none-any.whl"
    wheel_local = "./dist/" + wheel_name
    wheel_remote = "https://benched.github.io/basic_litho_sim/" + wheel_name

    is_pyodide = "pyodide" in sys.modules

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("plotly")
        await micropip.install("sympy")
        await micropip.install(wheel_remote)
    else:
        import subprocess, pathlib
        import os
        wheel = wheel_local # wheel_local if os.path.exists(wheel_local) else wheel_remote
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", wheel], check=True)
    return


@app.cell
def _():
    from goodoptics import utils, shapes, fourier, plot, zernikes
    return fourier, plot, shapes, zernikes


@app.cell
def _(mo):
    mo.md(
        r"""
    # Fourier and fourieroptics demo
    In this notebook we play with a very basic shape to demonstrate some of the basic properties of fourieroptics on a simple shape.
    """
    )
    return


@app.cell
def _(plot, shapes):
    tophat = shapes.top_hat_2d_da(100, 100, -1, 1, -1, 1, -2, 2, -2, 2)
    plot.real_surface(tophat)
    return (tophat,)


@app.cell
def _(mo):
    mo.md(r"""In fourierspace we get purely real transformation - the imaginary part has only very small values.""")
    return


@app.cell
def _(fourier, plot, tophat):
    ft_tophat = fourier.ft2d(tophat)
    plot.complex_surface(ft_tophat)
    return (ft_tophat,)


@app.cell
def _(mo):
    mo.md(r"""A second fouriertransform gives us back the original shape.""")
    return


@app.cell
def _(fourier, ft_tophat, plot):
    tophat_image = fourier.ft2d(ft_tophat)
    plot.complex_surface(tophat_image)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### NA!
    Interestingly - we can get a pretty decent version of the original by throwing away the high frequent content. This is somewhat similar to having a limited NA.
    """
    )
    return


@app.cell
def _(ft_tophat, plot, shapes):
    plot.complex_surface(shapes.zero_outside_radius(ft_tophat, 4))
    return


@app.cell
def _(mo):
    mo.md(r"""We do indeed recover the original fairly well. Try the slider for different values - you will see that the level of detail decreases the smaller the radius.""")
    return


@app.cell
def _(ft_tophat, mo, np):
    max_radius = np.ceil(np.sqrt(2) * float(abs(ft_tophat.x).max()))
    radius_slider = mo.ui.slider(start=0, stop=max_radius, step=.1, value=4)
    return (radius_slider,)


@app.cell
def _(fourier, ft_tophat, mo, plot, radius_slider, shapes):
    mo.vstack([
        radius_slider,
        plot.complex_surface(
        fourier.ft2d(
            shapes.zero_outside_radius(ft_tophat, radius_slider.value)
        )
    )
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Bonus - Zernike fit demo
    Just to demo some of the zernike tooling! In later notebooks we will make more use of Zernike polynomials - they can be used to fit real functions on the unit disk. Here we show how that works with the tools in this repository:
    """
    )
    return


@app.cell
def _(mo):
    fit_order = mo.ui.slider(start=1, stop=120, step=1, value=49, show_value=True, debounce=True)
    return (fit_order,)


@app.cell
def _(fit_order, mo, tophat, zernikes):
    fit = zernikes.fit_zernikes_to_da(tophat, int(fit_order.value))
    mo.vstack(
        [
        fit_order,
        fit.plot_surface()
        ]
    )
    return


if __name__ == "__main__":
    app.run()
