import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import zernikes
    from zernikes import zernike, plot_zernike_triangle, plotly_zernike_pyramid, fringe_to_nm, nm_to_fringe, zernike_aberration, fit_zernikes_to_da
    import numpy as np
    import xarray as xr
    import math
    import matplotlib.pyplot as plt
    import plot
    return (
        fit_zernikes_to_da,
        mo,
        plot_zernike_triangle,
        plotly_zernike_pyramid,
        zernike,
        zernike_aberration,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    A lithography system diffracts light on a mask. The diffraction orders are then projected by a system of lenses or mirrors onto the photoresist layer on a silicon wafer. 

    In reality the lens is not perfect. We can imagine an ideal case to mean that for some plane in the lens, called a "pupil plane", all the light is traveling perfectly in phase. In that case an imperfection means that the phase of the light is ahead or behind of the the average phase in that plane.

    Think of the following visualization:
    """
    )
    return


@app.cell
def _(zernike_aberration):
    ab = zernike_aberration({2:1e-9, 9:5e-10, 8:-1e-9})
    return (ab,)


@app.cell
def _(ab, fit_zernikes_to_da):
    result = fit_zernikes_to_da(ab.sample_da(), 25)
    return (result,)


@app.cell
def _(ab, result):
    result.coefficients - ab.coefficients
    return


@app.cell
def _(ab):
    ab.coefficients
    return


@app.cell
def _(zernike_aberration):
    ab = zernike_aberration({2:1e-9, 9:5e-10, 8:-1e-9})
    ab2 = zernike_aberration({2:1e-9, 8:-1e-9, 12:-8e-9})
    return ab, ab2


@app.cell
def _(ab):
    ab.add_zernikes({7:1e-9})
    return


@app.cell
def _(ab):
    zer = ab.zernikes[0]
    return (zer,)


@app.cell
def _(zer):
    zer.plot_surface()
    return


@app.cell
def _(ab, ab2):
    (ab - ab2)
    return


@app.cell
def _(ab):
    ab.plot_surface(cmap="Viridis")
    return


@app.cell
def _(ab, ab2):
    (ab + ab2).plot_surface()
    return


@app.cell
def _(ab, ab2):
    ab2 + ab
    return


@app.cell
def _(ab):
    ab.add_zernikes({25:1e-9, 19:2e-10})
    return


@app.cell
def _(ab):
    ab.coefficients
    return


@app.cell
def _(ab):
    ab.plot_surface()
    return


@app.cell
def _(ab):
    ab.sample()
    return


@app.cell
def _(ab):
    ab.coefficients.sortby('zernike')
    return


@app.cell
def _(zernike):
    zernike(9).plot_surface()
    return


@app.class_definition
class zernike_aberrations():
    def __init__(self):
        self.zernike_coefficeints = []


@app.cell
def _(zernike):
    zernike(2, -2).plot()
    return


@app.cell
def _(zernike):
    z = zernike(23, 3)
    z.plot()
    return (z,)


@app.cell
def _(z):
    z.plot_surface()
    return


@app.cell
def _(plot_zernike_triangle):
    plot_zernike_triangle()
    return


@app.cell
def _(plotly_zernike_pyramid):
    plotly_zernike_pyramid()
    return


if __name__ == "__main__":
    app.run()
