import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import numpy as np
    import xarray as xr
    import math
    import matplotlib.pyplot as plt
    return mo, sys


@app.cell
async def _(sys):
    wheel_name = "goodoptics-0.1.0-py3-none-any.whl"
    wheel_remote = "https://benched.github.io/basic_litho_sim/" + wheel_name

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install("plotly")
        await micropip.install("sympy")
        if "goodoptics" not in sys.modules:
            await micropip.install(wheel_remote)
    else:
        from pathlib import Path

        candidate_paths = []
        notebook_file = globals().get("__file__")
        if notebook_file is not None:
            candidate_paths.append(Path(notebook_file).resolve().parents[1] / "src")
        candidate_paths.extend([Path.cwd() / "src", Path.cwd().parent / "src"])

        for src_path in candidate_paths:
            if src_path.exists():
                src_path_str = str(src_path)
                if src_path_str not in sys.path:
                    sys.path.insert(0, src_path_str)
                break
        else:
            raise ModuleNotFoundError(
                "Could not locate src/goodoptics for local notebook execution."
            )

    goodoptics_ready = True
    return (goodoptics_ready,)


@app.cell
def _(goodoptics_ready):
    from goodoptics.zernikes import (
        fit_zernikes_to_da,
        plot_zernike_triangle,
        plotly_zernike_pyramid,
        zernike,
        zernike_aberration,
    )

    return (
        fit_zernikes_to_da,
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
    ab_fit = zernike_aberration({2:1e-9, 9:5e-10, 8:-1e-9})
    return (ab_fit,)


@app.cell
def _(ab_fit, fit_zernikes_to_da):
    result = fit_zernikes_to_da(ab_fit.sample_da(), 25)
    return (result,)


@app.cell
def _(ab_fit, result):
    result.coefficients - ab_fit.coefficients
    return


@app.cell
def _(ab_fit):
    ab_fit.coefficients
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
