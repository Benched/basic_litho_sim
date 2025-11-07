import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    return mo, np, plt, sys, xr


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
    from goodoptics import zernikes, fourier, plot, shapes
    return fourier, plot, shapes, zernikes


@app.cell
def _():
    N = 600
    return (N,)


@app.cell
def _(N, zernikes):
    z2 = zernikes.zernike(2)
    z2.N = N
    return


@app.cell
def _():
    # distortion = zernikes.zernike_aberration({2:10, 4:180, 5:180, 7:-20, 9:30}, N=N)
    return


@app.cell
def _(np, xr):

    def grating_x_da(
        total_area=100,
        used_area=30,
        line_width=1,
        N=600,
        overall_offset_x=0,
        overall_offset_y=0,
    ):
        """
        Generate a 2D vertical grating pattern with perfectly aligned,
        full-width stripes and no edge leftovers.
        """
        # --- Grid ---
        x = np.linspace(-total_area, total_area, N, endpoint=False)
        y = np.linspace(-total_area, total_area, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # --- Correct cutoff to nearest multiple of period (2*line_width) ---
        period = 2 * line_width
        full_extent = (used_area // period) * period
        cutoff = full_extent / 2  # symmetric half-width

        # --- Define shifted coordinates ---
        Xs = X - overall_offset_x
        Ys = Y - overall_offset_y

        # --- Inside aperture ---
        inside = (np.abs(Xs) <= cutoff) & (np.abs(Ys) <= used_area)

        # --- Periodic grating mask (modulo-based, robust to float rounding) ---
        local_x = (Xs + cutoff) % (2 * line_width)
        line_mask = local_x < line_width  # 1st half = bright line

        # --- Final pattern ---
        pattern = np.where(inside & line_mask, 1.0, 0.0)

        da = xr.DataArray(
            pattern,
            coords={"x": x, "y": y},
            dims=("x", "y"),
            attrs={
                "dx": (2 * total_area / N),
                "dy": (2 * total_area / N),
                "N": N,
                "cutoff": cutoff,
                "period": period,
            },
        )

        return da
    return


@app.cell
def _(shapes, xr):
    def grating_x(total_area=100, used_area=30, line_width=1, N=600, overall_offset_x=0, overall_offset_y=0):
        iterations = used_area // line_width
        lines = []
        for i in range(iterations):
            offset_x = -used_area +  2*line_width*i
            lines.append(shapes.top_hat_2d_da(N, N, overall_offset_x+offset_x, overall_offset_x+offset_x+line_width,overall_offset_y-used_area,overall_offset_y+used_area, -total_area, total_area, -total_area, total_area))
        return xr.concat(lines, dim="i").sum('i')

    return (grating_x,)


@app.cell
def _(shapes, xr):
    def grating_y(total_area=100, used_area=30, line_width=1, N=600, overall_offset_x=0, overall_offset_y=0):
        iterations = used_area // line_width
        lines = []
        for i in range(iterations):
            offset_y = -used_area +  2*line_width*i
            lines.append(shapes.top_hat_2d_da(N, N,overall_offset_x-used_area,overall_offset_x+used_area, overall_offset_y+offset_y, overall_offset_y+offset_y+line_width, -total_area, total_area, -total_area, total_area))
        return xr.concat(lines, dim="i").sum('i')
    return (grating_y,)


@app.cell
def _(N, grating_x, grating_y):
    lines = grating_y(overall_offset_y = -40, N = N) + grating_x(overall_offset_y = 40, N=N) 
    return (lines,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Gratings and their imaging
    Starting with a basic usecase - horizontal and vertical lines we will show the impact of some imaging distortions.
    """
    )
    return


@app.cell
def _(lines, plt):
    plt.imshow(lines)
    return


@app.cell
def _(fourier, lines, shapes):
    ft_lines_zeroed = shapes.zero_outside_radius(fourier.ft2d(lines), 1.5)
    return (ft_lines_zeroed,)


@app.cell
def _(mo):
    mo.md(r"""If we take a fouriertransform of such a structure - and limit the redius slightly - we still get a very good image of the original. The plots have a reduced density compared to the actual calculations.""")
    return


@app.cell
def _(ft_lines_zeroed, plot, shapes):
    plot.complex_surface(shapes.reduce_density(ft_lines_zeroed, 2, 2))
    return


@app.cell
def _(fourier, ft_lines_zeroed):
    image_lines = fourier.ft2d(ft_lines_zeroed)
    return (image_lines,)


@app.cell
def _(image_lines, plot, shapes):
    plot.real_surface(shapes.reduce_density(abs(image_lines), 2, 2))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Zernikes
    Aberations are added via zernikes in an additional exponent (explanation in a later update).

    Let's start with a simple case of some amount of Zernike 4 - which results in a defocus.
    """
    )
    return


@app.cell
def _():
    max_zenike = 25
    return (max_zenike,)


@app.cell
def _(max_zenike, mo):
    zernike_sliders= {i : mo.ui.slider(start=-40, stop=40, step=0.2, value=0, show_value=True, full_width=True) for i in range(2, max_zenike+1)}
    return (zernike_sliders,)


@app.cell
def _(zernike_sliders):
    def split_into_columns(d: dict, ncols: int = 3):
        items = list(d.items())
        total = len(items)
        base, extra = divmod(total, ncols)

        columns = []
        start = 0
        for i in range(ncols):
            end = start + base + (1 if i < extra else 0)
            columns.append(items[start:end])
            start = end
        return columns

    columns = split_into_columns(zernike_sliders, 3)
    return (split_into_columns,)


@app.cell
def _(mo):
    apply_button = mo.ui.button(value=True, label="apply")
    return (apply_button,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Zernike sliders
    Below you can use the zernike sliders to adjsut the image 
    interesting use-cases: 

    1. Z2 = 4
        * Z2 is pure overlay
        * Also try Z3
    2. Z4 = 4
        * Z4 changes the focus for all lines
    3. Z4 = 4 and Z5 = 4
        * Z4 changes the focus for all lines, Z5 changes the focus - but it depends on orientation
    4. Z4 = 4 and Z5 = -4
        * Notice how this is the opposite of the previous experiment?
    5. Z7 = 4
        * Coma is an overlay effect - and feature dependent
        * Also try Z8

    For large overlay effects the periodic boundary becomes visible
    """
    )
    return


@app.cell
def _(apply_button, mo, split_into_columns, zernike_sliders):
    sliders_reshaped = split_into_columns(zernike_sliders)
    mo.vstack([
        mo.hstack([mo.vstack([mo.hstack([i[0], i[1]]) for i in col]) for col in sliders_reshaped]), apply_button
    ])
    return


@app.cell
def _(apply_button, zernike_sliders):
    v = apply_button.value
    zernike_values = {k: v.value for k, v in zernike_sliders.items()}
    return (zernike_values,)


@app.cell
def _(lines, zernike_values, zernikes):
    distortion = zernikes.zernike_aberration(zernike_values, N=len(lines))
    return (distortion,)


@app.cell
def _(fourier, np, shapes, zernikes):
    def fourier_optics(mask, distortion=None, radius=None):
        ft = fourier.ft2d(mask)
        if radius is None:
            radius = float(ft.x.max())
        ft_radius_applied = shapes.zero_outside_radius(ft, radius)
        if distortion is not None:
            if isinstance(distortion, dict):
                distortion = zernikes.zernike_aberration(distortion, N=len(mask.x))
            else:
                distortion.N = len(mask.x)
            distored_ft = ft_radius_applied * np.exp(complex(0, 1)*2*np.pi* distortion.sample())
            return fourier.ft2d(distored_ft)
        else:
            return fourier.ft2d(ft_radius_applied)
    return (fourier_optics,)


@app.cell
def _(distortion, fourier_optics, lines):
    lines_imaged = fourier_optics(lines, distortion=distortion, radius = None)
    return (lines_imaged,)


@app.cell
def _(lines_imaged, plt):
    plt.imshow(abs(lines_imaged))
    return


@app.cell
def _(lines_imaged, plot, shapes):
    plot.real_surface(shapes.reduce_density(abs(lines_imaged), 2))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Quick demo of a small radius 
    The individual lines fall below the threshold of what can be imaged - they all melt into the square blocks that are then the same total volume (and therefore half the height)
    """
    )
    return


@app.cell
def _(fourier_optics, lines, plot, shapes):
    plot.real_surface(
        shapes.reduce_density(
            abs(
                fourier_optics(lines, radius=.2)
            ),
        2, 2)
    )
    return


if __name__ == "__main__":
    app.run()
