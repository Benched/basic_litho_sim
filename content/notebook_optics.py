import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import shapes
    import fourier
    import zernikes
    import plot
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    return fourier, np, plot, plt, shapes, xr, zernikes


@app.cell
def _(shapes):
    tophat = shapes.top_hat_2d_da(400, 400, -1, 1, -1, 1, -2, 2, -2, 2)
    return (tophat,)


@app.cell
def _(plot, tophat):
    plot.real_surface(tophat)
    return


@app.cell
def _(fourier, tophat):
    ft_tophat = fourier.ft2d(tophat)
    return (ft_tophat,)


@app.cell
def _(ft_tophat, plot):
    plot.complex_surface(ft_tophat)
    return


@app.cell
def _(fourier, ft_tophat):
    tophat_image = fourier.ft2d(ft_tophat)
    return (tophat_image,)


@app.cell
def _(plot, tophat_image):
    plot.complex_surface(tophat_image)
    return


@app.cell
def _(fourier, ft_tophat, plot, shapes):
    plot.complex_surface(
        fourier.ft2d(
            shapes.zero_outside_radius(ft_tophat, 4)
        )
    )
    return


@app.cell
def _(fourier, ft_tophat, plot, shapes):
    plot.complex_surface(
        fourier.ft2d(
            shapes.zero_outside_radius(ft_tophat, 2)
        )
    )
    return


@app.cell
def _(fourier, ft_tophat, plot, shapes):
    plot.complex_surface(
        fourier.ft2d(
            shapes.zero_outside_radius(ft_tophat, 1)
        )
    )
    return


@app.cell
def _(fourier, ft_tophat, plot, shapes):
    plot.complex_surface(
        fourier.ft2d(
            shapes.zero_outside_radius(ft_tophat, 0.5)
        )
    )
    return


@app.cell
def _(ft_tophat):
    ft_tophat
    return


@app.cell
def _(fourier, ft_tophat, plot, shapes):
    rad4_ft = fourier.ft2d(
            shapes.zero_outside_radius(ft_tophat, 4)
        )
    plot.complex_surface(
        rad4_ft
    )
    return (rad4_ft,)


@app.cell
def _(np, rad4_ft, zernikes):
    fit = zernikes.fit_zernikes_to_da(np.real(rad4_ft), 120)
    return (fit,)


@app.cell
def _(fit):
    fit.plot_surface()
    return


@app.cell
def _(ft_tophat, shapes):
    ft_tophat_zeroed = shapes.zero_outside_radius(ft_tophat, 4)
    return (ft_tophat_zeroed,)


@app.cell
def _(ft_tophat_zeroed):
    ft_tophat_zeroed
    return


@app.cell
def _(zernikes):
    z2 = zernikes.zernike(2)
    z2.N = 400
    return


@app.cell
def _(zernikes):
    distortion = zernikes.zernike_aberration({2:10, 4:180, 5:180, 7:-20, 9:30}, N=400)
    return (distortion,)


@app.cell
def _():
    return


@app.cell
def _(distortion, fourier, ft_tophat_zeroed, np):
    results_z2 = fourier.ft2d(
        ft_tophat_zeroed * np.exp(complex(0, 1)*2*np.pi* distortion.sample())
    )
    return (results_z2,)


@app.cell
def _():
    return


@app.cell
def _(distortion, ft_tophat_zeroed, np, plot):
    plot.complex_surface(ft_tophat_zeroed * np.exp(complex(0, 1)*2*np.pi* distortion.sample()))
    return


@app.cell
def _(plot, results_z2):
    plot.complex_surface(
        abs(results_z2)
    )
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
    return (grating_x_da,)


@app.cell
def _(grating_x_da, plt):
    plt.imshow(grating_x_da(line_width = 3))
    return


@app.cell
def _(grating_x):
    old_implementation = grating_x(line_width=3)
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
def _(grating_x, grating_y):
    lines = grating_y(overall_offset_y = -40) + grating_x(overall_offset_y = 40) 
    return (lines,)


@app.cell
def _(lines, plt):
    plt.imshow(lines)
    return


@app.cell
def _(ft_lines_zeroed):
    ft_lines_zeroed.x.max()
    return


@app.cell
def _(fourier, lines, shapes):

    ft_lines_zeroed = shapes.zero_outside_radius(fourier.ft2d(lines), 1.5)
    return (ft_lines_zeroed,)


@app.cell
def _(ft_lines_zeroed, plot):
    plot.complex_surface(ft_lines_zeroed)
    return


@app.cell
def _(fourier, ft_lines_zeroed):
    image_lines = fourier.ft2d(ft_lines_zeroed)
    return (image_lines,)


@app.cell
def _(image_lines, plot):
    plot.real_surface(abs(image_lines))
    return


@app.cell
def _(lines, zernikes):
    distortion_lines = zernikes.zernike_aberration({2:0, 4:4, 5:-4}, N=len(lines))
    return (distortion_lines,)


@app.cell
def _(distortion_lines, ft_lines_zeroed, np):
    distored_ft_lines_zeroed = ft_lines_zeroed * np.exp(complex(0, 1)*2*np.pi* distortion_lines.sample())
    return (distored_ft_lines_zeroed,)


@app.cell
def _():
    #plot.complex_surface(distored_ft_lines_zeroed)
    return


@app.cell
def _(distored_ft_lines_zeroed, fourier, plot):
    plot.real_surface(abs(fourier.ft2d(distored_ft_lines_zeroed)))
    return


@app.cell
def _(distored_ft_lines_zeroed, fourier, plt):
    plt.imshow(abs(fourier.ft2d(distored_ft_lines_zeroed)))
    return


@app.cell
def _(ft_lines_zeroed, lines, np, zernikes):
    distortion_lines_2 = zernikes.zernike_aberration({2:0, 4:4, 5:4}, N=len(lines))
    distored_ft_lines_zeroed_2 = ft_lines_zeroed * np.exp(complex(0, 1)*2*np.pi* distortion_lines_2.sample())
    return (distored_ft_lines_zeroed_2,)


@app.cell
def _(distored_ft_lines_zeroed_2, fourier, plt):
    plt.imshow(abs(fourier.ft2d(distored_ft_lines_zeroed_2)))
    return


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
def _(fourier_optics, lines, plt):
    plt.imshow(
        abs(
            fourier_optics(lines, {4:0.4, 5:0.4})
        )
    )
    return


@app.cell
def _(fourier_optics, lines, plt):
    plt.imshow(
        abs(
            fourier_optics(lines, {2:20})
        )
    )
    return


@app.cell
def _(fourier_optics, lines, plt):
    plt.imshow(
        abs(
            fourier_optics(lines, {7:5})
        )
    )

    return


@app.cell
def _(fourier_optics, lines, plt):
    plt.imshow(
        abs(
            fourier_optics(lines, {8:5})
        )
    )
    return


@app.cell
def _(fourier_optics, lines, plt):
    plt.imshow(
        abs(
            fourier_optics(lines, {6:5})
        )
    )
    return


@app.cell
def _(fourier_optics, lines, plt):
    plt.imshow(
        abs(
            fourier_optics(lines, {5:5})
        )
    )
    return


@app.cell
def _(fourier_optics, lines, plt):
    plt.imshow(
        abs(
            fourier_optics(lines, {4:5})
        )
    )
    return


@app.cell
def _(fourier_optics, lines, plot):
    plot.real_surface(
        abs(
            fourier_optics(lines, {4:5, 9:5})
        )
    )
    return


@app.cell
def _(fourier_optics, lines, plot):
    plot.real_surface(
        abs(
            fourier_optics(lines, {4:5, 9:0})
        )
    )
    return


@app.cell
def _(fourier_optics, lines):
    (abs(
        fourier_optics(lines, {4:5, 9:0})
    )>0.8).mean()
    return


@app.cell
def _(fourier_optics, lines):
    (abs(
        fourier_optics(lines, {4:5, 9:5})
    )>0.8).mean()
    return


@app.cell
def _(fourier_optics, lines, plot):
    plot.real_surface(
        abs(
            fourier_optics(lines, radius=.5)
        )
    )
    return


@app.cell
def _(fourier_optics, lines, plot):
    plot.real_surface(
        abs(
            fourier_optics(lines, radius=.2)
        )
    )
    return


if __name__ == "__main__":
    app.run()
