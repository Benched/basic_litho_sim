import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="medium",
    layout_file="layouts/notebook_fourier_design_and_test.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import inspect
    import shapes
    import utils
    import fourier
    import plot
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return fourier, inspect, mo, np, plot, plt, shapes, utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Background
    Let's first explore ow to do a 1 dimensional Fourier transform.

    The definition we will follow (there is some variation in normalization conventions) is:

    $$
    \hat{f}(k) = \int_\infty^{\infty} f(x) e^{-2 \pi i x k} dx
    $$

    We will make use of the discrete Fouerier Transform to approximate this quantitiy
    The fft algorithm is an [efficient way](https://en.wikipedia.org/wiki/Fast_Fourier_transform) to calculate the discrete Fourier transform.

    The discrete Fourier transform here is defined as:

    $$
    A_k = \sum_{m=0}^{n-1} a_m \exp\left\{ - 2 \pi i \frac{mk}{n} \right\}, \ \ k = 0,\dots,n-1
    $$

    This quantity can be calculated in $n \log(n)$ operations instead of the naive $n^2$ operations, by making use of repeating terms in the sum. We will use the implementation used in [numpy](https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Relation to Fourier transform

    First assume that f is supported on $\left(-t_0, t_0 \right)$, let $N$ be the number of intervals used for the discretizations. This then determines the time intervals as $dt = \frac{2 t_0}{N}$, and the corresponding sampling frequency as $F_s = \frac{1}{dt} = \frac{N}{2 t_0}$

    $$
    \begin{array}{ll}
    \hat{f}(l) = \int_\infty^{\infty} f(x) e^{-2 \pi i x l} dx & \approx \sum_{n=0}^{N-1} f(-t_0 + n\cdot dt) e^{-2 \pi i l(-t_0 + n \cdot dt)}\cdot dt \\
     & =  \frac{1}{F_s} \cdot \sum_{n=0}^{N-1} f\left(-t_0 + \frac{n}{F_s}\right) e^{2 \pi i l t_0} \cdot e^{-2 \pi i \frac{l n}{F_s}} \\
     & =  \frac{e^{2 \pi i l t_0}}{F_s} \cdot \sum_{n=0}^{N-1} f\left(-t_0 + \frac{n}{F_s}\right)  \cdot e^{-2 \pi i \frac{l n}{F_s}} \\
     & =  \frac{1}{F_s} \cdot e^{2 \pi i l t_0} \cdot \sum_{n=0}^{N-1} f\left(-t_0 + \frac{n}{F_s}\right)  \cdot e^{-2 \pi i \frac{l 2 t_0 n}{N}} \\
    \end{array}
    $$

    Let $k = \frac{l N}{F_s} = l 2 t_0$, an $g[n] = f\left(-t_0 + \frac{n}{F_s}\right)$ then:

    $$
    \hat{f}(l) \approx \frac{1}{F_s} \cdot e^{2 \pi i l t_0} \cdot \sum_{n=0}^{N-1} g[n]  \cdot e^{-2 \pi i \frac{n k}{N}}
    $$

    Resulting in an efficient way to approximate the continuous time Fourier transform, more background can be found [here](https://dspillustrations.com/pages/posts/misc/approximating-the-fourier-transform-with-dft.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Implementation
    The two step implementation, first a function that takes an xarray, determines $Fs$, $t_0$ and calls an internal function that no longer knows about the xarray structure and implements the above expression. Then, the result is mapped back to xarray.
    """)
    return


@app.cell
def _(fourier, inspect, mo):
    mo.md(f"""
    ```python\n{inspect.getsource(fourier.ft)}\n```
    """)
    return


@app.cell
def _(fourier, inspect, mo):
    mo.md(f"""
    ```python\n{inspect.getsource(fourier._ft)}\n```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Theoretical results
    We can test our implementation against theoretically known fourier transforms.

    ## 3.1 Theoretical results 1: top hat
    Let $f$ be the indicator function on $[\epsilon_0, \epsilon_1]$, where $0 < \epsilon_0 < \epsilon_1 < 1$, it is easy to determine that

    $$
    (\mathcal{F}f)(k) = \frac{1}{2 \pi i k}\left( e^{-2 \pi i k \epsilon_0} - e^{-2 \pi i k \epsilon_1}\right)
    $$

    where for $k=0$ the limit value of $\epsilon_0 - \epsilon_1$ should be used.
    """)
    return


@app.cell
def _(mo):
    N_input = mo.ui.slider(start=3, stop=20000, step=1,show_value=True,value=10000)
    N_input
    return (N_input,)


@app.cell
def _(mo):
    epsilon0_input = mo.ui.slider(start=-1, stop=0.98, step=0.02, show_value=True, value=-0.1)
    epsilon0_input
    return (epsilon0_input,)


@app.cell
def _(N_input, epsilon0_input):
    epsilon0 = epsilon0_input.value
    N = N_input.value
    return N, epsilon0


@app.cell
def _(epsilon0, mo):
    epsilon1_input = mo.ui.slider(start=epsilon0, stop=1, step=0.02, show_value=True, value=max(0.2, epsilon0))
    epsilon1_input
    return (epsilon1_input,)


@app.cell
def _(epsilon1_input):
    epsilon1 = epsilon1_input.value
    return (epsilon1,)


@app.cell
def _(N, epsilon0, epsilon1, shapes):
    tophat = shapes.top_hat_da(N, epsilon0, epsilon1, -1, 1)
    tophat.plot()
    return (tophat,)


@app.cell
def _(fourier, tophat):
    # implementation in these tools
    tophat_transform = fourier.ft(tophat)
    return (tophat_transform,)


@app.cell
def _(N, epsilon0, epsilon1, np):
    # theoretical results
    k0 = np.linspace(-N / 4, N / 4, N)
    with np.errstate(divide='ignore', invalid='ignore'):
        ft_theory_th = np.zeros(N, dtype=complex)
        ft_theory_th = np.divide(1, 2*np.pi*complex(0, 1)*k0) * (np.exp(-2*np.pi*complex(0, 1)*k0*epsilon0) - np.exp(-2*np.pi*complex(0, 1)*k0*epsilon1))

    if 0 in k0:
        idx = np.where(k0 == 0)[0][0]
        ft_theory_th[idx] = epsilon1 - epsilon0
    return ft_theory_th, k0


@app.cell(hide_code=True)
def _(ft_theory_th, k0, np, plt, tophat_transform):
    def plot_compare(plot1, plot2):

        fig, (ax1, ax2) = plt.subplots(1, 2)

        np.real(plot1).plot(ax=ax1)
        ax1.plot(k0, np.real(plot2), linestyle='--')
        ax1.set_xlim(-80, 80);
        ax1.title.set_text("Real")

        np.imag(plot1).plot(ax=ax2)
        ax2.plot(k0, np.imag(plot2), linestyle='--')
        #np.imag(qd_theory).plot(linestyle='--', ax=ax2)
        ax2.set_xlim(-80, 80);
        ax2.title.set_text("Imaginary")

        fig.set_size_inches(10, 5)
        fig.suptitle("Real transform compared to approximation: tophat")
        return fig

    plot_compare(tophat_transform, ft_theory_th)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Theoretical results 2: Quadratic decay
    Let $f:\mathbb{R} \to \mathbb{R}$ be given by

    $$
    f(x) = \dfrac{1}{x^2+1}
    $$

    then:

    $$
    (\mathcal{F}f)(k) = \pi e^{-2 \pi |k|}
    $$

    The function looks as follows:
    """)
    return


@app.cell
def _(mo):
    Nqd_input = mo.ui.slider(start=3, stop=20000, step=1,show_value=True,value=10000)
    Nqd_input
    return (Nqd_input,)


@app.cell
def _(Nqd_input, np, plt, shapes):
    Nqd = Nqd_input.value
    qd = shapes.quadratic_decay_da(Nqd, -np.sqrt(Nqd), np.sqrt(Nqd)) # reason for the large range is that makes the resulting transform near the origin more accurate.
    qd.plot()
    plt.xlim(-10, 10)
    plt.title("Quadriatic Decay function")
    return Nqd, qd


@app.cell
def _(fourier, qd):
    qd_ft = fourier.ft(qd)
    return (qd_ft,)


@app.cell
def _(Nqd, np, utils):
    qd_theory_fn = lambda w: np.pi * np.exp(- 2*np.pi * np.abs(w))
    qd_theory = utils.sample_fn(qd_theory_fn, Nqd, -3,  3)
    return (qd_theory,)


@app.cell(hide_code=True)
def _(np, plt, qd_ft, qd_theory):
    def plot_qd(qd_theory, qd_ft):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        np.real(qd_ft).plot(ax=ax1)
        np.real(qd_theory).plot(linestyle='--', ax=ax1)
        ax1.set_xlim(-2, 2);
        ax1.title.set_text("Real")

        np.imag(qd_ft).plot(ax=ax2)
        np.imag(qd_theory).plot(linestyle='--', ax=ax2)
        ax2.set_xlim(-3, 3);
        ax2.set_ylim(-1, 1)
        ax2.title.set_text("Imaginary")

        fig.set_size_inches(10, 5)
        fig.suptitle("Real transform compared to approximation: quadratic decay")
        return fig

    plot_qd(qd_theory, qd_ft)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 Inversion
    In this section we verify that the transform satisfies the following theoretical property:

    $$
    \mathcal{F}[\mathcal{F}[f(t)]](\omega) = f(-t)
    $$

    That is, applying the fourier transform twice results in the original function, with an additional minus sign on its input variable. Meaning applying a fourier transform twice mirrors a function in 0.
    """)
    return


@app.cell
def _(mo):
    Ninv_slider = mo.ui.slider(start=3, stop=20000, step=1,show_value=True,value=10000)
    Ninv_slider
    return (Ninv_slider,)


@app.cell
def _(Ninv_slider, fourier, mo, np, utils):
    Ninv = Ninv_slider.value
    assym_fn = lambda x : np.sin(x)*(x + 1)**2 / ((x/ 10)**6 + 1) / 63.67921267
    assym_fn_reversed = lambda x : assym_fn(-x)
    sampled = utils.sample_fn(assym_fn, Ninv, -np.sqrt(Ninv), np.sqrt(Ninv))
    double_ft = fourier.ft(fourier.ft(sampled))
    double_ft_domain = double_ft.x.values
    reverse_sample_image = list(map(assym_fn_reversed, double_ft_domain))
    reverse_sample = utils.to_da(double_ft_domain, reverse_sample_image)
    mo.md(f"abs max error of the double fourier transform: {float(abs(reverse_sample - double_ft).max())}")
    return double_ft, reverse_sample, sampled


@app.cell(hide_code=True)
def _(plt, sampled):
    figure = sampled.plot()
    plt.xlim(-100, 100)
    plt.xlabel("x")
    plt.title("Non-trivial assymetrical function")
    figure
    return


@app.cell(hide_code=True)
def _(double_ft, np, plt, reverse_sample):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    np.real(reverse_sample).plot(ax=ax1)
    np.real(double_ft).plot(linestyle='--', ax=ax1)
    ax1.title.set_text("Real")
    ax1.set_xlim(-75, 75)

    np.imag(reverse_sample).plot(ax=ax2)
    np.imag(double_ft).plot(linestyle='--', ax=ax2)
    ax2.title.set_text("Imaginary")
    ax2.set_xlim(-75, 75)
    ax2.set_ylim(-1, 1)

    ax1.figure.suptitle('Double fourier transform compared with inversed function')
    fig.set_size_inches(10, 5)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. 2D Fourier transforms
    The two dimensional transform works in a similar fashion:

    $$
    \hat{f}(k, l) = \int_\infty^{\infty} \int_\infty^{\infty} f(x, y) e^{-2 \pi i (x \cdot k + y \cdot l)} dx dy
    $$

    for $f = \mathbb{1}_{[\epsilon_0, \epsilon_1] \times [\delta_0, \delta_1]} = \mathbb{1}_{[\epsilon_0, \epsilon_1]}(x) \cdot \mathbb{1}_{ [\delta_0, \delta_1]}(y)$ we get

    $$
    \begin{array}{ll}
    \hat{f}(k, l) & = \int_\infty^{\infty} \int_\infty^{\infty} f(x, y) e^{-2 \pi i (x \cdot k + y \cdot l)} dx dy \\
    & = \int_\infty^{\infty} \int_\infty^{\infty} \mathbb{1}_{[\epsilon_0, \epsilon_1] \times [\delta_0, \delta_1]}(x, y) e^{-2 \pi i (x \cdot k + y \cdot l)} dx dy \\
    & =  \int_\infty^{\infty}  \mathbb{1}_{[\epsilon_0, \epsilon_1]}(x) e^{-2 \pi i x \cdot k} dx \int_\infty^{\infty} \mathbb{1}_{ [\delta_0, \delta_1]}(y)  e^{-2 \pi i y \cdot l} dy \\
    & = \frac{-1}{4 \pi^2 k l} \left( e^{-2 \pi i k \epsilon_0} - e^{-2 \pi i k \epsilon_1}\right) \left( e^{-2 \pi i l \delta_0} - e^{-2 \pi i l \delta_1}\right)
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sampling: $nx$ and $ny$
    """)
    return


@app.cell
def _(mo):
    nx_slider = mo.ui.slider(start=3, stop=2047, step=1,show_value=True,value=1024, label="nx")
    nx_slider
    return (nx_slider,)


@app.cell
def _(nx_slider):
    nx = nx_slider.value
    return (nx,)


@app.cell
def _(mo):
    ny_slider = mo.ui.slider(start=3, stop=2047, step=1,show_value=True,value=1024, label="ny")
    ny_slider
    return (ny_slider,)


@app.cell
def _(ny_slider):
    ny = ny_slider.value
    return (ny,)


@app.cell
def _(mo):
    mo.md(r"""
    The domain of the tophat function: $x_{\min}, x_{\max}, y_{\min}, y_{\max}$.
    """)
    return


@app.cell
def _(mo):
    xmin_slider = mo.ui.slider(start=-30, stop=0, step=1, show_value=True, value=-10, label="x-min (domain)")
    xmin_slider
    return (xmin_slider,)


@app.cell
def _(mo):
    xmax_slider = mo.ui.slider(start=0, stop=30, step=1, show_value=True, value=10, label="x-max (domain")
    xmax_slider
    return (xmax_slider,)


@app.cell
def _(mo):
    ymin_slider = mo.ui.slider(start=-30, stop=0, step=1, show_value=True, value=-10, label="y-min (domain)")
    ymin_slider
    return (ymin_slider,)


@app.cell
def _(mo):
    ymax_slider = mo.ui.slider(start=0, stop=30, step=1, show_value=True, value=10, label="y-max (domain)")
    ymax_slider
    return (ymax_slider,)


@app.cell
def _(xmax_slider, xmin_slider, ymax_slider, ymin_slider):
    xmin, xmax = xmin_slider.value, xmax_slider.value
    ymin, ymax = ymin_slider.value, ymax_slider.value
    return xmax, xmin, ymax, ymin


@app.cell
def _(mo):
    x0_slider = mo.ui.slider(start=-5, stop=0, step=0.1, show_value=True, value=0, label="tophat x-min (support)")
    x1_slider = mo.ui.slider(start=0, stop=5, step=0.1, show_value=True, value=2.2, label="tophat x-max (support)")
    y0_slider = mo.ui.slider(start=-5, stop=0, step=0.1, show_value=True, value=-1, label="tophat y-min (support)")
    y1_slider = mo.ui.slider(start=0, stop=5, step=0.1, show_value=True, value=1.8, label="tophat y-max (support)")
    return x0_slider, x1_slider, y0_slider, y1_slider


@app.cell
def _(x0_slider):
    x0_slider
    return


@app.cell
def _(x1_slider):
    x1_slider
    return


@app.cell
def _(y0_slider):
    y0_slider
    return


@app.cell
def _(y1_slider):
    y1_slider
    return


@app.cell
def _(x0_slider, x1_slider, y0_slider, y1_slider):
    x0, x1 = x0_slider.value, x1_slider.value
    y0, y1 = y0_slider.value, y1_slider.value
    return x0, x1, y0, y1


@app.cell
def _(mo):
    mo.md(r"""
    Shift the slice that is being plotted by a single sampling step.
    """)
    return


@app.cell
def _(mo, nx):
    x_shift_slider = mo.ui.slider(start=-nx//2, stop=nx//2, step=1, show_value=True, value=0, label="shift slice by")
    return (x_shift_slider,)


@app.cell
def _(fourier, mo, np, nx, ny, shapes, x0, x1, xmax, xmin, y0, y1, ymax, ymin):
    # --- Generate signal ---
    signal = shapes.top_hat_2d_da(nx, ny, x0, x1, y0, y1, xmin, xmax, ymin, ymax)

    # --- Compute 2D Fourier Transform ---
    ft_numerical = fourier.ft2d(signal)

    # --- Theoretical FT ---
    kx = ft_numerical.coords["x"].values
    ky = ft_numerical.coords["y"].values
    ft_theory = shapes.ft_tophat_2d(kx, ky, x0, x1, y0, y1)

    # --- Compare Results ---
    error = np.abs(ft_numerical.values - ft_theory.values)
    max_relative_error = np.max(error) / abs(ft_theory).max()
    mo.md(f"Max error: {max_relative_error:.2e}")
    return ft_numerical, ft_theory, kx, ky


@app.cell
def _(ft_numerical, ft_theory, np, nx, plt, x_shift_slider):
    fig_2d, (ax3, ax4) = plt.subplots(1, 2)

    np.real(ft_numerical.isel(x=[nx//2+x_shift_slider.value])).plot(hue='x', ax=ax3);
    np.real(ft_theory.isel(x=[nx//2+x_shift_slider.value])).plot(hue='x', linestyle='--', ax=ax3);
    ax3.set_xlim(-5, 5)
    ax3.figure.set_size_inches(10, 5)
    ax3.title.set_text("Real part")

    np.imag(ft_numerical.isel(x=[nx//2+x_shift_slider.value])).plot(hue='x', ax=ax4);
    np.imag(ft_theory.isel(x=[nx//2+x_shift_slider.value])).plot(hue='x', linestyle='--', ax=ax4);
    ax4.set_xlim(-5, 5)
    ax4.title.set_text("Imaginary part")

    ax3.figure.suptitle("Compare the 2d transform  for a few slices to the theoretical result")
    fig_2d
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Plot of numerical and theoretical transform
    """)
    return


@app.cell
def _(ft_numerical, plot, shapes):
    plot.complex_surface(
        shapes.reduce_density(
            shapes.apply_xy_lims_to_da(ft_numerical, [-2, 2], [-2, 2])
            , 2, 2)
    )
    return


@app.cell
def _(ft_theory, plot, shapes):
    plot.complex_surface(
        shapes.reduce_density(
            shapes.apply_xy_lims_to_da(ft_theory, [-2, 2], [-2, 2])
            , 2, 2)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2 Gaussian
    Another test case for a two dimensional gaussian function, the variance in x is twice the variance in y.
    """)
    return


@app.cell
def _(mo):
    nx_gauss_slider = mo.ui.slider(start=3, stop=2047, step=1,show_value=True,value=1024, label="nx")
    ny_gauss_slider = mo.ui.slider(start=3, stop=2047, step=1,show_value=True,value=1024, label="ny")
    return nx_gauss_slider, ny_gauss_slider


@app.cell
def _(mo):
    mo.md(r"""
    The number of samples in the x and y directions, $nx$, $ny$.
    """)
    return


@app.cell
def _(nx_gauss_slider):
    nx_gauss_slider
    return


@app.cell
def _(ny_gauss_slider):
    ny_gauss_slider
    return


@app.cell
def _(nx_gauss_slider, ny_gauss_slider):
    nx_gauss = nx_gauss_slider.value
    ny_gauss = ny_gauss_slider.value
    return (nx_gauss,)


@app.cell
def _():
    #gaussian_2d_sample = shapes.gaussian_2d_da(nx, ny, xmin, xmax, ymin, ymax, sigma_x=2)
    #ft_gaussian_2d_numerical = fourier.ft2d(gaussian_2d_sample)
    return


@app.cell
def _(mo):
    mo.md(r"""
    \[
    \mathcal{F}\{f\}(k_x, k_y)
    = N \exp\!\left[
       -2\pi^2\left(\sigma_x^2 k_x^2 + \sigma_y^2 k_y^2\right)
      \right]
      \exp\!\left[-2\pi i \left(k_x x_0 + k_y y_0\right)\right]
    \]
    """)
    return


@app.cell
def _(fourier, kx, ky, mo, np, nx, ny, shapes, xmax, xmin, ymax, ymin):
    gaussian_2d_sample = shapes.gaussian_2d_da(nx, ny, xmin, xmax, ymin, ymax, sigma_x=2)
    ft_gaussian_2d_numerical = fourier.ft2d(gaussian_2d_sample)

    kx_1 = ft_gaussian_2d_numerical.coords["x"].values
    ky_1 = ft_gaussian_2d_numerical.coords["y"].values
    ft_theory_gauss = shapes.ft_gaussian_2d(kx, ky,sigma_x=2)

    error_gauss = np.abs(ft_gaussian_2d_numerical.values - ft_theory_gauss)
    max_relative_error_gauss = np.max(error_gauss) / abs(ft_theory_gauss).max()
    mo.md(f"Max error: {max_relative_error_gauss:.2e}")
    return ft_gaussian_2d_numerical, ft_theory_gauss


@app.cell
def _(mo, nx_gauss):
    x_shift_slider_gauss = mo.ui.slider(start=-nx_gauss//2, stop=nx_gauss//2, step=1, show_value=True, value=0, label="shift slice by")
    x_shift_slider_gauss
    return (x_shift_slider_gauss,)


@app.cell
def _(
    ft_gaussian_2d_numerical,
    ft_theory_gauss,
    np,
    nx_gauss,
    plt,
    x_shift_slider_gauss,
):
    fig_2d_gauss, (ax5, ax6) = plt.subplots(1, 2)

    np.real(ft_gaussian_2d_numerical.isel(x=[nx_gauss//2+x_shift_slider_gauss.value])).plot(hue='x', ax=ax5);
    np.real(ft_theory_gauss.isel(x=[nx_gauss//2+x_shift_slider_gauss.value])).plot(hue='x', linestyle='--', ax=ax5);
    ax5.set_xlim(-5, 5)
    ax5.figure.set_size_inches(10, 5)
    ax5.title.set_text("Real part")

    np.imag(ft_gaussian_2d_numerical.isel(x=[nx_gauss//2+x_shift_slider_gauss.value])).plot(hue='x', ax=ax6);
    np.imag(ft_theory_gauss.isel(x=[nx_gauss//2+x_shift_slider_gauss.value])).plot(hue='x', linestyle='--', ax=ax6);
    ax6.set_xlim(-5, 5)
    ax6.title.set_text("Imaginary part")

    ax5.figure.suptitle("Compare the 2d transform  for a few slices to the theoretical result")
    fig_2d_gauss
    return


@app.cell
def _():
    ### Numerical vs theoretical results
    return


@app.cell
def _(ft_gaussian_2d_numerical, plot, shapes):
    plot.complex_surface(shapes.apply_xy_lims_to_da(ft_gaussian_2d_numerical, [-1, 1],[-1, 1]))
    return


@app.cell
def _(ft_theory_gauss, plot, shapes):
    plot.complex_surface(shapes.apply_xy_lims_to_da(ft_theory_gauss, [-1, 1], [-1, 1]))
    return


if __name__ == "__main__":
    app.run()
