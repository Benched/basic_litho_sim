import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # The purpose of these pages
    These notebooks are designed to help you get familiar with the basics of how computer chips are made, with a special focus on the optics involved in that process. We’ll use visualizations, and interactive simulations where you can turn the knobs yourself to see how light is used to build the technology inside our phones, cars, and computers.

    Currently the following notebooks are supported:

    1. [Introduction to the working of chips](./notebook_introduction.html)
        * Introduction to what a chips is - intended to be very lightweight
        * Binary logic
        * Economics and Moore's law
        * Transistors in semiconductors (work in progress)
    2. [Basic examples of fourier optics package](./notebook_optics_basics.html)
        * Shows of some of the capabilities of the Fourier optics package
    3. [First look at fourier optics on a grating usecase](./notebook_zernikes_and_gratings.html)
    """
    )
    return


if __name__ == "__main__":
    app.run()
