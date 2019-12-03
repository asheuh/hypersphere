"""
imports
"""
import click
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_coordinates(d: int, r: int) -> np.ndarray:
    """
    Parameters
    ----------
    d, r: int
        d is the number of dimentions
        r is the radius

    Returns
    -------
    out: ndarray, shape
        Random values
    """
    return np.random.uniform(-1, r, d)


def circle(n: int, d: int, R: int) -> tuple:
    """
    Parameters
    ----------
    n, d, R: int
        n represents the number of random points
        d is the number of dimentions
        R is the radius

    Returns
    -------
    out : ndarray, ndarray, float
        x and y samples to draw and an approximation of py
    """
    # Fixing random state for reproducibility
    np.random.seed(1)

    rejected = np.zeros((int(1e6), d))
    rho = []

    for i in range(n):
        # sample random coordinates for each d-dimentions
        coords = get_coordinates(d, R)

        # check if the point is inside the circle
        # r**2 = x1**2 + x2**2 + ... + xd**2
        r = np.sqrt(sum(coords**2)) * 1
        rho.append(r)

        if r > 1:
            rejected[i, ] = coords

    theta = get_coordinates(n, 2 * np.pi)
    rho = np.array(rho)

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    accepted = np.stack((x, y))
    df = pd.DataFrame(accepted)
    _, ncol = df.shape

    rejected = rejected[~np.all(rejected == 0, axis=1)]
    dr = pd.DataFrame(rejected)

    # Calculate the acceptance rate
    accept_rate = ncol / (ncol + len(dr.index))
    approx_pi = accept_rate * 4  # Approximation of pi if needed to explain how the algo works

    return x, y, approx_pi


def sphere(n: int, d: int) -> np.ndarray:
    """
    Parameters
    ----------
    n, d: int
        n represents the number of random points
        d is the number of dimentions

    Returns
    -------
    out : ndarray
        Sampled D vectors of N Gaussian coordinates
    """

    samples = np.random.normal(size=(n, d))

    # Normalise all distances (radii) to 1
    norm = samples / np.linalg.norm(samples)

    return np.reshape(norm, (d, n))


@click.command()
@click.argument('n')
@click.argument('radius')
def two_d(n: int, radius: int):
    """
    Plots a two dimentional (d-1)-sphere
    where d is the number of dimentions

    Parameters
    ----------
    n: int
        n represents the number of random points
    """
    n, r = int(n), float(radius)
    x, y, approx_pi = circle(n, 2, r)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set(title=f"Approximation estimate of PI = {approx_pi}")
    ax.scatter(x, y, c="tab:blue", alpha=0.8)

    if r < 0 or r > 1:
        print("Radius should range from 0 to 1 (not less 0 or greater than 1)")
    else:
        plt.show()


@click.command()
@click.argument('n')
def three_d(n: int):
    """
    Plots a three dimentional (d-1)-sphere
    where d is the number of dimentions

    Parameters
    ----------
    n: int
        n represents the number of random points
    """
    N, D = int(n), 3

    samples = sphere(N, D)

    x = samples[0, ]
    y = samples[1, ]
    z = samples[2, ]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='3d'))

    ax.set(title="3 dimentions projection using Gaussian distribution")
    ax.scatter(x, y, z, c="tab:blue", alpha=0.8)
    plt.show()


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(three_d)
    cli.add_command(two_d)

    cli()
