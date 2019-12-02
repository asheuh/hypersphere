"""
imports
"""
import click
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_coordinates(d: int) -> np.ndarray:
    """
    Parameters
    ----------
    n: int
        n represents the number of random points
        d is the number of dimentions

    Returns
    -------
    out: ndarray, shape
        Random values
    """
    return np.random.uniform(-1, 1, d)


def hypersphere(n: int, d: int):
    """
    Parameters
    ----------
    n, d: int
        n represents the number of random points
        d is the number of dimentions

    Returns
    -------
    """
    # Fixing random state for reproducibility
    np.random.seed(193873)

    accepted = np.zeros((n, d))
    rejected = np.zeros((int(1e6), d))

    #     fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    fig, ax = plt.subplots()

    for i in range(n):

        # sample random coordinates for each d-dimentions
        coords = get_coordinates(d)

        # check if the point is inside the circle
        # r**2 = x1**2 + x2**2 + ... + xd**2
        r = math.sqrt(sum(coords**2))

        if r < 1:
            accepted[i, ] = coords
        else:
            rejected[i, ] = coords

    rejected = rejected[~np.all(rejected == 0, axis=1)]
    m = len(rejected)
    rejected = np.reshape(rejected, (d, m))
    accepted = np.reshape(accepted, (d, n))

    return accepted, rejected


@click.command()
@click.argument('n')
def two_d(n: int):
    """
    Plots a two dimentional (d-1)-sphere
    where d is the number of dimentions

    Parameters
    ----------
    n: int
        n represents the number of random points
    """

    accepted, rejected = hypersphere(int(n), 2)
    ax, ay = accepted
    rx, ry = rejected
    plt.plot(ax, ay, 'o', color="b", alpha=0.8)
    plt.plot(rx, ry, 'o', color="r", alpha=0.8)
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
    accepted, rejected = hypersphere(int(n), 3)
    ax, ay, az = accepted
    rx, ry, rz = rejected
    plt.plot(ax, ay, az, 'o', color="b", alpha=0.8)
    plt.plot(rx, ry, rz, 'o', color="r", alpha=0.8)
    plt.show()


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(three_d)
    cli.add_command(two_d)
    cli()
