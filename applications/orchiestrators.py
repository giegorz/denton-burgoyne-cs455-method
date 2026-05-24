import logging

import numpy as np

import pandas as pd

from scripts.denton_logic import Capacity, ThetasField, Denton, create_moment_field


def denton_orchestrator(capacity: Capacity,
                        moment_triad: list[float] | np.ndarray,
                        *,
                        thetas_field: ThetasField = None,
                        convert_angles_to_degrees: bool= True) -> Denton:
    """

    Args:
        capacity: list of capacity values
        angles: list of angles values
        moment_triad: moment triad values
        thetas_field: discretized field of angles
        convert_angles_to_degrees: convert angles to degrees if angles defined in radians

    Returns:
        Denton object
    """

    if convert_angles_to_degrees:
        capacity.convert_to_radians()

    if thetas_field is None:
        thetas_field = ThetasField.default_thetas_field()

    dent=  Denton(
        moments_triad=moment_triad,
        capacity=capacity,
        thetas_field=thetas_field
    )
    return dent


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:\t%(message)s")

    c = [100, 35]
    a = [0, 70]
    triad = [35, 15, 10]
    capacity = Capacity(capacity=c, angles=a)


if __name__ == "__main__":
    main()