from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Capacity:
    capacity: list[float]
    angles: list[float]

    def __post_init__(self):
        logger.info("\tCapacity initialised")
        if len(self.capacity) != len(self.angles):
            raise ValueError("Capacity and angles must have the same length")
        if len(self.capacity) <2:
            raise ValueError("Capacity must have at least 2 elements")
        if len(self.angles) <2:
            raise ValueError("Angles must have at least 2 elements")

    def to_dataframe(self):
        """

        Returns:
            pd.DataFrame

        """
        df = pd.DataFrame(
            {
                "capacity": self.capacity,
                "angles": self.angles,
            }
        )
        return df

    def convert_to_radians(self):
        """

        Returns:
            array with thetas in radians

        """
        logger.info("\tConverting capacity angles to radians")
        self.angles = np.deg2rad(self.angles).tolist()

    def to_triad(self) -> np.ndarray:
        logger.info("\tConverting capacity angles to triad")
        df = self.to_dataframe()
        mom = df["capacity"].to_numpy(dtype=float)
        ang = df["angles"].to_numpy(dtype=float)

        mx = np.sum(mom * np.cos(ang) ** 2)
        my = np.sum(mom * np.sin(ang) ** 2)
        mxy = -1 * np.sum(mom * np.sin(ang) * np.cos(ang))

        return np.array([mx, my, mxy], dtype=float)

@dataclass
class ThetasField:
    start_angle: float = 0
    end_angle: float = np.pi
    number_of_divisions: int = 180

    def __post_init__(self):
        logger.info("\tAngles field initialised")

    @property
    def x(self):
        return np.linspace(self.start_angle, self.end_angle, self.number_of_divisions)

    @classmethod
    def default_field(cls):
        return cls(start_angle=0, end_angle=np.pi, number_of_divisions=180)

def create_moment_field(triad: np.ndarray | list[float],
                        angles_field: np.ndarray = None) -> np.ndarray:
    logger.info("\tCreating moment field")
    Mx, My, Mxy = triad

    if angles_field is None:
        logging.info("\tAngle field not initialised, creating default angles field")
        angles_field = ThetasField().x

    s = np.sin(angles_field)
    c = np.cos(angles_field)
    return Mx * c ** 2 + My * s ** 2 - 2 * Mxy * s * c

def list_of_loads(df: pd.DataFrame) -> list[Any]:
    return df["load"].unique().tolist()

def results_mean_by_node(df: pd.DataFrame) -> pd.DataFrame:
    mean_forces = df.groupby(['load', 'node'])[['mxx', 'myy', 'mxy']].mean().reset_index()
    results_mean = df[['elem', 'load', 'node']].merge(mean_forces, 'left', on=['load', 'node'])
    return results_mean

class Denton:
    def __init__(self,
                 moments_triad: np.ndarray | list[float],
                 capacity: Capacity,
                 thetas_field: ThetasField
    ):
        logger.info("\tDenton initialised")
        self.moments_triad = moments_triad
        self.capacity = capacity
        self.thetas_field = thetas_field or ThetasField.default_field()
        self.angles_field = thetas_field.x
        self.capacities_triad = self.capacity.to_triad()
        self.moment_field = create_moment_field(self.moments_triad, self.angles_field)
        self.capacities_field = create_moment_field(self.capacities_triad, self.angles_field)
        self.gamma = None
        self.theta = None
        self.calculate_gamma_theta()



    def calculate_gamma_theta(self,
                              eps=1e-12,
                              tol=1e-12) -> tuple[float, float]:

        MR = self.capacities_field
        MN = self.moment_field
        X = self.angles_field

        # jeżeli funkcja nośności MR < 0
        if np.any((MR < 0) & (MN > eps)):
            return 1000., 0.
        mask = (MN > 0) & (MR >= 0)
        if not np.any(mask):
            return 1000., 0.
        ratio = MR[mask] / MN[mask]
        min_arg = np.argmin(ratio)
        gamma = ratio[min_arg]
        theta = X[mask][min_arg]

        # walidacja
        if np.any(gamma * MN > MR + tol):
            den = np.clip(MN, eps, None)
            gamma = np.min((MR + tol) / den)

        self.gamma = gamma
        self.theta = theta
        return float(gamma), float(theta)

def merge_df(results_mean_by_node: pd.DataFrame) -> pd.DataFrame:
    nodes = results_mean_by_node[['node', 'x', 'y']]
    results_mean_by_node['gamma'] = (results_mean_by_node.apply(lambda r: _calc_gamma(r, MR=MR), axis=1))
    results_mean_by_node = results_mean_by_node.merge(nodes[['node', 'x', 'y']], 'left', on='node')
    return results_mean_by_node

def orchestrator(capacity: list[float],
                 angles: list[float],
                 moment_triad: list[float],
                 *,
                 thetas_field: ThetasField = None,
                 angles_in_degrees: bool= True) -> Denton:

    cap = Capacity(capacity, angles)

    if angles_in_degrees:
        cap.convert_to_radians()

    if thetas_field is None:
        thetas_field = ThetasField.default_field()

    dent=  Denton(moments_triad=moment_triad,
                          capacity=cap,
                          thetas_field=thetas_field)
    return dent


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:\t%(message)s")

    c = [100, 35]
    a = [0, 70]
    triad = [35, 15, 10]

    a = orchestrator(capacity=c, angles=a, moment_triad=triad)

if __name__ == "__main__":
    main()