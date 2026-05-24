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
    def x(self) -> np.ndarray:
        return np.linspace(self.start_angle, self.end_angle, self.number_of_divisions)

    @classmethod
    def default_thetas_field(cls):
        return cls(start_angle=0, end_angle=np.pi, number_of_divisions=180)

def create_moment_field(
        triad: np.ndarray | list[float],
        angles_field: np.ndarray = None
) -> np.ndarray:
    logger.info(f"\tCreating moment field from triad: {triad}")
    Mx, My, Mxy = triad

    if angles_field is None:
        logger.info("\tAngle field not initialised, creating default angles field")
        angles_field = ThetasField.default_thetas_field().x

    s = np.sin(angles_field)
    c = np.cos(angles_field)
    return Mx * c ** 2 + My * s ** 2 - 2 * Mxy * s * c


class Denton:
    def __init__(
        self,
        moments_triad: np.ndarray | list[float],
        capacity: Capacity,
        thetas_field: ThetasField
    ):
        logger.info("\tDenton initialised")
        self.moments_triad = np.asarray(moments_triad)
        self.capacity = capacity
        self.thetas_field = thetas_field or ThetasField.default_thetas_field()
        self.angles_field = thetas_field.x
        self.capacities_triad = self.capacity.to_triad()
        self.moment_field = create_moment_field(
            self.moments_triad,
            self.angles_field)
        self.capacities_field = create_moment_field(
            self.capacities_triad,
            self.angles_field)
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

def _extract_moment_components(results: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moments = results[["mxx", "myy", "mxy"]].to_numpy(dtype=float)
    nodes = results["node"].to_numpy()
    loads = results["load"].to_numpy()
    return moments, nodes, loads


def _capacity_resistance_field(
    capacity: Capacity,
    thetas: np.ndarray,
) -> np.ndarray:
    cap_triad = capacity.to_triad()
    return create_moment_field(cap_triad, thetas)


def _moment_field(
    moments: np.ndarray,
    thetas: np.ndarray,
) -> np.ndarray:
    mx = moments[:, 0][:, None]
    my = moments[:, 1][:, None]
    mxy = moments[:, 2][:, None]

    s = np.sin(thetas)
    c = np.cos(thetas)

    return mx * c**2 + my * s**2 - 2 * mxy * s * c


def _gamma_theta_from_fields(
    resistance_field: np.ndarray,
    moment_field: np.ndarray,
    thetas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = moment_field > 0
    ratio = np.where(mask, resistance_field / moment_field, np.inf)
    idx = np.argmin(ratio, axis=1)

    gamma = ratio[np.arange(moment_field.shape[0]), idx]
    theta = thetas[idx]
    return gamma, theta

def calculate_gammas(
    results: pd.DataFrame,
    capacity: Capacity,
    *,
    thetas_field: ThetasField | None = None,
) -> pd.DataFrame:
    if thetas_field is None:
        thetas_field = ThetasField.default_thetas_field()

    thetas = thetas_field.x
    moments, nodes, loads = _extract_moment_components(results)

    resistance_field = _capacity_resistance_field(capacity, thetas)
    moment_field = _moment_field(moments, thetas)
    gamma, theta = _gamma_theta_from_fields(resistance_field, moment_field, thetas)

    return pd.DataFrame({
        "node": nodes,
        "loads": loads,
        "gamma": gamma,
        "theta": theta,
    })