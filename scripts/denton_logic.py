from dataclasses import dataclass

import pandas as pd
import numpy as np
import logging

from importers import import_results, import_elements, import_nodes

logger = logging.getLogger(__name__)


EPS = 1e-12
TOL = 1e-12

#---------------------------------
#   MAIN CLASSES
#---------------------------------

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
        self.convert_to_radians()

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

class Denton:
    def __init__(
        self,
        moments_triad: np.ndarray | list[float],
        capacity: Capacity,
        thetas_field: ThetasField | None = None,
    ):
        logger.info("Denton initialised")

        # --- validation ---
        self.moments_triad = np.asarray(moments_triad, dtype=float)
        if self.moments_triad.shape != (3,):
            raise ValueError("moments_triad must have shape (3,)")

        self.capacity = capacity
        self.thetas_field = thetas_field or ThetasField.default_thetas_field()

        # --- precompute ---
        self.capacities_triad = self.capacity.to_triad()

        self._moment_field = None
        self._capacities_field = None

        # --- outputs ---
        self.gamma: float | None = None
        self.theta: float | None = None

        # --- compute ---
        self._compute_fields()
        self.calculate_gamma_theta()

    # -------------------------
    # Properties
    # -------------------------
    @property
    def angles_field(self) -> np.ndarray:
        return self.thetas_field.x

    @property
    def moment_field(self) -> np.ndarray:
        return self._moment_field

    @property
    def capacities_field(self) -> np.ndarray:
        return self._capacities_field

    # -------------------------
    # Internal methods
    # -------------------------
    def _compute_fields(self) -> None:
        """Compute moment and capacity fields."""
        angles = self.angles_field

        logger.debug("Computing moment and capacity fields")

        self._moment_field = create_moment_field(
            self.moments_triad,
            angles,
        )

        self._capacities_field = create_moment_field(
            self.capacities_triad,
            angles,
        )

        if not np.all(np.isfinite(self._moment_field)):
            raise ValueError("moment_field contains invalid values")

        if not np.all(np.isfinite(self._capacities_field)):
            raise ValueError("capacities_field contains invalid values")

    # -------------------------
    # Main computation
    # -------------------------
    def calculate_gamma_theta(
        self,
        eps: float = EPS,
        tol: float = TOL,
    ) -> tuple[float, float]:
        """
        Compute gamma and theta based on Denton-Burgoyne criterion.
        """

        MR = self.capacities_field
        MN = self.moment_field
        X = self.angles_field

        # Case: invalid resistance
        if np.any((MR < 0) & (MN > eps)):
            logger.warning("Negative resistance detected")
            self.gamma, self.theta = np.nan, np.nan
            return self.gamma, self.theta

        mask = (MN > eps) & (MR >= 0)

        if not np.any(mask):
            logger.warning("No valid gamma found")
            self.gamma, self.theta = np.nan, np.nan
            return self.gamma, self.theta

        ratio = MR[mask] / MN[mask]

        idx = np.argmin(ratio)
        gamma = ratio[idx]
        theta = X[mask][idx]

        # --- validation correction ---
        if np.any(gamma * MN > MR + tol):
            logger.debug("Gamma correction applied")
            den = np.clip(MN, eps, None)
            gamma = np.min((MR + tol) / den)

        self.gamma = float(gamma)
        self.theta = float(theta)

        return self.gamma, self.theta


#---------------------------------
#   FUNCTIONS TO CALCULATE GAMMA
#---------------------------------

def create_moment_field(
        triad: np.ndarray | list[float],
        angles_field: np.ndarray = None
) -> np.ndarray:
    logger.debug(f"\tCreating moment field from triad: {triad}")

    if triad.shape !=(3,):
        raise ValueError("Triad must be shape (3,)")

    Mx, My, Mxy = triad

    if angles_field is None:
        logger.info("\tAngle field not initialised, creating default angles field")
        angles_field = ThetasField.default_thetas_field().x

    s = np.sin(angles_field)[None,:]
    c = np.cos(angles_field)[None,:]
    return Mx * c ** 2 + My * s ** 2 - 2 * Mxy * s * c

def extract_data_from_results(results: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    elements = results["elem"].to_numpy()
    moments = results[["mxx", "myy", "mxy"]].to_numpy(dtype=float)
    nodes = results["node"].to_numpy()
    loads = results["load"].to_numpy()
    return moments, nodes, loads, elements


def create_capacity_field(
    capacity: Capacity,
    thetas: np.ndarray,
) -> np.ndarray:
    cap_triad = capacity.to_triad()
    return create_moment_field(cap_triad, thetas)


def create_moments_field(
    moments: np.ndarray,
    thetas: np.ndarray,
) -> np.ndarray:
    mx = moments[:, 0][:, None]
    my = moments[:, 1][:, None]
    mxy = moments[:, 2][:, None]

    s = np.sin(thetas)
    c = np.cos(thetas)

    return mx * c**2 + my * s**2 - 2 * mxy * s * c


def calculate_denton_burgoyne_gamma(
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

def denton_burgoyne_orchestrator(
    results: pd.DataFrame,
    capacity: Capacity,
    *,
    thetas_field: ThetasField | None = None,
) -> pd.DataFrame:
    if thetas_field is None:
        thetas_field = ThetasField.default_thetas_field()

    thetas = thetas_field.x
    moments, nodes, loads, elements = extract_data_from_results(results)

    resistance_field = create_capacity_field(capacity, thetas)
    moment_field = create_moments_field(moments, thetas)
    gamma, theta = calculate_denton_burgoyne_gamma(resistance_field, moment_field, thetas)

    return pd.DataFrame({
        "node": nodes,
        "elem": elements,
        "loads": loads,
        "gamma": gamma,
        "theta": theta,
    })

def group_gammas_by_elements(
    denton_burgoyne_results: pd.DataFrame,
    how: str = "min"
) -> pd.DataFrame:
    _map = {
        "min": denton_burgoyne_results.groupby(["elem", "loads"])["gamma"].min().reset_index(),
        "mean": denton_burgoyne_results.groupby(["elem", "loads"])["gamma"].mean().reset_index()
    }

    if how not in _map.keys():
        raise ValueError(f"[[{how}]] is not a valid option, please choose 'min' or 'mean' ")

    return _map[how]