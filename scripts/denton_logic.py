from dataclasses import dataclass
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
        df = pd.DataFrame(
            {
                "capacity": self.capacity,
                "angles": self.angles,
            }
        )
        return df

    def convert_to_radians(self):
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
class AnglesField:
    start_angle: float = 0
    end_angle: float = np.pi
    number_of_divisions: int = 180

    def __post_init__(self):
        logger.info("\tAngles field initialised")

    @property
    def x(self):
        return np.linspace(self.start_angle, self.end_angle, self.number_of_divisions)


def create_moment_field(triad: np.ndarray | list[float],
                        angles_field: np.ndarray = None) -> np.ndarray:
    logger.info("\tCreating moment field")
    Mx, My, Mxy = triad

    if angles_field is None:
        logging.info("\tAngle field not initialised, creating default angles field")
        angles_field = AnglesField().x

    s = np.sin(angles_field)
    c = np.cos(angles_field)
    return Mx * c ** 2 + My * s ** 2 - 2 * Mxy * s * c



def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:\t%(message)s")

    # c = [100, 35]
    # a = [0, 70]
    # cap = Capacity(c,a)
    # cap.convert_to_radians()
    # capacity_triad = cap.to_triad()
    # moment_field = create_moment_field(capacity_triad)

    triad = np.array([35, 15, 10])
    moment_field = create_moment_field(triad)


if __name__ == "__main__":
    main()