import numpy as np
import pandas as pd
import pytest

from scripts.denton_logic import (
    Capacity,
    Denton,
    ThetasField,
    calculate_denton_burgoyne_gamma,
    denton_burgoyne_by_node,
)


def test_calculate_denton_burgoyne_gamma_normal_case():
    thetas = np.array([0.0, np.pi / 2])
    MR = np.array([[2.0, 2.0]])
    MN = np.array([[1.0, 4.0]])

    gamma, theta = calculate_denton_burgoyne_gamma(MR, MN, thetas)

    assert gamma[0] == pytest.approx(0.5)
    assert theta[0] == pytest.approx(np.pi / 2)


def test_calculate_denton_burgoyne_gamma_no_demand_is_infinite():
    """No theta has positive moment -> the field never governs -> +inf."""
    thetas = np.array([0.0, np.pi / 2])
    MR = np.array([[1.0, 1.0]])
    MN = np.array([[-1.0, -1.0]])

    gamma, _ = calculate_denton_burgoyne_gamma(MR, MN, thetas)

    assert np.isinf(gamma[0])


def test_calculate_denton_burgoyne_gamma_negative_resistance_is_zero():
    """Positive demand where the resistance is negative -> no capacity at
    all in that direction -> treated as the most critical case (gamma=0),
    not folded into an unrelated ratio from a different theta."""
    thetas = np.array([0.0, np.pi / 2])
    MR = np.array([[-1.0, 5.0]])
    MN = np.array([[1.0, 1.0]])

    gamma, _ = calculate_denton_burgoyne_gamma(MR, MN, thetas)

    assert gamma[0] == 0.0


def test_calculate_denton_burgoyne_gamma_is_vectorized_across_rows():
    thetas = np.array([0.0, np.pi / 2])
    MR = np.array([[2.0, 2.0]])  # shared resistance field, shape (1, n_thetas)
    MN = np.array([
        [1.0, 4.0],   # normal case -> 0.5
        [-1.0, -1.0],  # no demand -> inf
    ])

    gamma, _ = calculate_denton_burgoyne_gamma(MR, MN, thetas)

    assert gamma[0] == pytest.approx(0.5)
    assert np.isinf(gamma[1])


def test_denton_class_matches_vectorized_helper():
    """The OOP Denton class and the vectorized function must agree, since
    the class now delegates to it instead of keeping its own logic."""
    capacity = Capacity([750, 500], [0, 70])
    denton = Denton([50, 20, 15], capacity)

    thetas = ThetasField.default_thetas_field().x
    gamma, theta = calculate_denton_burgoyne_gamma(
        denton.capacities_field, denton.moment_field, thetas
    )

    assert denton.gamma == pytest.approx(float(gamma[0]))
    assert denton.theta == pytest.approx(float(theta[0]))


def test_denton_burgoyne_by_node_one_gamma_per_node_load():
    node_moments = pd.DataFrame({
        "node": [1, 2],
        "load": ["L1", "L1"],
        "mxx": [50.0, 10.0],
        "myy": [20.0, 5.0],
        "mxy": [15.0, 2.0],
    })
    capacity = Capacity([750, 500], [0, 70])

    out = denton_burgoyne_by_node(node_moments, capacity)

    assert list(out.columns) == ["node", "load", "gamma", "theta"]
    assert len(out) == 2
    assert (out["gamma"] > 0).all()
