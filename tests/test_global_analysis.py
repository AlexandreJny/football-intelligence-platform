
import pandas as pd
import pytest

from src.global_analysis import (
    get_shots,
    calculate_shot_metrics,
    analyze_shot_zones,
)


@pytest.fixture
def sample_events():

    return pd.DataFrame({
        "event_id": ["e1", "e2", "e3", "e4"],
        "match_id": [1, 1, 1, 2],
        "team_id": [129, 129, 129, 129],
        "player_id": [10, 11, 12, 10],
        "player_name": ["Player A", "Player B", "Player C", "Player A"],
        "type": ["Shot", "Shot", "Pass", "Shot"],
        "statsbomb_xg": [0.5, 0.2, None, 0.3],
        "outcome_name": ["Goal", "Saved", None, "Off T"],
        "location_x": [110, 90, 80, 108],
        "location_y": [40, 30, 40, 50],
        "assisted_shot_id": [None, None, None, None]
    })


def test_get_shots(sample_events):

    shots = get_shots(sample_events)

    assert len(shots) == 3

    assert all(
        shots["type"].str.lower() == "shot"
    )


def test_shot_count(sample_events):

    shots = get_shots(sample_events)

    metrics = calculate_shot_metrics(shots)

    assert metrics["total_shots"] == 3


def test_total_xg(sample_events):

    shots = get_shots(sample_events)

    metrics = calculate_shot_metrics(shots)

    assert metrics["total_xg"] == pytest.approx(1.0)


def test_average_xg(sample_events):

    shots = get_shots(sample_events)

    metrics = calculate_shot_metrics(shots)

    assert metrics["average_xg"] == pytest.approx(1.0 / 3)


def test_xg_per_shot(sample_events):

    shots = get_shots(sample_events)

    metrics = calculate_shot_metrics(shots)

    assert metrics["xg_per_shot"] == pytest.approx(1.0 / 3)


def test_conversion_rate(sample_events):

    shots = get_shots(sample_events)

    metrics = calculate_shot_metrics(shots)

    assert metrics["goals"] == 1

    assert metrics["conversion_rate"] == pytest.approx(
        100 / 3
    )


def test_shot_zones(sample_events):

    shots = get_shots(sample_events)

    zones = analyze_shot_zones(shots)

    assert "shot_zone" in zones.columns

    assert zones["total_shots"].sum() == 3


def test_empty_shots():

    columns = [
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "type",
        "statsbomb_xg",
        "outcome_name",
        "location_x",
        "location_y"
    ]

    empty_df = pd.DataFrame(columns=columns)

    metrics = calculate_shot_metrics(empty_df)

    assert metrics["total_shots"] == 0

    assert metrics["conversion_rate"] == 0
