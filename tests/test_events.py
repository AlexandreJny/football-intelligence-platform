
import pandas as pd

from src.events import build_events


def test_build_events_returns_dataframe(sample_events):

    events = build_events(sample_events)

    assert isinstance(events, pd.DataFrame)


def test_build_events_correct_number_of_rows(sample_events):

    events = build_events(sample_events)

    assert len(events) == len(sample_events)


def test_build_events_has_16_columns(sample_events):

    events = build_events(sample_events)

    assert len(events.columns) == 16


def test_events_have_match_id(sample_events):

    events = build_events(sample_events)

    assert events["match_id"].notna().all()


def test_event_ids_are_unique(sample_events):

    events = build_events(sample_events)

    assert events["id"].is_unique


def test_events_have_required_columns(sample_events):

    events = build_events(sample_events)

    required_columns = [
        "match_id",
        "id",
        "index",
        "period",
        "timestamp",
        "type",
        "team",
        "player",
        "position",
        "location",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "player_position_id",
        "player_position_name"
    ]

    assert list(events.columns) == required_columns


def test_events_have_correct_types(sample_events):

    events = build_events(sample_events)

    assert events["type"].tolist() == [
        "Pass",
        "Shot",
        "Pass"
    ]


def test_events_have_valid_locations(sample_events):

    events = build_events(sample_events)

    assert events.loc[0, "location"] == (50.0, 25.0)
    assert events.loc[1, "location"] == (100.0, 40.0)


def test_missing_locations_are_none(sample_events):

    sample_events.loc[0, "location_x"] = None

    events = build_events(sample_events)

    assert events.loc[0, "location"] is None


def test_build_events_preserves_original(sample_events):

    original = sample_events.copy()

    build_events(sample_events)

    pd.testing.assert_frame_equal(
        sample_events,
        original
    )


def test_build_events_resets_index(sample_events):

    sample_events.index = [10, 20, 30]

    events = build_events(sample_events)

    assert events.index.tolist() == [0, 1, 2]
