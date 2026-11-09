
import pandas as pd

from src.events import build_events

from src.relational import (
    build_relational_tables,
    validate_relational_tables
)


def test_build_relational_tables_returns_dictionary(sample_events):

    events = build_events(sample_events)

    tables = build_relational_tables(events)

    assert isinstance(tables, dict)


def test_all_relational_tables_exist(sample_events):

    events = build_events(sample_events)

    tables = build_relational_tables(events)

    expected_tables = [
        "competitions",
        "seasons",
        "matches",
        "teams",
        "players",
        "events",
        "match_teams"
    ]

    assert set(tables.keys()) == set(expected_tables)


def test_competitions_table(sample_events):

    tables = build_relational_tables(
        build_events(sample_events)
    )

    assert len(tables["competitions"]) == 1
    assert tables["competitions"].iloc[0]["competition_id"] == "L1"


def test_seasons_table(sample_events):

    tables = build_relational_tables(
        build_events(sample_events)
    )

    assert len(tables["seasons"]) == 1
    assert tables["seasons"].iloc[0]["season_id"] == "2025-2026"


def test_matches_have_unique_ids(sample_events):

    tables = build_relational_tables(
        build_events(sample_events)
    )

    assert tables["matches"]["match_id"].is_unique


def test_teams_have_unique_ids(sample_events):

    tables = build_relational_tables(
        build_events(sample_events)
    )

    assert tables["teams"]["team_id"].is_unique


def test_players_have_unique_ids(sample_events):

    tables = build_relational_tables(
        build_events(sample_events)
    )

    assert tables["players"]["player_id"].is_unique


def test_event_ids_are_unique_in_relational_table(sample_events):

    tables = build_relational_tables(
        build_events(sample_events)
    )

    assert tables["events"]["event_id"].is_unique


def test_events_have_match_id(sample_events):

    tables = build_relational_tables(
        build_events(sample_events)
    )

    assert tables["events"]["match_id"].notna().all()


def test_match_teams_are_unique(sample_events):

    tables = build_relational_tables(
        build_events(sample_events)
    )

    assert not tables["match_teams"].duplicated(
        subset=["match_id", "team_id"]
    ).any()


def test_duplicate_events_are_removed(sample_events):

    events = build_events(sample_events)

    events = pd.concat(
        [events, events.iloc[[0]]],
        ignore_index=True
    )

    tables = build_relational_tables(events)

    assert len(tables["events"]) == 3


def test_validate_relational_tables(sample_events):

    events = build_events(sample_events)

    tables = build_relational_tables(events)

    report = validate_relational_tables(tables)

    for table_name, information in report.items():

        assert information["duplicate_primary_keys"] == 0

        assert all(
            value == 0
            for value in information["missing_primary_keys"].values()
        )


def test_relational_foreign_keys(sample_events):

    events = build_events(sample_events)

    tables = build_relational_tables(events)

    report = validate_relational_tables(tables)

    assert report["matches"]["foreign_keys"]["competition_id"]["orphan_values"] == 0

    assert report["matches"]["foreign_keys"]["season_id"]["orphan_values"] == 0

    assert report["events"]["foreign_keys"]["match_id"]["orphan_values"] == 0

    assert report["events"]["foreign_keys"]["team_id"]["orphan_values"] == 0

    assert report["events"]["foreign_keys"]["player_id"]["orphan_values"] == 0
