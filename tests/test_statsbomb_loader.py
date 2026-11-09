
import pandas as pd
import pytest

from src.statsbomb_loader import (
    load_csv,
    load_season,
    standardize_events,
    validate_events
)


# ==========================================
# LOAD CSV
# ==========================================

def test_load_csv(sample_csv):

    df = load_csv(sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "match_id" in df.columns


def test_load_csv_missing_file(tmp_path):

    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(file_path)


def test_load_csv_invalid_extension(tmp_path):

    file_path = tmp_path / "events.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError):
        load_csv(file_path)


def test_load_csv_empty_file(tmp_path):

    file_path = tmp_path / "empty.csv"
    file_path.write_text("")

    with pytest.raises(ValueError):
        load_csv(file_path)


# ==========================================
# LOAD SEASON
# ==========================================

def test_load_season(tmp_path):

    df1 = pd.DataFrame({
        "match_id": [1001],
        "id": ["event_1"]
    })

    df2 = pd.DataFrame({
        "match_id": [1002],
        "id": ["event_2"]
    })

    df1.to_csv(
        tmp_path / "Ligue 1 2025_2026 AS Monaco p1.csv",
        index=False
    )

    df2.to_csv(
        tmp_path / "Ligue 1 2025_2026 AS Monaco p2.csv",
        index=False
    )

    result = load_season(tmp_path)

    assert len(result) == 2
    assert "source_file" in result.columns
    assert result["match_id"].nunique() == 2


def test_load_season_missing_directory(tmp_path):

    directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_season(directory)


def test_load_season_no_csv(tmp_path):

    with pytest.raises(FileNotFoundError):
        load_season(tmp_path)


# ==========================================
# STANDARDIZE EVENTS
# ==========================================

def test_standardize_events(sample_events):

    result = standardize_events(sample_events)

    assert "event_id" in result.columns
    assert "type" in result.columns
    assert "team" in result.columns
    assert "player" in result.columns


def test_standardize_events_preserves_original(sample_events):

    original = sample_events.copy()

    standardize_events(sample_events)

    pd.testing.assert_frame_equal(
        sample_events,
        original
    )


def test_standardize_events_adds_missing_columns():

    df = pd.DataFrame({
        "id": ["event_1"]
    })

    result = standardize_events(df)

    required_columns = [
        "match_id",
        "event_id",
        "index",
        "period",
        "timestamp",
        "type",
        "team",
        "player",
        "location_x",
        "location_y"
    ]

    for column in required_columns:
        assert column in result.columns


# ==========================================
# VALIDATE EVENTS
# ==========================================

def test_validate_events_success(sample_events):

    events = standardize_events(sample_events)

    assert validate_events(events) is True


def test_validate_events_empty_dataframe():

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_events(df)


def test_validate_events_missing_match_id(sample_events):

    events = standardize_events(sample_events)

    events.loc[0, "match_id"] = None

    with pytest.raises(ValueError):
        validate_events(events)


def test_validate_events_missing_event_id(sample_events):

    events = standardize_events(sample_events)

    events.loc[0, "event_id"] = None

    with pytest.raises(ValueError):
        validate_events(events)


def test_validate_events_missing_column(sample_events):

    events = standardize_events(sample_events)

    events = events.drop(columns=["match_id"])

    with pytest.raises(ValueError):
        validate_events(events)
