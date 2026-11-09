
import duckdb
import pandas as pd
import pytest

from src.database import (
    create_database,
    load_tables,
    check_primary_keys,
    validate_foreign_keys,
    display_database_summary
)


TABLE_NAMES = [
    "competitions",
    "seasons",
    "matches",
    "teams",
    "players",
    "events",
    "match_teams"
]


@pytest.fixture
def database_tables(tmp_path):
    """Prépare les sept fichiers CSV nécessaires."""

    data_directory = tmp_path / "processed"
    data_directory.mkdir()

    tables = {
        "competitions": pd.DataFrame({
            "competition_id": ["L1"],
            "name": ["Ligue 1"]
        }),

        "seasons": pd.DataFrame({
            "season_id": ["2025-2026"],
            "season_name": ["2025/2026"]
        }),

        "matches": pd.DataFrame({
            "match_id": [1001],
            "competition_id": ["L1"],
            "season_id": ["2025-2026"]
        }),

        "teams": pd.DataFrame({
            "team_id": [10, 20],
            "name": ["AS Monaco", "PSG"]
        }),

        "players": pd.DataFrame({
            "player_id": [101, 201],
            "name": ["Joueur A", "Joueur B"]
        }),

        "events": pd.DataFrame({
            "event_id": ["event_1", "event_2"],
            "match_id": [1001, 1001],
            "team_id": [10, 20],
            "player_id": [101, 201]
        }),

        "match_teams": pd.DataFrame({
            "match_id": [1001, 1001],
            "team_id": [10, 20]
        })
    }

    for name, df in tables.items():

        df.to_csv(
            data_directory / f"{name}_2025_2026.csv",
            index=False
        )

    return data_directory


@pytest.fixture
def connection():

    conn = duckdb.connect(":memory:")

    yield conn

    conn.close()


# ==========================================
# CREATE DATABASE
# ==========================================

def test_create_database(tmp_path):

    database_path = tmp_path / "database" / "test.duckdb"

    conn = create_database(database_path)

    assert database_path.exists()

    conn.close()


def test_create_database_creates_parent_directory(tmp_path):

    database_path = (
        tmp_path / "new_folder" / "football.duckdb"
    )

    conn = create_database(database_path)

    assert database_path.parent.exists()

    conn.close()


# ==========================================
# LOAD TABLES
# ==========================================

def test_load_tables(connection, database_tables):

    load_tables(connection, database_tables)

    tables = connection.execute(
        "SHOW TABLES"
    ).fetchdf()

    loaded_tables = set(tables["name"])

    assert loaded_tables == set(TABLE_NAMES)


def test_load_tables_correct_row_count(
    connection,
    database_tables
):

    load_tables(connection, database_tables)

    result = connection.execute(
        "SELECT COUNT(*) FROM events"
    ).fetchone()[0]

    assert result == 2


def test_load_tables_missing_file(connection, tmp_path):

    with pytest.raises(FileNotFoundError):

        load_tables(connection, tmp_path)


# ==========================================
# PRIMARY KEYS
# ==========================================

def test_check_primary_keys_success(
    connection,
    database_tables
):

    load_tables(connection, database_tables)

    check_primary_keys(connection)


def test_check_primary_keys_detects_duplicates(connection):

    connection.execute("""
        CREATE TABLE competitions AS
        SELECT * FROM (
            VALUES
                ('L1', 'Ligue 1'),
                ('L1', 'Ligue 1')
        ) AS t(competition_id, name)
    """)

    with pytest.raises(ValueError):

        check_primary_keys(connection)


def test_check_primary_keys_detects_null(connection):

    connection.execute("""
        CREATE TABLE competitions AS
        SELECT * FROM (
            VALUES
                (NULL, 'Ligue 1')
        ) AS t(competition_id, name)
    """)

    with pytest.raises(ValueError):

        check_primary_keys(connection)


# ==========================================
# FOREIGN KEYS
# ==========================================

def test_validate_foreign_keys_success(
    connection,
    database_tables
):

    load_tables(connection, database_tables)

    validate_foreign_keys(connection)


def test_validate_foreign_keys_detects_orphans(connection):

    connection.execute("""
        CREATE TABLE competitions AS
        SELECT 'L1' AS competition_id
    """)

    connection.execute("""
        CREATE TABLE seasons AS
        SELECT '2025-2026' AS season_id
    """)

    connection.execute("""
        CREATE TABLE matches AS
        SELECT
            1001 AS match_id,
            'INVALID' AS competition_id,
            '2025-2026' AS season_id
    """)

    with pytest.raises(ValueError):

        validate_foreign_keys(connection)


# ==========================================
# DATABASE SUMMARY
# ==========================================

def test_display_database_summary(
    connection,
    database_tables,
    capsys
):

    load_tables(connection, database_tables)

    display_database_summary(connection)

    captured = capsys.readouterr()

    assert "events" in captured.out
    assert "competitions" in captured.out
    assert "matches" in captured.out
