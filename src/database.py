
from pathlib import Path
import duckdb


def create_database(database_path):
    """
    Create or connect to a DuckDB database.
    """

    database_path = Path(database_path)

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = duckdb.connect(str(database_path))

    return connection


def load_tables(connection, data_directory):
    """
    Import processed CSV files into DuckDB.
    """

    data_directory = Path(data_directory)

    table_names = [
        "competitions",
        "seasons",
        "matches",
        "teams",
        "players",
        "events",
        "match_teams"
    ]

    for table_name in table_names:

        csv_path = (
            data_directory /
            f"{table_name}_2025_2026.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {csv_path}"
            )

        # Sécuriser le chemin pour SQL
        csv_path_sql = csv_path.resolve().as_posix().replace(
            "'", "''"
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv(
                '{csv_path_sql}',
                header = true,
                auto_detect = true
            )
            """
        )

        print(f"[OK] Table {table_name} importée.")


def check_primary_keys(connection):

    primary_keys = {
        "competitions": ["competition_id"],
        "seasons": ["season_id"],
        "matches": ["match_id"],
        "teams": ["team_id"],
        "players": ["player_id"],
        "events": ["event_id"],
        "match_teams": ["match_id", "team_id"]
    }

    print("\n===== CONTRÔLE DES CLÉS PRIMAIRES =====")

    for table, columns in primary_keys.items():

        group_columns = ", ".join(columns)

        query = f"""
            SELECT
                {group_columns},
                COUNT(*) AS occurrences
            FROM {table}
            GROUP BY {group_columns}
            HAVING COUNT(*) > 1
        """

        duplicates = connection.execute(query).fetchdf()

        if not duplicates.empty:
            raise ValueError(
                f"Doublons détectés dans {table} :\n"
                f"{duplicates.head(10)}"
            )

        null_conditions = " OR ".join(
            f"{column} IS NULL"
            for column in columns
        )

        null_query = f"""
            SELECT COUNT(*)
            FROM {table}
            WHERE {null_conditions}
        """

        null_count = connection.execute(
            null_query
        ).fetchone()[0]

        if null_count > 0:
            raise ValueError(
                f"{table} contient {null_count} "
                "clé(s) primaire(s) manquante(s)."
            )

        print(f"[OK] {table} : clé primaire valide.")



def validate_foreign_keys(connection):
    """
    Validate foreign key relationships without
    relying on ALTER TABLE constraints.
    """

    checks = {
        "matches_competitions": """
            SELECT COUNT(*)
            FROM matches m
            LEFT JOIN competitions c
                ON m.competition_id = c.competition_id
            WHERE c.competition_id IS NULL
        """,

        "matches_seasons": """
            SELECT COUNT(*)
            FROM matches m
            LEFT JOIN seasons s
                ON m.season_id = s.season_id
            WHERE s.season_id IS NULL
        """,

        "events_matches": """
            SELECT COUNT(*)
            FROM events e
            LEFT JOIN matches m
                ON e.match_id = m.match_id
            WHERE e.match_id IS NOT NULL
              AND m.match_id IS NULL
        """,

        "events_teams": """
            SELECT COUNT(*)
            FROM events e
            LEFT JOIN teams t
                ON e.team_id = t.team_id
            WHERE e.team_id IS NOT NULL
              AND t.team_id IS NULL
        """,

        "events_players": """
            SELECT COUNT(*)
            FROM events e
            LEFT JOIN players p
                ON e.player_id = p.player_id
            WHERE e.player_id IS NOT NULL
              AND p.player_id IS NULL
        """,

        "match_teams_matches": """
            SELECT COUNT(*)
            FROM match_teams mt
            LEFT JOIN matches m
                ON mt.match_id = m.match_id
            WHERE m.match_id IS NULL
        """,

        "match_teams_teams": """
            SELECT COUNT(*)
            FROM match_teams mt
            LEFT JOIN teams t
                ON mt.team_id = t.team_id
            WHERE t.team_id IS NULL
        """
    }

    print("\n===== VALIDATION DES RELATIONS =====")

    for relation, query in checks.items():

        count = connection.execute(query).fetchone()[0]

        if count == 0:
            print(f"[OK] {relation}")
        else:
            print(f"[ERREUR] {relation} : {count} relation(s) invalide(s)")

            raise ValueError(
                f"Relations invalides détectées : {relation}"
            )


def display_database_summary(connection):

    print("\n===== RÉSUMÉ DE LA BASE DUCKDB =====")

    tables = connection.execute(
        "SHOW TABLES"
    ).fetchdf()

    for table_name in tables["name"]:

        count = connection.execute(
            f'SELECT COUNT(*) FROM "{table_name}"'
        ).fetchone()[0]

        print(f"{table_name} : {count} lignes")
