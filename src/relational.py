import pandas as pd

def build_relational_tables(events_df):
    """
    Build normalized relational tables from StatsBomb events.

    Parameters
    ----------
    events_df : pandas.DataFrame
        Standardized Events table.

    Returns
    -------
    dict
        Dictionary containing relational DataFrames.
    """

    df = events_df.copy()

    # -------------------------------------------------
    # 1. COMPETITIONS
    # -------------------------------------------------

    competitions_df = pd.DataFrame({
        "competition_id": ["L1"],
        "name": ["Ligue 1"]
    })

    # -------------------------------------------------
    # 2. SEASONS
    # -------------------------------------------------

    seasons_df = pd.DataFrame({
        "season_id": ["2025-2026"],
        "season_name": ["2025/2026"]
    })

    # -------------------------------------------------
    # 3. MATCHES
    # -------------------------------------------------

    matches_df = (
        df[["match_id"]]
        .dropna(subset=["match_id"])
        .drop_duplicates()
        .copy()
    )

    matches_df["competition_id"] = "L1"
    matches_df["season_id"] = "2025-2026"

    # La date n'est pas disponible dans les exports actuels
    matches_df["match_date"] = pd.NaT

    # -------------------------------------------------
    # 4. TEAMS
    # -------------------------------------------------

    teams_df = (
        df[["team_id", "team_name"]]
        .dropna(subset=["team_id"])
        .drop_duplicates()
        .copy()
    )

    # Un identifiant StatsBomb doit correspondre
    # à un seul nom dans notre référentiel
    teams_df = (
        teams_df.sort_values("team_name", na_position="last")
        .drop_duplicates(subset=["team_id"], keep="first")
    )

    teams_df = teams_df.rename(
        columns={"team_name": "name"}
    )

    # -------------------------------------------------
    # 5. PLAYERS
    # -------------------------------------------------

    players_df = (
        df[["player_id", "player_name"]]
        .dropna(subset=["player_id"])
        .drop_duplicates()
        .copy()
    )

    players_df = (
        players_df.sort_values("player_name", na_position="last")
        .drop_duplicates(subset=["player_id"], keep="first")
    )

    players_df = players_df.rename(
        columns={"player_name": "name"}
    )

    # -------------------------------------------------
    # 6. EVENTS
    # -------------------------------------------------


    # -------------------------------------------------
    # 6. EVENTS
    # -------------------------------------------------

    event_columns = [
        "id",
        "match_id",
        "team_id",
        "player_id",
        "type",
        "timestamp",
        "period",
        "index",
        "location"
    ]

    for column in event_columns:
        if column not in df.columns:
            df[column] = pd.NA

    events_df = df[event_columns].copy()

    events_df = events_df.rename(
        columns={"id": "event_id"}
    )

    # Supprimer les doublons d'événements
    initial_count = len(events_df)

    events_df = events_df.drop_duplicates(
        subset=["event_id"],
        keep="first"
    ).reset_index(drop=True)

    removed_count = initial_count - len(events_df)

    print("\n===== NETTOYAGE DES ÉVÉNEMENTS =====")
    print("Événements avant nettoyage :", initial_count)
    print("Doublons supprimés :", removed_count)
    print("Événements après nettoyage :", len(events_df))

    # -------------------------------------------------
    # 7. MATCH_TEAMS
    # -------------------------------------------------

    match_teams_df = (
        df[["match_id", "team_id"]]
        .dropna(subset=["match_id", "team_id"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # 8. REGROUPER LES TABLES
    # -------------------------------------------------

    tables = {
        "competitions": competitions_df.reset_index(drop=True),
        "seasons": seasons_df.reset_index(drop=True),
        "matches": matches_df.reset_index(drop=True),
        "teams": teams_df.reset_index(drop=True),
        "players": players_df.reset_index(drop=True),
        "events": events_df.reset_index(drop=True),
        "match_teams": match_teams_df
    }

    return tables



def validate_relational_tables(tables):
    """
    Validate primary keys, missing values and foreign keys.

    Parameters
    ----------
    tables : dict
        Relational DataFrames.

    Returns
    -------
    dict
        Validation report.
    """

    primary_keys = {
        "competitions": ["competition_id"],
        "seasons": ["season_id"],
        "matches": ["match_id"],
        "teams": ["team_id"],
        "players": ["player_id"],
        "events": ["event_id"],
        "match_teams": ["match_id", "team_id"]
    }

    foreign_keys = {
        "matches": {
            "competition_id": ("competitions", "competition_id"),
            "season_id": ("seasons", "season_id")
        },
        "events": {
            "match_id": ("matches", "match_id"),
            "team_id": ("teams", "team_id"),
            "player_id": ("players", "player_id")
        },
        "match_teams": {
            "match_id": ("matches", "match_id"),
            "team_id": ("teams", "team_id")
        }
    }

    report = {}

    for table_name, df in tables.items():

        pk_columns = primary_keys[table_name]

        # Doublons de clés primaires
        duplicate_count = df.duplicated(
            subset=pk_columns
        ).sum()

        # Valeurs manquantes dans les clés primaires
        missing_pk = df[pk_columns].isna().sum().to_dict()

        report[table_name] = {
            "rows": len(df),
            "duplicate_primary_keys": int(duplicate_count),
            "missing_primary_keys": missing_pk,
            "foreign_keys": {}
        }

        # Vérification des clés étrangères
        for column, (parent_table, parent_column) in (
            foreign_keys.get(table_name, {}).items()
        ):

            parent_ids = set(
                tables[parent_table][parent_column].dropna()
            )

            missing_relation = (
                df[column].notna()
                & ~df[column].isin(parent_ids)
            )

            report[table_name]["foreign_keys"][column] = {
                "missing_values": int(df[column].isna().sum()),
                "orphan_values": int(missing_relation.sum())
            }

    return report


def display_validation_report(report):

    print("\n===== VALIDATION DU MODÈLE RELATIONNEL =====")

    for table_name, information in report.items():

        print(f"\n--- {table_name.upper()} ---")

        print("Nombre de lignes :", information["rows"])

        print(
            "Doublons PK :",
            information["duplicate_primary_keys"]
        )

        print(
            "Valeurs PK manquantes :",
            information["missing_primary_keys"]
        )

        for column, details in information["foreign_keys"].items():

            print(f"FK {column} :")

            print(
                "  Valeurs manquantes :",
                details["missing_values"]
            )

            print(
                "  Relations orphelines :",
                details["orphan_values"]
            )
