import pandas as pd
from pandas.api.types import is_object_dtype


def normalize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    boolean_columns = {
        "pass_cross",
        "pass_cut_back",
        "pass_switch",
        "one_on_one",
        "open_goal",
        "shot_shot_assist",
        "shot_goal_assist",
        "is_transition",
        "is_controlled_possession",
    }

    for col in df.columns:

        if not is_object_dtype(df[col]):
            continue

        # Nettoyage des chaînes
        df[col] = df[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        values = df[col].dropna()

        if values.empty:
            continue

        # 1. Colonnes booléennes connues
        if col in boolean_columns:

            df[col] = df[col].map(
                lambda x: (
                    None if pd.isna(x)
                    else str(x).lower() in {"true", "1"}
                )
            )

            continue

        # 2. Colonnes numériques
        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        # Les valeurs non numériques sont conservées comme NaN
        if converted.notna().sum() > 0:

            df[col] = converted

    return df



def build_events_enriched(raw_df):
    """
    Build three analytical tables from StatsBomb raw events.

    Original raw data is never modified.
    """

    df = raw_df.copy()

    # -------------------------------------------------
    # 1. EVENTS ENRICHED
    # -------------------------------------------------

    # Les colonnes freeze_frame et formation sont
    # des données répétées associées à un même événement.

    events = df.drop_duplicates(
        subset=["match_id", "id"]
    ).copy()

    # Supprimer les colonnes contenant les joueurs
    # des freeze frames et des formations.
    columns_to_remove = [
        col for col in events.columns
        if col.startswith("freeze_frame_")
        or col.startswith("formation_player_")
        or col.startswith("formation_position_")
        or col == "formation_jersey_number"
    ]

    events = events.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    events = events.rename(columns={
        "id": "event_id",
        "event_type_name": "type"
    })

    # -------------------------------------------------
    # 2. SHOT FREEZE FRAME
    # -------------------------------------------------

    freeze_columns = [
        "match_id",
        "id",
        "freeze_frame_player_id",
        "freeze_frame_player_name",
        "freeze_frame_player_position_id",
        "freeze_frame_player_position_name",
        "freeze_frame_teammate",
        "freeze_frame_x",
        "freeze_frame_y"
    ]

    freeze_columns = [
        col for col in freeze_columns
        if col in df.columns
    ]

    freeze_frame = df.loc[
        df["event_type_name"] == "Shot",
        freeze_columns
    ].copy()

    freeze_frame = freeze_frame.dropna(
        subset=["freeze_frame_player_id"]
    )

    freeze_frame = freeze_frame.rename(columns={
        "id": "event_id",
        "freeze_frame_player_id": "player_id",
        "freeze_frame_player_name": "player_name",
        "freeze_frame_player_position_id": "position_id",
        "freeze_frame_player_position_name": "position_name",
        "freeze_frame_teammate": "teammate",
        "freeze_frame_x": "x",
        "freeze_frame_y": "y"
    })

    freeze_frame = freeze_frame.drop_duplicates()

    # -------------------------------------------------
    # 3. FORMATION PLAYERS
    # -------------------------------------------------

    formation_columns = [
        "match_id",
        "id",
        "event_type_name",
        "team_id",
        "team_name",
        "formation_player_id",
        "formation_player_name",
        "formation_position_id",
        "formation_position_name",
        "formation_jersey_number"
    ]

    formation_columns = [
        col for col in formation_columns
        if col in df.columns
    ]

    formation_players = df.loc[
        df["event_type_name"].isin(
            ["Starting XI", "Tactical Shift"]
        ),
        formation_columns
    ].copy()

    formation_players = formation_players.dropna(
        subset=["formation_player_id"]
    )

    formation_players = formation_players.rename(
        columns={
            "id": "event_id",
            "formation_player_id": "player_id",
            "formation_player_name": "player_name",
            "formation_position_id": "position_id",
            "formation_position_name": "position_name",
            "formation_jersey_number": "jersey_number"
        }
    )

    formation_players = formation_players.drop_duplicates()

    return {
        "events_enriched": events.reset_index(drop=True),
        "shot_freeze_frame": freeze_frame.reset_index(drop=True),
        "formation_players": formation_players.reset_index(drop=True)
    }


def save_enriched_tables(connection, enriched_tables):

    for table_name, df in enriched_tables.items():

        print(f"Enregistrement de {table_name}...")

        # Normaliser les types du DataFrame
        df = normalize_dataframe_types(df)

        # Enregistrer temporairement dans DuckDB
        connection.register(
            "temp_dataframe",
            df
        )

        # Créer la table correspondante
        connection.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM temp_dataframe
        """)

        connection.unregister("temp_dataframe")

    print("Les trois tables ont été enregistrées.")