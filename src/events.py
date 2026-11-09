import pandas as pd


def build_events(df):

    events = df.copy()

    # Colonnes principales et enrichissement
    source_mapping = {
        "event_type_name": "type",
        "team_name": "team",
        "player_name": "player",
        "player_position_name": "position"
    }

    # Créer les colonnes standardisées sans supprimer
    # les colonnes StatsBomb originales
    for source, target in source_mapping.items():
        if source in events.columns:
            events[target] = events[source]
        else:
            events[target] = pd.NA

    # Garantir la présence des autres colonnes
    required_columns = [
        "match_id", "id", "index", "period", "timestamp",
        "type", "team", "player", "position",
        "team_id", "team_name",
        "player_id", "player_name",
        "player_position_id", "player_position_name",
        "location_x", "location_y"
    ]

    for column in required_columns:
        if column not in events.columns:
            events[column] = pd.NA

    # Construire les coordonnées
    events["location"] = [
        (x, y) if pd.notna(x) and pd.notna(y) else None
        for x, y in zip(
            events["location_x"],
            events["location_y"]
        )
    ]

    # Sélectionner les 16 colonnes demandées
    final_columns = [
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

    return events[final_columns].copy().reset_index(drop=True)