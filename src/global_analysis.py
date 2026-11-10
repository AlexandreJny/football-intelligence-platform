from pathlib import Path
import duckdb
import pandas as pd

MONACO_TEAM_ID = 129


def load_events(database_path, team_id=MONACO_TEAM_ID):
    """
    Load all events from matches played by a specific team.

    This includes events from both the selected team
    and its opponents.
    """

    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(
            f"Base de données introuvable : {database_path}"
        )

    connection = duckdb.connect(
        str(database_path),
        read_only=True
    )

    try:

        events = connection.execute(
            """
            SELECT e.*
            FROM events_enriched e

            INNER JOIN match_teams mt
                ON e.match_id = mt.match_id

            WHERE mt.team_id = ?

            ORDER BY e.match_id, e.index
            """,
            [team_id]
        ).fetchdf()

    finally:
        connection.close()

    return events


def get_shots(events):
    """
    Extract shot events.
    """

    required_columns = [
        "type",
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "statsbomb_xg",
        "outcome_name",
        "location_x",
        "location_y"
    ]

    missing = [
        col for col in required_columns
        if col not in events.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )

    shots = events[
        events["type"].str.lower() == "shot"
    ].copy()

    return shots.reset_index(drop=True)



def calculate_shot_metrics(shots):
    """
    Calculate general offensive shooting metrics.
    """

    total_shots = len(shots)

    total_xg = shots["statsbomb_xg"].sum()

    # Number of distinct matches
    matches = shots["match_id"].nunique()

    # Average xG per match
    average_xg = (
        total_xg / matches
        if matches > 0 else 0
    )

    xg_per_shot = (
        total_xg / total_shots
        if total_shots > 0 else 0
    )

    goals = (
        shots["outcome_name"].str.lower() == "goal"
    ).sum()

    conversion_rate = (
        goals / total_shots * 100
        if total_shots > 0 else 0
    )

    return {
        "total_shots": total_shots,
        "matches": matches,
        "shots_per_match": (
            total_shots / matches if matches else 0
        ),
        "total_xg": total_xg,
        "average_xg": average_xg,
        "xg_per_shot": xg_per_shot,
        "goals": int(goals),
        "conversion_rate": conversion_rate,
        "goals_minus_xg": goals - total_xg
    }



def analyze_shot_zones(shots):

    zones = shots.copy()

    x = pd.to_numeric(
        zones["location_x"],
        errors="coerce"
    )

    y = pd.to_numeric(
        zones["location_y"],
        errors="coerce"
    )

    distance_to_goal = pd.concat(
        [x, 120 - x],
        axis=1
    ).min(axis=1)

    valid = (
        x.between(0, 120)
        & y.between(0, 80)
    )

    # Outside penalty area
    outside = (
        valid
        & (distance_to_goal > 18)
        & ~y.between(30, 50)
    )

    # Central outside area
    central_outside = (
        valid
        & (distance_to_goal > 18)
        & y.between(30, 50)
    )

    # Penalty area
    penalty = (
        valid
        & (distance_to_goal <= 18)
        & ~y.between(30, 50)
    )

    # Central penalty area
    central_penalty = (
        valid
        & (distance_to_goal <= 18)
        & y.between(30, 50)
    )

    zones["shot_zone"] = "Unknown"

    zones.loc[outside, "shot_zone"] = (
        "Outside penalty area"
    )

    zones.loc[central_outside, "shot_zone"] = (
        "Central outside area"
    )

    zones.loc[penalty, "shot_zone"] = (
        "Penalty area"
    )

    zones.loc[central_penalty, "shot_zone"] = (
        "Central penalty area"
    )

    summary = (
        zones.groupby("shot_zone")
        .agg(
            total_shots=("event_id", "count"),
            total_xg=("statsbomb_xg", "sum"),
            average_xg=("statsbomb_xg", "mean")
        )
        .reset_index()
    )

    summary["shot_percentage"] = (
        summary["total_shots"] / len(shots) * 100
        if len(shots) else 0
    )

    return summary


def analyze_shot_assists(events, shots):
    """
    Link passes to shots using assisted_shot_id.
    Return descriptive information without IDs.
    """

    passes = events[
        (events["type"].str.lower() == "pass")
        & events["assisted_shot_id"].notna()
    ].copy()

    pass_columns = passes[
        [
            "event_id",
            "match_id",
            "team_id",
            "player_name",
            "assisted_shot_id"
        ]
    ].rename(columns={
        "event_id": "pass_event_id",
        "match_id": "pass_match_id",
        "team_id": "pass_team_id",
        "player_name": "passer_name"
    })

    shot_columns = shots[
        [
            "event_id",
            "match_id",
            "team_id",
            "player_name",
            "statsbomb_xg",
            "outcome_name"
        ]
    ].rename(columns={
        "event_id": "shot_event_id",
        "match_id": "shot_match_id",
        "team_id": "shot_team_id",
        "player_name": "shooter_name"
    })

    linked = pass_columns.merge(
        shot_columns,
        left_on="assisted_shot_id",
        right_on="shot_event_id",
        how="inner"
    )

    linked = linked[
        (linked["pass_match_id"] == linked["shot_match_id"])
        & (linked["pass_team_id"] == linked["shot_team_id"])
    ].copy()

    # Keep only descriptive columns
    linked = linked[
        [
            "pass_match_id",
            "passer_name",
            "shooter_name",
            "statsbomb_xg",
            "outcome_name"
        ]
    ]

    return linked



def run_offensive_analysis(database_path, team_id=MONACO_TEAM_ID):
    """
    Run the complete offensive analysis.
    """

    events = load_events(
        database_path,
        team_id
    )

    shots = get_shots(events)

    shots = shots[
        shots["team_id"] == team_id
    ].copy()

    metrics = calculate_shot_metrics(shots)

    zones = analyze_shot_zones(shots)

    assists = analyze_shot_assists(
        events,
        shots
    )

    return {
        "metrics": metrics,
        "shots": shots,
        "zones": zones,
        "assists": assists
    }
