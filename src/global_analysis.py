from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

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
            f"Database not found: {database_path}"
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

def _valid_coordinates(df, x_col="location_x", y_col="location_y"):
    """Keeps only events with valid coordinates."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return df.iloc[0:0].copy()

    result = df.copy()
    result[x_col] = pd.to_numeric(result[x_col], errors="coerce")
    result[y_col] = pd.to_numeric(result[y_col], errors="coerce")

    return result[
        result[x_col].between(0, 120)
        & result[y_col].between(0, 80)
    ].copy()

def _successful_passes(passes):
    """
    In StatsBomb, a null outcome_name generally indicates
    a completed pass.
    """
    if passes.empty:
        return passes.copy()

    return passes[passes["outcome_name"].isna()].copy()

def _get_coordinate_events(events, event_types, action_label):
    """
    Extract coordinate-based events with a common schema.

    Coordinates are converted to numeric values and rows without
    complete start/end coordinates are discarded.
    """
    required_columns = [
        "type",
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "player_name",
        "location_x",
        "location_y",
        "end_location_x",
        "end_location_y",
    ]

    missing = [
        column for column in required_columns
        if column not in events.columns
    ]
    if missing:
        raise ValueError(
            f"Missing columns for the {action_label} : {missing}"
        )

    normalized_types = {
        str(event_type).strip().casefold()
        for event_type in event_types
    }
    event_type_series = (
        events["type"].astype("string").str.strip().str.casefold()
    )
    selected = events[event_type_series.isin(normalized_types)].copy()

    coordinate_columns = [
        "location_x",
        "location_y",
        "end_location_x",
        "end_location_y",
    ]
    selected[coordinate_columns] = selected[coordinate_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    selected = selected.dropna(subset=coordinate_columns)

    return selected.reset_index(drop=True)

def get_passes(events):
    """
    Extract all pass events from the dataset.

    Includes passes from both the selected team and its opponents.
    """
    return _get_coordinate_events(
        events,
        event_types={"pass"},
        action_label="passes",
    )

def get_carries(events):
    """
    Extract all carry events from the dataset.

    Accepts singular and plural labels used by event data sources.
    """
    return _get_coordinate_events(
        events,
        event_types={"carry", "carries"},
        action_label="conduites",
    )

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

def prepare_line_breaking_passes(passes, team_id):
    """
    Prepare passes for potential line-breaking analysis.

    Keep only valid forward passes from the selected team.
    """

    required_columns = [
        "match_id",
        "period",
        "team_id",
        "player_id",
        "location_x",
        "location_y",
        "end_location_x",
        "end_location_y"
    ]

    missing = [
        col for col in required_columns
        if col not in passes.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = passes[
        passes["team_id"] == team_id
    ].copy()

    coordinate_columns = [
        "location_x",
        "location_y",
        "end_location_x",
        "end_location_y"
    ]

    for col in coordinate_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove invalid coordinates
    df = df.dropna(
        subset=coordinate_columns
    )

    # Keep coordinates inside the pitch
    valid_coordinates = (
        df["location_x"].between(0, 120)
        & df["end_location_x"].between(0, 120)
        & df["location_y"].between(0, 80)
        & df["end_location_y"].between(0, 80)
    )

    df = df[valid_coordinates].copy()

    # Calculate forward progression
    df["progression_x"] = (
        df["end_location_x"] - df["location_x"]
    )

    # Calculate Euclidean pass distance
    df["pass_distance"] = (
        (
            df["end_location_x"] - df["location_x"]
        ) ** 2
        +
        (
            df["end_location_y"] - df["location_y"]
        ) ** 2
    ) ** 0.5

    return df.reset_index(drop=True)

def _get_defensive_events(events, event_type, extra_filter=None):
    """
    Extract a defensive event type from the full event dataset.
    """
    required_columns = [
        "type",
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "player_name",
        "location_x",
        "location_y"
    ]

    missing = [
        col for col in required_columns
        if col not in events.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns for {event_type}: {missing}"
        )

    result = events[
        events["type"].str.lower() == event_type.lower()
    ].copy()

    if extra_filter is not None:
        result = result[extra_filter(result)].copy()

    for col in ["location_x", "location_y"]:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    result = result.dropna(
        subset=["location_x", "location_y"]
    )

    return result.reset_index(drop=True)

def get_interceptions(events):
    """Extract all interceptions."""
    return _get_defensive_events(
        events,
        "Interception"
    )

def get_tackles(events):
    """
    Extract tackles from Duel events.

    type_name distinguishes tackles from aerial duels.
    """
    if "type_name" not in events.columns:
        raise ValueError(
            "The 'type_name' column is required to identify tackles."
        )

    return _get_defensive_events(
        events,
        "Duel",
        extra_filter=lambda df:
            df["type_name"].str.lower() == "tackle"
    )

def get_pressures(events):
    """Extract all pressure events."""
    return _get_defensive_events(
        events,
        "Pressure"
    )

def get_recoveries(events):
    """Extract all ball recovery events."""
    return _get_defensive_events(
        events,
        "Ball Recovery"
    )

def get_blocks(events):
    """Extract all block events."""
    return _get_defensive_events(
        events,
        "Block"
    )

def get_fouls(events):
    """Extract all committed fouls."""
    return _get_defensive_events(
        events,
        "Foul Committed"
    )

def detect_line_breaking_passes(
    prepared_passes,
    min_progression=10,
    final_third_start=80
):
    """
    Detect potential line-breaking passes.

    A pass must:
    - Progress at least 10 meters forward.
    - Start before the final third.
    - Reach the final third.
    """

    df = prepared_passes.copy()

    line_breaking = df[
        (
            df["progression_x"] >= min_progression
        )
        &
        (
            df["location_x"] < final_third_start
        )
        &
        (
            df["end_location_x"] >= final_third_start
        )
    ].copy()

    return line_breaking.reset_index(drop=True)

def prepare_between_lines_receptions(events, team_id):
    """
    Prepare Monaco ball receptions for between-the-lines analysis.

    This is a geometric proxy, not a detection of actual
    defensive lines.
    """

    required_columns = [
        "match_id",
        "period",
        "team_id",
        "player_id",
        "location_x",
        "location_y"
    ]

    missing = [
        col for col in required_columns
        if col not in events.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    receptions = events[
        (events["type"].str.strip().str.lower() == "ball receipt*")
        & (events["team_id"] == team_id)
    ].copy()

    receptions["location_x"] = pd.to_numeric(
        receptions["location_x"],
        errors="coerce"
    )

    receptions["location_y"] = pd.to_numeric(
        receptions["location_y"],
        errors="coerce"
    )

    receptions = receptions.dropna(
        subset=["location_x", "location_y"]
    )

    valid_coordinates = (
        receptions["location_x"].between(0, 120)
        & receptions["location_y"].between(0, 80)
    )

    receptions = receptions[
        valid_coordinates
    ].copy()

    return receptions.reset_index(drop=True)

def detect_between_lines_receptions(
    prepared_receptions,
    final_third_start=80,
    central_y_min=20,
    central_y_max=60
):
    """
    Detect receptions in central areas of the final third.

    This is a geometric proxy for between-the-lines receptions.
    """

    df = prepared_receptions.copy()

    between_lines = df[
        (
            df["location_x"] >= final_third_start
        )
        &
        (
            df["location_y"].between(
                central_y_min,
                central_y_max
            )
        )
    ].copy()

    return between_lines.reset_index(drop=True)

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

def analyze_progressive_passes(passes, team_id=MONACO_TEAM_ID):
    """
    Identify progressive passes for a specific team.

    Provisional assumption:
    The selected team attacks towards x = 120.

    Progressive pass thresholds:
    - Own half: minimum 30 meters
    - Opponent half: minimum 15 meters
    - Final 40 meters: minimum 10 meters
    """

    team_passes = passes[
        passes["team_id"] == team_id
    ].copy()

    # Convert StatsBomb coordinates into meters
    scale_x = 105 / 120
    scale_y = 68 / 80

    start_x = team_passes["location_x"] * scale_x
    start_y = team_passes["location_y"] * scale_y

    end_x = team_passes["end_location_x"] * scale_x
    end_y = team_passes["end_location_y"] * scale_y

    # Euclidean distance
    team_passes["pass_distance"] = (
        (end_x - start_x) ** 2
        + (end_y - start_y) ** 2
    ) ** 0.5

    # Progression towards the opponent's goal
    team_passes["progression"] = end_x - start_x

    # Distance from the opponent's goal
    distance_to_goal = 105 - start_x

    # Progressive pass thresholds
    team_passes["progressive_threshold"] = 0

    team_passes.loc[
        distance_to_goal > 50,
        "progressive_threshold"
    ] = 30

    team_passes.loc[
        (distance_to_goal <= 50)
        & (distance_to_goal > 40),
        "progressive_threshold"
    ] = 15

    team_passes.loc[
        distance_to_goal <= 40,
        "progressive_threshold"
    ] = 10

    # Identify progressive passes
    team_passes["is_progressive"] = (
        team_passes["progression"]
        >= team_passes["progressive_threshold"]
    )

    progressive_passes = team_passes[
        team_passes["is_progressive"]
    ].copy()

    summary = {
        "total_progressive_passes": len(progressive_passes),
        "total_passes": len(team_passes),
        "progressive_passes_percentage": (
            len(progressive_passes) / len(team_passes) * 100
            if len(team_passes) else 0
        ),
        "average_progression": (
            progressive_passes["progression"].mean()
            if not progressive_passes.empty else 0
        ),
        "average_pass_distance": (
            progressive_passes["pass_distance"].mean()
            if not progressive_passes.empty else 0
        )
    }

    return {
        "summary": summary,
        "events": progressive_passes
    }

def analyze_progressive_carries(carries, team_id=MONACO_TEAM_ID):
    """
    Identify progressive carries for a specific team.

    Provisional assumption:
    The selected team attacks towards x = 120.

    Progressive carry thresholds:
    - Own half: minimum 30 meters
    - Opponent half: minimum 15 meters
    - Final 40 meters: minimum 10 meters
    """

    team_carries = carries[
        carries["team_id"] == team_id
    ].copy()

    # Convert StatsBomb coordinates into meters
    scale_x = 105 / 120
    scale_y = 68 / 80

    start_x = team_carries["location_x"] * scale_x
    start_y = team_carries["location_y"] * scale_y

    end_x = team_carries["end_location_x"] * scale_x
    end_y = team_carries["end_location_y"] * scale_y

    # Distance travelled
    team_carries["carry_distance"] = (
        (end_x - start_x) ** 2
        + (end_y - start_y) ** 2
    ) ** 0.5

    # Progression towards opponent's goal
    team_carries["progression"] = end_x - start_x

    # Distance from opponent's goal
    distance_to_goal = 105 - start_x

    # Progressive carry thresholds
    team_carries["progressive_threshold"] = 0

    team_carries.loc[
        distance_to_goal > 50,
        "progressive_threshold"
    ] = 30

    team_carries.loc[
        (distance_to_goal <= 50)
        & (distance_to_goal > 40),
        "progressive_threshold"
    ] = 15

    team_carries.loc[
        distance_to_goal <= 40,
        "progressive_threshold"
    ] = 10

    # Identify progressive carries
    team_carries["is_progressive"] = (
        team_carries["progression"]
        >= team_carries["progressive_threshold"]
    )

    progressive_carries = team_carries[
        team_carries["is_progressive"]
    ].copy()

    summary = {
        "total_progressive_carries": len(progressive_carries),
        "total_carries": len(team_carries),
        "progressive_carries_percentage": (
            len(progressive_carries) / len(team_carries) * 100
            if len(team_carries) else 0
        ),
        "average_progression": (
            progressive_carries["progression"].mean()
            if not progressive_carries.empty else 0
        ),
        "average_carry_distance": (
            progressive_carries["carry_distance"].mean()
            if not progressive_carries.empty else 0
        )
    }

    return {
        "summary": summary,
        "events": progressive_carries
    }

def analyze_final_third_passes(passes, team_id=MONACO_TEAM_ID):
    """
    Identify passes entering the opponent's final third.

    Provisional assumption:
    The selected team attacks towards x = 120.

    Final third:
    x >= 80 on StatsBomb's 120x80 pitch.
    """

    team_passes = passes[
        passes["team_id"] == team_id
    ].copy()

    # Identify passes entering the final third
    team_passes["starts_outside_final_third"] = (
        team_passes["location_x"] < 80
    )

    team_passes["ends_inside_final_third"] = (
        team_passes["end_location_x"] >= 80
    )

    team_passes["is_final_third_entry"] = (
        team_passes["starts_outside_final_third"]
        & team_passes["ends_inside_final_third"]
    )

    final_third_passes = team_passes[
        team_passes["is_final_third_entry"]
    ].copy()

    # Calculate progression
    final_third_passes["progression"] = (
        final_third_passes["end_location_x"]
        - final_third_passes["location_x"]
    ) * (105 / 120)

    # Summary
    summary = {
        "total_final_third_passes": len(final_third_passes),

        "total_passes": len(team_passes),

        "final_third_passes_percentage": (
            len(final_third_passes) / len(team_passes) * 100
            if len(team_passes) else 0
        ),

        "average_progression": (
            final_third_passes["progression"].mean()
            if not final_third_passes.empty else 0
        ),

        "matches": final_third_passes["match_id"].nunique(),

        "players_involved": (
            final_third_passes["player_id"].nunique()
        )
    }

    return {
        "summary": summary,
        "events": final_third_passes
    }

def analyze_line_breaking_passes(passes, team_id):
    """
    Analyze potential line-breaking passes.

    Uses geometric progression into the final third
    as a proxy for defensive line penetration.
    """

    # 1. Prepare passes
    prepared = prepare_line_breaking_passes(
        passes,
        team_id
    )

    # 2. Detect potential line-breaking passes
    line_breaking = detect_line_breaking_passes(
        prepared
    )

    total_passes = len(prepared)

    total_line_breaking = len(line_breaking)

    percentage = (
        total_line_breaking / total_passes * 100
        if total_passes > 0 else 0
    )

    matches = (
        line_breaking["match_id"].nunique()
        if not line_breaking.empty else 0
    )

    players = (
        line_breaking["player_id"].nunique()
        if not line_breaking.empty else 0
    )

    average_progression = (
        line_breaking["progression_x"].mean()
        if not line_breaking.empty else 0
    )

    average_distance = (
        line_breaking["pass_distance"].mean()
        if not line_breaking.empty else 0
    )

    summary = {
        "total_potential_line_breaks": total_line_breaking,
        "total_passes": total_passes,
        "potential_line_breaks_percentage": percentage,
        "average_progression": average_progression,
        "average_pass_distance": average_distance,
        "matches": matches,
        "players_involved": players
    }

    return {
        "summary": summary,
        "events": line_breaking
    }

def analyze_between_lines_receptions(events, team_id):
    """
    Analyze receptions in central areas of the final third.

    This is a geometric proxy for between-the-lines receptions,
    not a detection of actual defensive lines.
    """

    # 1. Prepare receptions
    prepared = prepare_between_lines_receptions(
        events,
        team_id
    )

    # 2. Detect receptions in the target area
    between_lines = detect_between_lines_receptions(
        prepared
    )

    total_receptions = len(prepared)

    total_between_lines = len(between_lines)

    percentage = (
        total_between_lines / total_receptions * 100
        if total_receptions > 0 else 0
    )

    matches = (
        between_lines["match_id"].nunique()
        if not between_lines.empty else 0
    )

    players = (
        between_lines["player_id"].nunique()
        if not between_lines.empty else 0
    )

    summary = {
        "total_between_lines_receptions": total_between_lines,
        "total_receptions": total_receptions,
        "between_lines_percentage": percentage,
        "matches": matches,
        "players_involved": players
    }

    return {
        "summary": summary,
        "events": between_lines
    }

def analyze_defensive_events(defensive_events, team_id):
    """
    Calculate basic defensive statistics for one team.
    """

    team_events = defensive_events[
        defensive_events["team_id"] == team_id
    ].copy()

    total = len(team_events)

    summary = {
        "total_events": total,
        "matches": (
            team_events["match_id"].nunique()
            if total > 0 else 0
        ),
        "players_involved": (
            team_events["player_id"].nunique()
            if total > 0 else 0
        )
    }

    if "outcome_name" in team_events.columns:
        summary["outcomes"] = (
            team_events["outcome_name"]
            .fillna("Unknown")
            .value_counts()
            .to_dict()
        )

    return {
        "summary": summary,
        "events": team_events.reset_index(drop=True)
    }

def analyze_recovery_zones(recoveries, team_id):
    """
    Analyze ball recoveries by pre-existing location bucket.
    """

    team_recoveries = recoveries[
        recoveries["team_id"] == team_id
    ].copy()

    if "location_bucket" not in team_recoveries.columns:
        raise ValueError(
            "The 'location_bucket' column is required."
        )

    total = len(team_recoveries)

    zones = (
        team_recoveries
        .groupby("location_bucket", dropna=False)
        .size()
        .reset_index(name="recoveries")
        .sort_values("recoveries", ascending=False)
    )

    zones["percentage"] = (
        zones["recoveries"] / total * 100
        if total > 0 else 0
    )

    return {
        "summary": {
            "total_recoveries": total,
            "zones": zones.to_dict("records")
        },
        "events": team_recoveries.reset_index(drop=True),
        "zones": zones.reset_index(drop=True)
    }

def calculate_ppda(events, team_id, halfway_line=60):
    """
    Calculate PPDA for a team, match by match.

    PPDA = opponent passes / defensive actions.

    Pressing area:
    - Opponent passes in their own half.
    - Defensive actions in the opponent half.

    Defensive actions included:
    - Tackles (Duel + type_name = Tackle)
    - Interceptions
    - Fouls committed

    Pressures, blocks and ball recoveries are not included
    in this PPDA definition.
    """

    required_columns = [
        "type",
        "type_name",
        "team_id",
        "match_id",
        "location_x",
        "attacking_direction"
    ]

    missing = [
        col for col in required_columns
        if col not in events.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns for PPDA: {missing}"
        )

    data = events.copy()

    data["location_x"] = pd.to_numeric(
        data["location_x"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["location_x", "attacking_direction"]
    )

    data["type_lower"] = data["type"].str.lower()
    data["direction_lower"] = (
        data["attacking_direction"].str.lower()
    )

    # Opponent passes in their own half.
    opponent_passes = data[
        (data["team_id"] != team_id)
        & (data["type_lower"] == "pass")
    ].copy()

    opponent_passes_in_zone = (
        (
            (opponent_passes["direction_lower"] == "left_to_right")
            & (opponent_passes["location_x"] <= halfway_line)
        )
        |
        (
            (opponent_passes["direction_lower"] == "right_to_left")
            & (opponent_passes["location_x"] >= halfway_line)
        )
    )

    opponent_passes = opponent_passes[
        opponent_passes_in_zone
    ]

    # Defensive actions by the selected team.
    team_actions = data[
        data["team_id"] == team_id
    ].copy()

    is_tackle = (
        (team_actions["type_lower"] == "duel")
        & (team_actions["type_name"].str.lower() == "tackle")
    )

    is_interception = (
        team_actions["type_lower"] == "interception"
    )

    is_foul = (
        team_actions["type_lower"] == "foul committed"
    )

    team_actions = team_actions[
        is_tackle | is_interception | is_foul
    ].copy()

    # Defensive actions in the opponent half.
    actions_in_zone = (
        (
            (team_actions["direction_lower"] == "left_to_right")
            & (team_actions["location_x"] >= halfway_line)
        )
        |
        (
            (team_actions["direction_lower"] == "right_to_left")
            & (team_actions["location_x"] <= halfway_line)
        )
    )

    team_actions = team_actions[actions_in_zone]

    # Calculate PPDA for each match.
    match_ids = data["match_id"].dropna().unique()

    results = []

    for match_id in match_ids:

        passes_count = len(
            opponent_passes[
                opponent_passes["match_id"] == match_id
            ]
        )

        actions_count = len(
            team_actions[
                team_actions["match_id"] == match_id
            ]
        )

        ppda = (
            passes_count / actions_count
            if actions_count > 0 else None
        )

        results.append({
            "match_id": match_id,
            "opponent_passes": passes_count,
            "defensive_actions": actions_count,
            "ppda": ppda
        })

    match_results = pd.DataFrame(
        results,
        columns=[
            "match_id",
            "opponent_passes",
            "defensive_actions",
            "ppda"
        ]
    )

    total_passes = match_results["opponent_passes"].sum()
    total_actions = match_results["defensive_actions"].sum()

    overall_ppda = (
        total_passes / total_actions
        if total_actions > 0 else None
    )

    return {
        "summary": {
            "total_opponent_passes": int(total_passes),
            "total_defensive_actions": int(total_actions),
            "ppda": overall_ppda
        },
        "matches": match_results
    }

def analyze_pass_network(passes):
    """
    Builds the passing network match by match.

    Retourne :
    - summary: overall statistics
    - nodes: average player positions per match
    - edges: passer/recipient relationships per match
    - events: completed passes used
    """

    required = [
        "match_id",
        "event_id",
        "player_name",
        "pass_recipient_name",
        "location_x",
        "location_y",
        "end_location_x",
        "end_location_y",
        "pass_length",
    ]

    if passes.empty or not all(
        col in passes.columns for col in required
    ):
        return {
            "summary": {
                "successful_passes": 0,
                "links": 0,
                "players": 0,
            },
            "nodes": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "events": pd.DataFrame(),
        }

    # 1. Successful passes
    successful = _successful_passes(passes)

    successful = successful.dropna(
        subset=[
            "match_id",
            "player_name",
            "pass_recipient_name",
        ]
    ).copy()

    successful = successful[
        successful["player_name"]
        != successful["pass_recipient_name"]
    ]

    successful = _valid_coordinates(successful)

    successful = _valid_coordinates(
        successful,
        "end_location_x",
        "end_location_y",
    )

    if successful.empty:
        return {
            "summary": {
                "successful_passes": 0,
                "links": 0,
                "players": 0,
            },
            "nodes": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "events": successful,
        }

    # 2. Edges: one relationship per match
    edges = (
        successful.groupby(
            [
                "match_id",
                "player_name",
                "pass_recipient_name",
            ],
            as_index=False,
        )
        .agg(
            passes=("event_id", "count"),
            avg_pass_length=("pass_length", "mean"),
        )
        .rename(
            columns={
                "player_name": "source",
                "pass_recipient_name": "target",
            }
        )
        .sort_values("passes", ascending=False)
        .reset_index(drop=True)
    )

    # 3. Passer positions
    passer_positions = successful[
        [
            "match_id",
            "player_name",
            "location_x",
            "location_y",
        ]
    ].rename(
        columns={
            "player_name": "player",
            "location_x": "x",
            "location_y": "y",
        }
    )

    # 4. Receiver positions
    receiver_positions = successful[
        [
            "match_id",
            "pass_recipient_name",
            "end_location_x",
            "end_location_y",
        ]
    ].rename(
        columns={
            "pass_recipient_name": "player",
            "end_location_x": "x",
            "end_location_y": "y",
        }
    )

    positions = pd.concat(
        [
            passer_positions,
            receiver_positions,
        ],
        ignore_index=True,
    )

    # 5. Nodes: one player per match
    nodes = (
        positions.groupby(
            ["match_id", "player"],
            as_index=False,
        )
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
        )
    )

    # 6. Pass counts per player and match
    pass_counts = (
        successful.groupby(
            ["match_id", "player_name"]
        )
        .size()
        .rename("passes")
        .reset_index()
        .rename(columns={"player_name": "player"})
    )

    # 7. Reception counts per player and match
    reception_counts = (
        successful.groupby(
            ["match_id", "pass_recipient_name"]
        )
        .size()
        .rename("receptions")
        .reset_index()
        .rename(
            columns={"pass_recipient_name": "player"}
        )
    )

    nodes = nodes.merge(
        pass_counts,
        on=["match_id", "player"],
        how="left",
    )

    nodes = nodes.merge(
        reception_counts,
        on=["match_id", "player"],
        how="left",
    )

    nodes[["passes", "receptions"]] = (
        nodes[["passes", "receptions"]]
        .fillna(0)
        .astype(int)
    )

    # 8. General summary
    summary = {
        "successful_passes": len(successful),
        "links": len(edges),
        "players": nodes["player"].nunique(),
        "matches": successful["match_id"].nunique(),
    }

    return {
        "summary": summary,
        "nodes": nodes,
        "edges": edges,
        "events": successful.reset_index(drop=True),
    }

def analyze_player_heatmaps(team_events, bins_x=12, bins_y=8):
    """
    Calculates each player's activity zones.

    Heatmaps use the starting coordinates of events.
    Included events are passes, carries, receptions, dribbles, and shots.
    """

    if team_events.empty:
        return {
            "summary": {"players": 0, "events": 0},
            "zones": pd.DataFrame(),
            "events": pd.DataFrame(),
        }

    activity_types = [
        "pass",
        "carry",
        "carries",
        "ball receipt*",
        "dribble",
        "shot",
    ]

    events = team_events[
        team_events["type"].astype(str).str.lower().isin(activity_types)
    ].copy()

    events = _valid_coordinates(events)

    if events.empty or "player_name" not in events.columns:
        return {
            "summary": {"players": 0, "events": 0},
            "zones": pd.DataFrame(),
            "events": events,
        }

    events = events.dropna(subset=["player_name"])

    events["x_bin"] = pd.cut(
        events["location_x"],
        bins=np.linspace(0, 120, bins_x + 1),
        include_lowest=True,
        labels=False,
    )

    events["y_bin"] = pd.cut(
        events["location_y"],
        bins=np.linspace(0, 80, bins_y + 1),
        include_lowest=True,
        labels=False,
    )

    zones = (
        events.groupby(
            ["player_name", "x_bin", "y_bin"],
            observed=True,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "events"})
    )

    zones["percentage"] = (
        zones["events"]
        / zones.groupby("player_name")["events"].transform("sum")
        * 100
    )

    summary = {
        "players": events["player_name"].nunique(),
        "events": len(events),
    }

    return {
        "summary": summary,
        "zones": zones,
        "events": events,
    }

def analyze_space_occupation(team_events):
    """
    Distributes attacking actions across a grid of 3 longitudinal zones
    and 5 pitch lanes.
    """

    if team_events.empty:
        return {
            "summary": {"events": 0, "zones_used": 0},
            "zones": pd.DataFrame(),
            "events": pd.DataFrame(),
        }

    activity_types = [
        "pass",
        "carry",
        "carries",
        "ball receipt*",
        "dribble",
    ]

    events = team_events[
        team_events["type"].astype(str).str.lower().isin(activity_types)
    ].copy()

    events = _valid_coordinates(events)

    if events.empty:
        return {
            "summary": {"events": 0, "zones_used": 0},
            "zones": pd.DataFrame(),
            "events": events,
        }

    events["longitudinal_zone"] = pd.cut(
        events["location_x"],
        bins=[0, 40, 80, 120],
        labels=["Defensive", "Middle", "Attacking"],
        include_lowest=True,
    )

    events["lateral_zone"] = pd.cut(
        events["location_y"],
        bins=[0, 16, 32, 48, 64, 80],
        labels=[
            "Couloir haut",
            "Demi-espace haut",
            "Axe",
            "Demi-espace bas",
            "Couloir bas",
        ],
        include_lowest=True,
    )

    zones = (
        events.groupby(
            ["longitudinal_zone", "lateral_zone"],
            observed=True,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "events"})
    )

    total = zones["events"].sum()

    zones["percentage"] = (
        zones["events"] / total * 100
        if total > 0
        else 0
    )

    zones = zones.sort_values("events", ascending=False)

    summary = {
        "events": len(events),
        "zones_used": len(zones),
    }

    return {
        "summary": summary,
        "zones": zones,
        "events": events,
    }

def analyze_chance_creation(team_events, team_id):
    """
    Identifies shot-assisted passes using assisted_shot_id.
    Calculates the xG of the associated shots and distributes
    chance-creating passes across pitch zones.
    """

    empty_result = {
        "summary": {
            "chance_creating_passes": 0,
            "linked_shots": 0,
            "xg_created": 0.0,
        },
        "passes": pd.DataFrame(),
        "zones": pd.DataFrame(),
        "shots": pd.DataFrame(),
    }

    if team_events.empty:
        return empty_result

    # --------------------------------------------------------
    # 1. Extract AS Monaco events
    # --------------------------------------------------------

    team_events = team_events[
        team_events["team_id"] == team_id
    ].copy()

    if team_events.empty:
        return empty_result

    # --------------------------------------------------------
    # 2. Extract AS Monaco passes
    # --------------------------------------------------------

    passes = team_events[
        team_events["type"].astype(str).str.lower().eq("pass")
    ].copy()

    if passes.empty or "assisted_shot_id" not in passes.columns:
        return empty_result

    # Do NOT filter on outcome_name:
    # a shot-assisted pass may be incomplete.

    passes["assisted_shot_id"] = (
        passes["assisted_shot_id"].astype("string")
    )

    chance_passes = passes[
        passes["assisted_shot_id"].notna()
    ].copy()

    chance_passes = _valid_coordinates(chance_passes)

    if chance_passes.empty:
        return empty_result

    # --------------------------------------------------------
    # 3. Extract AS Monaco shots
    # --------------------------------------------------------

    shots = team_events[
        team_events["type"].astype(str).str.lower().eq("shot")
    ].copy()

    required_shot_columns = {
        "event_id",
        "match_id",
        "statsbomb_xg",
    }

    if not required_shot_columns.issubset(shots.columns):
        return empty_result

    shots = shots[
        ["event_id", "match_id", "statsbomb_xg"]
    ].rename(
        columns={
            "event_id": "shot_event_id",
            "statsbomb_xg": "shot_xg",
        }
    )

    # --------------------------------------------------------
    # 4. Link passes to shots
    # --------------------------------------------------------

    linked_shots = chance_passes.merge(
        shots,
        left_on=["assisted_shot_id", "match_id"],
        right_on=["shot_event_id", "match_id"],
        how="inner",
    )

    # --------------------------------------------------------
    # 5. Distribution of chance-creating passes by zone
    # --------------------------------------------------------

    zones_data = chance_passes.copy()

    zones_data["x_zone"] = pd.cut(
        zones_data["location_x"],
        bins=[0, 40, 80, 120],
        labels=["Defensive", "Middle", "Attacking"],
        include_lowest=True,
    )

    zones_data["y_zone"] = pd.cut(
        zones_data["location_y"],
        bins=[0, 16, 32, 48, 64, 80],
        labels=[
            "Couloir haut",
            "Demi-espace haut",
            "Axe",
            "Demi-espace bas",
            "Couloir bas",
        ],
        include_lowest=True,
    )

    zones = (
        zones_data.groupby(
            ["x_zone", "y_zone"],
            observed=True,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "chance_passes"})
        .sort_values(
            "chance_passes",
            ascending=False,
        )
    )

    # --------------------------------------------------------
    # 6. Calculate created xG
    # --------------------------------------------------------

    xg_created = pd.to_numeric(
        linked_shots["shot_xg"],
        errors="coerce",
    ).sum()

    # --------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------

    summary = {
        "chance_creating_passes": len(chance_passes),
        "linked_shots": len(linked_shots),
        "xg_created": float(xg_created),
    }

    return {
        "summary": summary,
        "passes": chance_passes,
        "zones": zones,
        "shots": linked_shots,
    }

def analyze_attacking_patterns(team_events):
    """
    Analyzes attacking patterns based on possessions.

    Produit :
    - passing and carrying sequences per possession;
    - the most frequent passer-recipient combinations.

    This analysis describes on-ball sequences,
    not off-ball movements.
    """

    empty_result = {
        "summary": {
            "possessions": 0,
            "combinations": 0,
        },
        "sequences": pd.DataFrame(),
        "combinations": pd.DataFrame(),
    }

    if team_events.empty:
        return empty_result

    required = ["match_id", "period", "possession", "index", "type"]

    missing = [
        col for col in required
        if col not in team_events.columns
    ]

    if missing:
        raise ValueError(
            f"Attacking patterns: missing columns: {missing}"
        )

    events = team_events[
        team_events["type"].astype(str).str.lower().isin(
            ["pass", "carry", "carries"]
        )
    ].copy()

    events = events.sort_values(
        ["match_id", "period", "possession", "index"]
    )

    if events.empty:
        return empty_result

    sequences = []

    for (match_id, period, possession), group in events.groupby(
        ["match_id", "period", "possession"],
        dropna=False,
        sort=False,
    ):
        actions = []

        for _, event in group.iterrows():
            player = event.get("player_name")
            event_type = str(event.get("type", "")).lower()

            if pd.isna(player):
                continue

            if event_type == "pass":
                recipient = event.get("pass_recipient_name")

                if pd.notna(recipient):
                    actions.append(f"{player} → {recipient}")
                else:
                    actions.append(f"{player} (passe)")

            else:
                actions.append(f"{player} (conduite)")

        if actions:
            sequences.append({
                "match_id": match_id,
                "period": period,
                "possession": possession,
                "actions_count": len(actions),
                "sequence": " | ".join(actions),
            })

    sequences_df = pd.DataFrame(sequences)

    passes = events[
        events["type"].astype(str).str.lower() == "pass"
    ].copy()

    passes = passes.dropna(
        subset=["player_name", "pass_recipient_name"]
    )

    combinations = (
        passes.groupby(
            ["player_name", "pass_recipient_name"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "player_name": "source",
                "pass_recipient_name": "target",
                "size": "passes",
            }
        )
        .sort_values("passes", ascending=False)
    )

    summary = {
        "possessions": len(sequences_df),
        "combinations": len(combinations),
    }

    return {
        "summary": summary,
        "sequences": sequences_df,
        "combinations": combinations,
    }

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

def run_progression_analysis(
    database_path,
    team_id=MONACO_TEAM_ID
):
    """
    Run the complete ball progression analysis.
    Return clean summary metrics and individual events.
    """

    # 1. Load events
    events = load_events(
        database_path,
        team_id
    )

    # 2. Extract passes and carries
    passes = get_passes(events)
    carries = get_carries(events)

    # 3. Progressive passes
    progressive_passes = analyze_progressive_passes(
        passes,
        team_id
    )

    # 4. Progressive carries
    progressive_carries = analyze_progressive_carries(
        carries,
        team_id
    )

    # 5. Final third passes
    final_third_passes = analyze_final_third_passes(
        passes,
        team_id
    )

    # 6. Potential line-breaking passes
    line_breaking_passes = analyze_line_breaking_passes(
        passes,
        team_id
    )

    # 7. Between-the-lines receptions
    between_lines_receptions = analyze_between_lines_receptions(
        events,
        team_id
    )

    # 8. Clean result structure
    return {

        "progressive_passes": {
            "summary": progressive_passes["summary"],
            "events": progressive_passes["events"]
        },

        "progressive_carries": {
            "summary": progressive_carries["summary"],
            "events": progressive_carries["events"]
        },

        "final_third_passes": {
            "summary": final_third_passes["summary"],
            "events": final_third_passes["events"]
        },

        "line_breaking_passes": {
            "summary": line_breaking_passes["summary"],
            "events": line_breaking_passes["events"]
        },

        "between_lines_receptions": {
            "summary": between_lines_receptions["summary"],
            "events": between_lines_receptions["events"]
        }
    }

def run_defensive_analysis(
    database_path,
    team_id=MONACO_TEAM_ID
):
    """
    Run the complete defensive analysis.
    """

    events = load_events(
        database_path,
        team_id
    )

    interceptions = get_interceptions(events)
    tackles = get_tackles(events)
    pressures = get_pressures(events)
    recoveries = get_recoveries(events)
    blocks = get_blocks(events)
    fouls = get_fouls(events)

    return {
        "interceptions": analyze_defensive_events(
            interceptions, team_id
        ),

        "tackles": analyze_defensive_events(
            tackles, team_id
        ),

        "pressures": analyze_defensive_events(
            pressures, team_id
        ),

        "recoveries": analyze_defensive_events(
            recoveries, team_id
        ),

        "blocks": analyze_defensive_events(
            blocks, team_id
        ),

        "fouls": analyze_defensive_events(
            fouls, team_id
        ),

        "recovery_zones": analyze_recovery_zones(
            recoveries, team_id
        ),

        "ppda": calculate_ppda(
            events, team_id
        )
    }

def run_collective_analysis(database_path, team_id=MONACO_TEAM_ID):
    """
    Runs AS Monaco's five collective analyses.

    load_events loads the matches AS Monaco participated in.
    AS Monaco events are then filtered for collective analyses
    to avoid mixing them with opposition actions.
    """

    events = load_events(
        database_path=database_path,
        team_id=team_id,
    )

    if events.empty:
        raise ValueError(
            f"No events found for team {team_id}."
        )

    team_events = events[
        events["team_id"] == team_id
    ].copy()

    passes = get_passes(events)

    team_passes = passes[
        passes["team_id"] == team_id
    ].copy()

    return {
        "pass_network": analyze_pass_network(team_passes),

        "heatmaps": analyze_player_heatmaps(
            team_events
        ),

        "space_occupation": analyze_space_occupation(
            team_events
        ),

        "chance_creation": analyze_chance_creation(
            events,
            team_id,
        ),

        "attacking_patterns": analyze_attacking_patterns(
            team_events
        ),
    }
