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


def get_passes(events):
    """
    Extract all pass events from the dataset.

    Includes passes from both the selected team
    and its opponents.
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
        "end_location_y"
    ]

    missing = [
        col for col in required_columns
        if col not in events.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes pour l'analyse des passes : {missing}"
        )

    passes = events[
        events["type"].str.lower() == "pass"
    ].copy()

    coordinate_columns = [
        "location_x",
        "location_y",
        "end_location_x",
        "end_location_y"
    ]

    for col in coordinate_columns:
        passes[col] = pd.to_numeric(
            passes[col],
            errors="coerce"
        )

    passes = passes.dropna(
        subset=coordinate_columns
    )

    return passes.reset_index(drop=True)



def get_carries(events):
    """
    Extract all carry events from the dataset.

    Includes carries from both the selected team
    and its opponents.
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
        "end_location_y"
    ]

    missing = [
        col for col in required_columns
        if col not in events.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes pour l'analyse des conduites : {missing}"
        )

    carries = events[
        events["type"].str.lower() == "carries"
    ].copy()

    coordinate_columns = [
        "location_x",
        "location_y",
        "end_location_x",
        "end_location_y"
    ]

    for col in coordinate_columns:
        carries[col] = pd.to_numeric(
            carries[col],
            errors="coerce"
        )

    carries = carries.dropna(
        subset=coordinate_columns
    )

    return carries.reset_index(drop=True)



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
