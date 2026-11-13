import sys
from pathlib import Path

# =========================================
# PROJECT PATH
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from numbers import Real

from config import (
    DB_PATH,
    APP_TITLE,
    COLORS,
    TEAM_ID,
    TEAM_NAME,
    CACHE_TTL_SECONDS,
    PITCH_LENGTH,
    PITCH_WIDTH
)

from src.global_analysis import (
    run_offensive_analysis,
    run_progression_analysis,
    run_defensive_analysis,
    run_collective_analysis,
)

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLORS["background"]};
        color: {COLORS["text"]};
    }}

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    h1, h2, h3 {{
        color: {COLORS["text"]} !important;
        font-weight: 700 !important;
    }}

    p, label {{
        color: {COLORS["text"]};
    }}

    .dashboard-header {{
        background-color: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 25px 30px;
        margin-bottom: 25px;
    }}

    .dashboard-subtitle {{
        color: {COLORS["muted"]};
        font-size: 14px;
        letter-spacing: 1px;
        font-weight: 600;
    }}

    .team-name {{
        color: {COLORS["primary"]};
        font-weight: 700;
    }}

    .dashboard-description {{
        color: {COLORS["muted"]};
        font-size: 14px;
        margin-bottom: 0;
    }}

    [data-testid="stMetric"] {{
        background-color: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(20, 40, 70, 0.04);
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS["muted"]} !important;
        font-weight: 600;
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS["primary"]} !important;
        font-weight: 700;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        overflow: hidden;
    }}

    hr {{
        border-color: {COLORS["border"]};
    }}

    .section-label {{
        color: {COLORS["muted"]};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# SHOT ZONE LEGEND
# =========================================

def display_shot_zone_legend(zones_df):

    if zones_df.empty:
        st.warning("No shot zone data available.")
        return

    zone_styles = {
        "outside penalty area": {
            "color": "#F0A43A",
            "description": "Area outside the penalty box."
        },
        "central outside area": {
            "color": "#F5CB78",
            "description": "Central channel in front of the penalty box."
        },
        "penalty area": {
            "color": "#3976B9",
            "description": "Entire penalty area."
        },
        "central penalty area": {
            "color": "#159B72",
            "description": "Central area inside the penalty box."
        }
    }

    st.markdown("### Shot distribution by zone")

    columns = st.columns(2)

    for index, row in zones_df.iterrows():

        zone_name = str(row["shot_zone"]).strip()
        zone_key = zone_name.casefold()

        total_shots = int(row["total_shots"])
        percentage = float(row["shot_percentage"])
        total_xg = float(row["total_xg"])
        average_xg = float(row["average_xg"])

        style = zone_styles.get(
            zone_key,
            {
                "color": "#718096",
                "description": "Shot area."
            }
        )

        with columns[index % 2]:

            st.markdown(f"### {zone_name.title()}")

            st.caption(style["description"])

            st.metric(
                label="Total shots",
                value=total_shots
            )

            st.progress(
                min(max(percentage / 100, 0), 1)
            )

            st.caption(
                f"{percentage:.1f}% of total shots"
            )

            st.caption(
                f"Total xG: {total_xg:.2f} | "
                f"Average xG: {average_xg:.3f}"
            )

# =========================================
# FOOTBALL PITCH
# =========================================

# =========================================
# SHARED FOOTBALL PITCH
# =========================================

def add_football_pitch(
    fig,
    *,
    pitch_color="#B8C2CC",
    line_width=1.5,
    height=600,
    x_range=(-2, PITCH_LENGTH + 2),
    y_range=(-2, PITCH_WIDTH + 2),
    invert_y=False,
    background_color="#FFFFFF",
    paper_color=None,
    show_legend=False,
):
    """
    Adds a standard football pitch to an existing Plotly figure.

    Coordinates used: 120 x 80.
    Call this on the figure BEFORE or AFTER adding data traces:
    - the pitch background is placed beneath the traces;
    - the pitch markings are placed above the traces to remain visible
      on heatmaps.

    `invert_y=True` preserves the orientation of heatmaps
    whose vertical axis runs from 80 down to 0.
    """
    if fig is None:
        fig = go.Figure()

    transparent = "rgba(0,0,0,0)"

    # White background and outer boundary
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH,
        fillcolor=background_color,
        line=dict(color=pitch_color, width=line_width),
        layer="below"
    )

    # Halfway line
    fig.add_shape(
        type="line",
        x0=PITCH_LENGTH / 2, y0=0,
        x1=PITCH_LENGTH / 2, y1=PITCH_WIDTH,
        line=dict(color=pitch_color, width=line_width),
        layer="above"
    )

    # Center circle and center spot
    fig.add_shape(
        type="circle",
        x0=PITCH_LENGTH / 2 - 10,
        y0=PITCH_WIDTH / 2 - 10,
        x1=PITCH_LENGTH / 2 + 10,
        y1=PITCH_WIDTH / 2 + 10,
        line=dict(color=pitch_color, width=line_width),
        fillcolor=transparent,
        layer="above"
    )
    fig.add_shape(
        type="circle",
        x0=PITCH_LENGTH / 2 - 0.4,
        y0=PITCH_WIDTH / 2 - 0.4,
        x1=PITCH_LENGTH / 2 + 0.4,
        y1=PITCH_WIDTH / 2 + 0.4,
        line=dict(color=pitch_color, width=1),
        fillcolor=pitch_color,
        layer="above"
    )

    # Penalty areas
    for x0, x1 in (
        (0, 18),
        (PITCH_LENGTH - 18, PITCH_LENGTH),
    ):
        fig.add_shape(
            type="rect",
            x0=x0, y0=18, x1=x1, y1=PITCH_WIDTH - 18,
            line=dict(color=pitch_color, width=line_width),
            fillcolor=transparent,
            layer="above"
        )

    # Goal areas
    for x0, x1 in (
        (0, 6),
        (PITCH_LENGTH - 6, PITCH_LENGTH),
    ):
        fig.add_shape(
            type="rect",
            x0=x0, y0=30, x1=x1, y1=PITCH_WIDTH - 30,
            line=dict(color=pitch_color, width=line_width),
            fillcolor=transparent,
            layer="above"
        )

    # Penalty spots
    for x in (12, PITCH_LENGTH - 12):
        fig.add_shape(
            type="circle",
            x0=x - 0.4, y0=PITCH_WIDTH / 2 - 0.4,
            x1=x + 0.4, y1=PITCH_WIDTH / 2 + 0.4,
            line=dict(color=pitch_color, width=1),
            fillcolor=pitch_color,
            layer="above"
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=paper_color or COLORS["card"],
        plot_bgcolor=background_color,
        height=height,
        margin=dict(l=5, r=5, t=10, b=10),
        showlegend=show_legend,
    )

    fig.update_xaxes(
        range=list(x_range),
        visible=False,
        fixedrange=True,
        constrain="domain",
    )
    fig.update_yaxes(
        range=list(y_range[::-1] if invert_y else y_range),
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True,
    )

    return fig

def create_shot_zone_pitch(zones_df=None):

    fig = go.Figure()

    zone_colors = {
        "outside penalty area": "#F0A43A",
        "central outside area": "#F5CB78",
        "penalty area": "#3976B9",
        "central penalty area": "#159B72"
    }

    zone_counts = {}
    if zones_df is not None and not zones_df.empty:
        zone_counts = {
            str(row["shot_zone"]).strip().casefold(): int(row["total_shots"])
            for _, row in zones_df.iterrows()
        }

    # The pitch background is added before the zones; the pitch markings
    # remain on top thanks to their layer="above".
    add_football_pitch(
        fig,
        height=600,
        x_range=(-4, 124),
        y_range=(-2, 82),
    )

    zones = [
        {
            "name": "Outside penalty area",
            "x0": 80, "x1": 120, "y0": 0, "y1": 18,
            "color": zone_colors["outside penalty area"], "label_y": 69
        },
        {
            "name": "Outside penalty area",
            "x0": 80, "x1": 120, "y0": 62, "y1": 80,
            "color": zone_colors["outside penalty area"], "label_y": 69
        },
        {
            "name": "Central outside area",
            "x0": 80, "x1": 102, "y0": 18, "y1": 62,
            "color": zone_colors["central outside area"], "label_y": 40
        },
        {
            "name": "Penalty area",
            "x0": 102, "x1": 120, "y0": 18, "y1": 62,
            "color": zone_colors["penalty area"], "label_y": 24
        },
        {
            "name": "Central penalty area",
            "x0": 102, "x1": 120, "y0": 30, "y1": 50,
            "color": zone_colors["central penalty area"], "label_y": 40
        }
    ]

    for zone in zones:
        count = zone_counts.get(zone["name"].casefold(), 0)
        center_x = (zone["x0"] + zone["x1"]) / 2

        fig.add_shape(
            type="rect",
            x0=zone["x0"], y0=zone["y0"],
            x1=zone["x1"], y1=zone["y1"],
            fillcolor=zone["color"],
            opacity=0.75,
            line=dict(color="#FFFFFF", width=1),
            layer="below",
        )

        fig.add_annotation(
            x=center_x,
            y=zone["label_y"],
            text=f"<b>{count}</b><br>SHOTS",
            showarrow=False,
            font=dict(size=14, color="#263746"),
            bgcolor="rgba(255,255,255,0.88)",
            borderpad=5,
            align="center",
        )

    # Goals are drawn outside the touchlines.
    for x0, x1 in ((-2, 0), (120, 122)):
        fig.add_shape(
            type="rect",
            x0=x0, y0=36, x1=x1, y1=44,
            line=dict(color="#B8C2CC", width=2),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    return fig

def create_progression_pitch():
    return add_football_pitch(go.Figure(), height=600)

def create_final_third_pitch():
    fig = add_football_pitch(go.Figure(), height=550)
    # Highlight the final third beneath the data traces and pitch markings.
    fig.add_shape(
        type="rect",
        x0=80, y0=0, x1=120, y1=80,
        fillcolor="rgba(21,155,114,0.13)",
        line=dict(color="#159B72", width=2, dash="dash"),
        layer="below",
    )
    return fig

def create_between_lines_pitch():
    return add_football_pitch(go.Figure(), height=550)

def create_defensive_pitch():
    return add_football_pitch(go.Figure(), height=600, show_legend=True)

def add_progressive_passes_to_pitch(fig, passes):

    if passes is None or passes.empty:
        return fig

    required_columns = [
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
        st.warning(f"Missing columns: {missing}")
        return fig

    valid = passes.dropna(
        subset=required_columns
    ).copy()

    if valid.empty:
        return fig

    # One trace for all pass origins
    fig.add_trace(
        go.Scatter(
            x=valid["location_x"],
            y=valid["location_y"],
            mode="markers",
            marker=dict(
                size=5,
                color="#3976B9",
                opacity=0.65
            ),
            text=valid.get(
                "player_name",
                pd.Series("Unknown", index=valid.index)
            ),
            customdata=valid[
                [
                    "end_location_x",
                    "end_location_y"
                ]
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Start: (%{x:.1f}, %{y:.1f})<br>"
                "End: (%{customdata[0]:.1f}, "
                "%{customdata[1]:.1f})"
                "<extra>Progressive Pass</extra>"
            ),
            showlegend=False
        )
    )

    # One trace for all pass trajectories
    xs = []
    ys = []

    for _, row in valid.iterrows():

        xs.extend([
            row["location_x"],
            row["end_location_x"],
            None
        ])

        ys.extend([
            row["location_y"],
            row["end_location_y"],
            None
        ])

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(
                color="#3976B9",
                width=1.5
            ),
            opacity=0.55,
            hoverinfo="skip",
            showlegend=False
        )
    )

    return fig

def add_progressive_carries_to_pitch(fig, carries):

    if carries is None or carries.empty:
        return fig

    required_columns = [
        "location_x",
        "location_y",
        "end_location_x",
        "end_location_y"
    ]

    missing = [
        col for col in required_columns
        if col not in carries.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    valid_carries = carries.dropna(
        subset=required_columns
    ).copy()

    for _, row in valid_carries.iterrows():

        start_x = row["location_x"]
        start_y = row["location_y"]

        end_x = row["end_location_x"]
        end_y = row["end_location_y"]

        player = row.get(
            "player_name",
            "Unknown player"
        )

        fig.add_trace(
            go.Scatter(
                x=[start_x, end_x],
                y=[start_y, end_y],
                mode="lines+markers",
                line=dict(
                    color="#159B72",
                    width=2
                ),
                marker=dict(
                    size=[5, 7],
                    color="#159B72",
                    symbol=["circle", "triangle-up"]
                ),
                name="Progressive Carries",
                showlegend=False,
                hovertemplate=(
                    f"<b>{player}</b><br>"
                    f"Start: ({start_x:.1f}, {start_y:.1f})"
                    f"<br>End: ({end_x:.1f}, {end_y:.1f})"
                    "<extra>Progressive Carry</extra>"
                )
            )
        )

    return fig

def add_final_third_passes_to_pitch(fig, passes):

    if passes is None or passes.empty:
        return fig

    required_columns = [
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
        st.warning(f"Missing columns: {missing}")
        return fig

    valid_passes = passes.dropna(
        subset=required_columns
    ).copy()

    # Keep passes ending in the final third
    valid_passes = valid_passes[
        valid_passes["end_location_x"].between(80, 120)
    ]

    if valid_passes.empty:
        return fig

    # Build all trajectories in one trace
    xs = []
    ys = []

    for _, row in valid_passes.iterrows():

        xs.extend([
            row["location_x"],
            row["end_location_x"],
            None
        ])

        ys.extend([
            row["location_y"],
            row["end_location_y"],
            None
        ])

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(
                color="#3976B9",
                width=1.2
            ),
            opacity=0.55,
            hoverinfo="skip",
            showlegend=False
        )
    )

    # Add reception points
    fig.add_trace(
        go.Scatter(
            x=valid_passes["end_location_x"],
            y=valid_passes["end_location_y"],
            mode="markers",
            marker=dict(
                size=5,
                color="#159B72",
                opacity=0.7
            ),
            text=valid_passes.get(
                "player_name",
                pd.Series(
                    "Unknown",
                    index=valid_passes.index
                )
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Reception: (%{x:.1f}, %{y:.1f})"
                "<extra>Final Third Entry</extra>"
            ),
            showlegend=False
        )
    )

    return fig

def add_between_lines_heatmap(fig, receptions):

    if receptions is None or receptions.empty:
        return fig

    required_columns = [
        "location_x",
        "location_y"
    ]

    missing = [
        col for col in required_columns
        if col not in receptions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    valid_receptions = receptions.dropna(
        subset=required_columns
    ).copy()

    valid_receptions = valid_receptions[
        valid_receptions["location_x"].between(0, 120)
        & valid_receptions["location_y"].between(0, 80)
    ]

    if valid_receptions.empty:
        return fig

    fig.add_trace(
        go.Histogram2dContour(
            x=valid_receptions["location_x"],
            y=valid_receptions["location_y"],
            colorscale=[
                [0, "rgba(255,255,255,0)"],
                [0.25, "#F5CB78"],
                [0.60, "#F0A43A"],
                [1, "#C0392B"]
            ],
            contours=dict(
                coloring="fill",
                showlines=False
            ),
            line=dict(
                width=0
            ),
            showscale=True,
            colorbar=dict(
                title="Reception density"
            ),
            hovertemplate=(
                "X: %{x:.1f}<br>"
                "Y: %{y:.1f}"
                "<extra>Reception density</extra>"
            ),
            ncontours=15,
            opacity=0.75
        )
    )

    return fig

def add_defensive_actions_to_pitch(fig, events):

    if events is None or events.empty:
        return fig

    required_columns = [
        "location_x",
        "location_y",
        "defensive_action"
    ]

    missing = [
        col for col in required_columns
        if col not in events.columns
    ]

    if missing:
        st.warning(f"Missing columns: {missing}")
        return fig

    valid = events.dropna(
        subset=["location_x", "location_y"]
    ).copy()

    valid = valid[
        valid["location_x"].between(0, 120)
        & valid["location_y"].between(0, 80)
    ]

    action_colors = {
        "Interceptions": "#3976B9",
        "Tackles": "#159B72",
        "Pressures": "#F0A43A",
        "Recoveries": "#805AD5",
        "Blocks": "#C0392B",
        "Fouls": "#718096"
    }

    for action, group in valid.groupby("defensive_action"):

        player_names = group.get(
            "player_name",
            pd.Series("Unknown", index=group.index)
        )

        fig.add_trace(
            go.Scatter(
                x=group["location_x"],
                y=group["location_y"],
                mode="markers",
                name=action,
                marker=dict(
                    size=9,
                    color=action_colors.get(action, "#718096"),
                    opacity=0.8,
                    line=dict(
                        color="#FFFFFF",
                        width=1
                    )
                ),
                text=player_names,
                customdata=group["defensive_action"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Action: %{customdata}<br>"
                    "Position: (%{x:.1f}, %{y:.1f})"
                    "<extra></extra>"
                )
            )
        )

    return fig

def display_progression_pitch(
    results,
    player=None,
    match_id=None
):

    st.subheader("Progression map")

    st.caption(
        "Visualization of progressive passes and carries "
        "by AS Monaco."
    )

    passes = results.get(
        "progressive_passes", {}
    ).get("events", pd.DataFrame())

    carries = results.get(
        "progressive_carries", {}
    ).get("events", pd.DataFrame())

    # Apply filters
    passes = filter_progression_events(
        passes,
        player=player,
        match_id=match_id
    )

    carries = filter_progression_events(
        carries,
        player=player,
        match_id=match_id
    )

    # Event counts
    st.write("Progressive passes:", len(passes))
    st.write("Progressive carries:", len(carries))

    # Create ONE pitch
    fig = create_progression_pitch()

    # Add progressive passes
    fig = add_progressive_passes_to_pitch(
        fig,
        passes
    )

    # Add progressive carries
    fig = add_progressive_carries_to_pitch(
        fig,
        carries
    )

    # Display ONE chart
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="progression_pitch_chart",
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )

    st.caption(
        f"Progressive passes: {len(passes):,} | "
        f"Progressive carries: {len(carries):,}"
    )

def filter_progression_events(
    events,
    player=None,
    match_id=None
):
    """
    Filter progression events by player and match.
    """

    if events is None or events.empty:
        return pd.DataFrame()

    filtered = events.copy()

    if player is not None:
        if "player_name" not in filtered.columns:
            return pd.DataFrame()

        filtered = filtered[
            filtered["player_name"] == player
        ]

    if match_id is not None:
        if "match_id" not in filtered.columns:
            return pd.DataFrame()

        filtered = filtered[
            filtered["match_id"].astype(str)
            == str(match_id)
        ]

    return filtered.reset_index(drop=True)

def filter_defensive_events(
    events,
    player=None,
    match_id=None
):
    if not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame()

    filtered = events.copy()

    if player is not None:
        if "player_name" not in filtered.columns:
            return pd.DataFrame()

        filtered = filtered[
            filtered["player_name"] == player
        ]

    if match_id is not None:
        if "match_id" not in filtered.columns:
            return pd.DataFrame()

        filtered = filtered[
            filtered["match_id"].astype(str) == str(match_id)
        ]

    return filtered.reset_index(drop=True)

def get_defensive_events(results):
    event_keys = [
        "interceptions",
        "tackles",
        "pressures",
        "recoveries",
        "blocks",
        "fouls"
    ]

    frames = []

    for key in event_keys:
        section = results.get(key, {})

        if not isinstance(section, dict):
            continue

        events = section.get("events", pd.DataFrame())

        if isinstance(events, pd.DataFrame) and not events.empty:
            events = events.copy()
            events["defensive_action"] = key.replace("_", " ").title()
            frames.append(events)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def display_progression_filters(results):

    passes = results.get(
        "progressive_passes", {}
    ).get("events", pd.DataFrame())

    carries = results.get(
        "progressive_carries", {}
    ).get("events", pd.DataFrame())

    combined = pd.concat(
        [passes, carries],
        ignore_index=True
    )

    if combined.empty:
        st.warning("No progression events available.")
        return None, None

    col1, col2 = st.columns(2)

    # Player filter
    players = (
        sorted(
            combined["player_name"]
            .dropna()
            .unique()
            .tolist()
        )
        if "player_name" in combined.columns
        else []
    )

    with col1:

        selected_player = st.selectbox(
            "Select player",
            options=["All players"] + players,
            index=0,
            key="progression_player_filter"
        )

    # Match filter
    matches = (
        sorted(
            combined["match_id"]
            .dropna()
            .unique()
            .tolist(),
            key=str,
            reverse=True
        )
        if "match_id" in combined.columns
        else []
    )

    match_options = ["All matches"] + matches

    with col2:

        selected_match = st.selectbox(
            "Select match",
            options=match_options,
            index=1 if matches else 0,
            format_func=lambda x: (
                "All matches" if x == "All matches"
                else f"Match {x}"
            ),
            key="progression_match_filter"
        )

    player = (
        None
        if selected_player == "All players"
        else selected_player
    )

    match_id = (
        None
        if selected_match == "All matches"
        else selected_match
    )

    return player, match_id

def display_final_third_analysis(
    results,
    player=None,
    match_id=None
):

    st.subheader("Final Third Entries")

    st.caption(
        "Analysis of passes entering the attacking final third."
    )

    # 1. Retrieve events
    section = results.get("final_third_passes", {})

    passes = section.get(
        "events",
        pd.DataFrame()
    )

    # 2. Safety check
    if not isinstance(passes, pd.DataFrame):
        passes = pd.DataFrame()

    # 3. Apply filters
    passes = filter_progression_events(
        passes,
        player=player,
        match_id=match_id
    )

    # 4. Display count
    st.write("Final third passes:", len(passes))

    # 5. Handle empty data
    if passes.empty:

        st.warning(
            "No passes entering the final third found."
        )

        return

    # 6. Create pitch
    fig = create_final_third_pitch()

    # 7. Add pass trajectories
    fig = add_final_third_passes_to_pitch(
        fig,
        passes
    )

    # 8. Display pitch
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="final_third_pitch_chart",
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )

def display_between_lines_analysis(
    results,
    player=None,
    match_id=None
):

    st.subheader("Between-the-Lines Receptions")

    st.caption(
        "Analysis of reception zones between the opposition lines."
    )

    receptions = results.get(
        "between_lines_receptions", {}
    ).get(
        "events",
        pd.DataFrame()
    )

    # Apply existing filters
    receptions = filter_progression_events(
        receptions,
        player=player,
        match_id=match_id
    )

    # Create pitch
    fig = create_between_lines_pitch()

    fig = add_between_lines_heatmap(
        fig,
        receptions
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="between_lines_pitch_chart",
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )

def display_defensive_filters(results):

    combined = get_defensive_events(results)

    if combined.empty:
        st.warning("No defensive events available.")
        return None, None

    col1, col2 = st.columns(2)

    with col1:

        players = (
            sorted(combined["player_name"].dropna().unique().tolist())
            if "player_name" in combined.columns
            else []
        )

        selected_player = st.selectbox(
            "Select player",
            options=["All players"] + players,
            index=0,
            key="defensive_player_filter"
        )

    with col2:

        matches = (
            sorted(
                combined["match_id"].dropna().unique().tolist(),
                key=str,
                reverse=True
            )
            if "match_id" in combined.columns
            else []
        )

        match_options = ["All matches"] + matches

        selected_match = st.selectbox(
            "Select match",
            options=match_options,
            index=1 if matches else 0,
            key="defensive_match_filter"
        )

    player = (
        None if selected_player == "All players"
        else selected_player
    )

    match_id = (
        None if selected_match == "All matches"
        else selected_match
    )

    return player, match_id

def display_defensive_kpis(results):

    configurations = [
        ("interceptions", "Interceptions"),
        ("tackles", "Tackles"),
        ("pressures", "Pressures"),
        ("recoveries", "Recoveries"),
        ("blocks", "Blocks"),
        ("fouls", "Fouls")
    ]

    st.subheader("Defensive Performance Indicators")

    columns = st.columns(3)

    for index, (key, label) in enumerate(configurations):

        section = results.get(key, {})
        summary = section.get("summary", {}) if isinstance(section, dict) else {}

        total = summary.get("total_events", 0)

        with columns[index % 3]:

            st.metric(
                label=label,
                value=f"{int(total):,}"
            )

def display_defensive_pitch(
    results,
    player=None,
    match_id=None
):

    st.subheader("Defensive Actions Map")

    st.caption(
        "Locations of interceptions, tackles, pressures, "
        "recoveries, blocks, and fouls."
    )

    events = get_defensive_events(results)

    events = filter_defensive_events(
        events,
        player=player,
        match_id=match_id
    )

    if events.empty:
        st.info("No defensive actions available for these filters.")
        return

    fig = create_defensive_pitch()

    fig = add_defensive_actions_to_pitch(
        fig,
        events
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="defensive_pitch_chart",
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )

    st.caption(
        f"Total defensive actions displayed: {len(events):,}"
    )

def display_recovery_zones(results):

    st.subheader("Ball Recovery Zones")

    st.caption(
        "Spatial distribution of ball recoveries."
    )

    section = results.get("recovery_zones", {})

    if not isinstance(section, dict):
        st.info("Recovery zone data unavailable.")
        return

    zones = section.get("zones", pd.DataFrame())
    summary = section.get("summary", {})

    if not isinstance(zones, pd.DataFrame) or zones.empty:
        st.info("No recovery zone data available.")
        return

    total = summary.get("total_recoveries", 0)

    st.metric(
        "Total recoveries",
        f"{int(total):,}"
    )

    if {"location_bucket", "recoveries", "percentage"}.issubset(zones.columns):

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=zones["location_bucket"].astype(str),
                y=zones["recoveries"],
                text=[
                    f"{value:.1f}%"
                    for value in zones["percentage"]
                ],
                textposition="outside",
                marker_color="#159B72",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Recoveries: %{y}<br>"
                    "%{text}"
                    "<extra></extra>"
                )
            )
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Pitch zone",
            yaxis_title="Recoveries",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="recovery_zones_chart"
        )

        st.dataframe(
            zones,
            use_container_width=True,
            hide_index=True
        )

def display_ppda(results):

    st.subheader("Pressing Intensity — PPDA")

    st.caption(
        "Opposition passes allowed per AS Monaco defensive action "
        "in the area defined by the model."
    )

    ppda_section = results.get("ppda", {})

    if not isinstance(ppda_section, dict):
        st.info("PPDA data unavailable.")
        return

    summary = ppda_section.get("summary", {})
    matches = ppda_section.get("matches", pd.DataFrame())

    col1, col2, col3 = st.columns(3)

    with col1:
        value = summary.get("ppda")

        st.metric(
            "Overall PPDA",
            f"{value:.2f}" if value is not None else "N/A"
        )

    with col2:
        st.metric(
            "Opponent Passes",
            f"{summary.get('total_opponent_passes', 0):,}"
        )

    with col3:
        st.metric(
            "Defensive Actions",
            f"{summary.get('total_defensive_actions', 0):,}"
        )

    if not isinstance(matches, pd.DataFrame) or matches.empty:
        st.info("No match-level PPDA data available.")
        return

    st.markdown("#### PPDA by match")

    if {"match_id", "ppda"}.issubset(matches.columns):

        chart_data = matches.dropna(subset=["ppda"]).copy()

        if not chart_data.empty:

            chart_data["match_id"] = (
                chart_data["match_id"].astype(str)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=chart_data["match_id"],
                    y=chart_data["ppda"],
                    marker_color="#3976B9",
                    text=chart_data["ppda"].map(
                        lambda value: f"{value:.2f}"
                    ),
                    textposition="outside",
                    hovertemplate=(
                        "Match: %{x}<br>"
                        "PPDA: %{y:.2f}"
                        "<extra></extra>"
                    )
                )
            )

            fig.update_layout(
                template="plotly_white",
                paper_bgcolor=COLORS["card"],
                plot_bgcolor=COLORS["card"],
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Match",
                yaxis_title="PPDA",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="ppda_match_chart"
            )

    st.dataframe(
        matches,
        use_container_width=True,
        hide_index=True
    )

# =========================================
# DATA LOADING
# =========================================

def _ensure_database_exists():
    """Raise a clear error when the configured database is missing."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_offensive_analysis():

    _ensure_database_exists()

    return run_offensive_analysis(
        database_path=DB_PATH,
        team_id=TEAM_ID
    )

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_progression_analysis():

    _ensure_database_exists()

    return run_progression_analysis(
        database_path=DB_PATH,
        team_id=TEAM_ID
    )

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_defensive_analysis():

    _ensure_database_exists()

    return run_defensive_analysis(
        database_path=DB_PATH,
        team_id=TEAM_ID
    )

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_collective_analysis():

    _ensure_database_exists()

    return run_collective_analysis(
        database_path=DB_PATH,
        team_id=TEAM_ID
    )

def display_defensive_analysis():

    st.markdown(
        '<div class="section-label">'
        'DEFENSIVE ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("Defensive Performance")

    st.caption(
        f"Defensive analysis | {TEAM_NAME}"
    )

    try:
        with st.spinner("Calculating defensive analysis..."):
            results = load_defensive_analysis()

    except Exception as error:
        st.error(
            f"Unable to load defensive analysis: {error}"
        )
        return

    # 01 — Defensive indicators
    st.markdown(
        '<div class="section-label">01 / Performance</div>',
        unsafe_allow_html=True
    )

    display_defensive_kpis(results)

    st.divider()

    # 02 — Filters
    st.markdown(
        '<div class="section-label">02 / Filters</div>',
        unsafe_allow_html=True
    )

    selected_player, selected_match = (
        display_defensive_filters(results)
    )

    st.divider()

    # 03 — Defensive actions map
    st.markdown(
        '<div class="section-label">03 / Defensive positioning</div>',
        unsafe_allow_html=True
    )

    display_defensive_pitch(
        results,
        player=selected_player,
        match_id=selected_match
    )

    st.divider()

    # 04 — Recovery zones
    st.markdown(
        '<div class="section-label">04 / Ball recovery</div>',
        unsafe_allow_html=True
    )

    display_recovery_zones(results)

    st.divider()

    # 05 — Pressing intensity
    st.markdown(
        '<div class="section-label">05 / Pressing intensity</div>',
        unsafe_allow_html=True
    )

    display_ppda(results)

    st.divider()

    st.caption(
        "FOOTBALL INTELLIGENCE PLATFORM | "
        "DEFENSIVE ANALYSIS ENGINE"
    )

    st.caption(
        f"Team ID: {TEAM_ID} | Database: {DB_PATH.name}"
    )

def display_progression_kpis(results):

    kpi_config = [
        {
            "key": "progressive_passes",
            "title": "Progressive Passes",
            "count": "total_progressive_passes",
            "percentage": "progressive_passes_percentage",
            "denominator": "total_passes"
        },
        {
            "key": "progressive_carries",
            "title": "Progressive Carries",
            "count": "total_progressive_carries",
            "percentage": "progressive_carries_percentage",
            "denominator": "total_carries"
        },
        {
            "key": "final_third_passes",
            "title": "Passes into Final Third",
            "count": "total_final_third_passes",
            "percentage": "final_third_passes_percentage",
            "denominator": "total_passes"
        },
        {
            "key": "line_breaking_passes",
            "title": "Potential Line-Breaking Passes",
            "count": "total_potential_line_breaks",
            "percentage": "potential_line_breaks_percentage",
            "denominator": "total_passes"
        },
        {
            "key": "between_lines_receptions",
            "title": "Between-the-Lines Receptions",
            "count": "total_between_lines_receptions",
            "percentage": "between_lines_percentage",
            "denominator": "total_receptions"
        }
    ]

    st.subheader("Progression Indicators")

    columns = st.columns(3)

    for index, config in enumerate(kpi_config):

        section = results.get(config["key"])

        if not isinstance(section, dict):
            st.error(f"Missing section: {config['key']}")
            continue

        summary = section.get("summary")

        if not isinstance(summary, dict):
            st.error(f"Missing summary: {config['key']}")
            continue

        count = summary.get(config["count"])
        percentage = summary.get(config["percentage"])
        denominator = summary.get(config["denominator"])

        if count is None:
            st.error(f"Missing KPI: {config['count']}")
            continue

        with columns[index % 3]:

            st.metric(
                label=config["title"],
                value=f"{int(count):,}",
                delta=(
                    f"{float(percentage):.2f}%"
                    if percentage is not None
                    else None
                )
            )

            if denominator is not None:
                st.caption(
                    f"Out of {int(denominator):,} events"
                )

def display_collective_kpis(results):

    configurations = [
        (
            "pass_network",
            "successful_passes",
            "Successful Passes",
            False
        ),
        (
            "pass_network",
            "players",
            "Players Involved",
            False
        ),
        (
            "chance_creation",
            "chance_creating_passes",
            "Chance-Creating Passes",
            False
        ),
        (
            "chance_creation",
            "xg_created",
            "xG Created",
            True
        ),
        (
            "attacking_patterns",
            "possessions",
            "Attacking Sequences",
            False
        ),
    ]

    st.subheader("Collective Performance Indicators")

    columns = st.columns(3)

    for index, (section_key, metric_key, label, is_xg) in enumerate(configurations):

        section = results.get(section_key, {})

        summary = (
            section.get("summary", {})
            if isinstance(section, dict)
            else {}
        )

        value = summary.get(metric_key, 0)

        if value is None:
            value = 0

        if is_xg:
            display_value = f"{float(value):.2f}"
        else:
            display_value = f"{value:,.0f}"

        with columns[index % 3]:

            st.metric(
                label=label,
                value=display_value
            )

def create_pass_network_pitch(network):

    fig = go.Figure()

    nodes = network.get("nodes", pd.DataFrame())
    edges = network.get("edges", pd.DataFrame())

    if nodes is None or nodes.empty:
        return add_football_pitch(fig, height=600)

    # Connections between players
    if edges is not None and not edges.empty:
        node_positions = (
            nodes.drop_duplicates("player")
            .set_index("player")[["x", "y"]]
            .to_dict("index")
        )

        for _, edge in edges.iterrows():
            source = edge["source"]
            target = edge["target"]

            if source not in node_positions or target not in node_positions:
                continue

            start = node_positions[source]
            end = node_positions[target]
            passes_count = edge.get("passes", 0)

            fig.add_trace(
                go.Scatter(
                    x=[start["x"], end["x"]],
                    y=[start["y"], end["y"]],
                    mode="lines",
                    line=dict(
                        color="#3976B9",
                        width=max(1, min(7, 1 + float(passes_count) / 3)),
                    ),
                    opacity=0.55,
                    hovertemplate=(
                        f"<b>{source}</b> → <b>{target}</b><br>"
                        f"Passes: {passes_count}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    # Player positions
    custom_columns = ["passes", "receptions"]
    customdata = (
        nodes[custom_columns].to_numpy()
        if set(custom_columns).issubset(nodes.columns)
        else None
    )

    fig.add_trace(
        go.Scatter(
            x=nodes["x"],
            y=nodes["y"],
            mode="markers+text",
            text=nodes["player"],
            textposition="top center",
            marker=dict(
                size=18,
                color="#159B72",
                line=dict(color="#FFFFFF", width=2),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Passes: %{customdata[0]}<br>"
                "Receptions: %{customdata[1]}<extra></extra>"
            ) if customdata is not None else "<b>%{text}</b><extra></extra>",
            showlegend=False,
        )
    )

    # The shared pitch is used for all spatial maps.
    return add_football_pitch(
        fig,
        height=600,
        x_range=(0, 120),
        y_range=(0, 80),
    )

def display_pass_network_filters(results):

    network = results.get("pass_network", {})

    nodes = network.get("nodes", pd.DataFrame())
    edges = network.get("edges", pd.DataFrame())

    if not isinstance(nodes, pd.DataFrame) or nodes.empty:
        st.warning("No nodes available.")
        return None, None

    col1, col2 = st.columns(2)

    with col1:

        players = (
            sorted(nodes["player"].dropna().unique().tolist())
            if "player" in nodes.columns
            else []
        )

        selected_player = st.selectbox(
            "Select player",
            ["All players"] + players,
            key="pass_network_player_filter"
        )

    with col2:

        matches = (
            sorted(
                nodes["match_id"].dropna().unique().tolist(),
                key=str,
                reverse=True
            )
            if "match_id" in nodes.columns
            else []
        )

        selected_match = st.selectbox(
            "Select match",
            ["All matches"] + matches,
            index=1 if matches else 0,
            format_func=lambda x: (
                "All matches" if x == "All matches"
                else f"Match {x}"
            ),
            key="pass_network_match_filter"
        )

    player = (
        None if selected_player == "All players"
        else selected_player
    )

    match_id = (
        None if selected_match == "All matches"
        else selected_match
    )

    return player, match_id

def filter_pass_network(
    dataframe,
    player=None,
    match_id=None
):

    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return pd.DataFrame()

    filtered = dataframe.copy()

    if player is not None:

        if "player" not in filtered.columns:
            return pd.DataFrame()

        filtered = filtered[
            filtered["player"] == player
        ]

    if match_id is not None:

        if "match_id" not in filtered.columns:
            return pd.DataFrame()

        filtered = filtered[
            filtered["match_id"].astype(str)
            == str(match_id)
        ]

    return filtered.reset_index(drop=True)

def display_pass_network(
    results,
    player=None,
    match_id=None
):

    st.subheader("Pass Network")

    st.caption(
        "Connections between players based on completed passes."
    )

    network = results.get("pass_network", {})

    if not isinstance(network, dict):
        st.info("Pass network unavailable.")
        return

    summary = network.get("summary", {})
    nodes = network.get("nodes", pd.DataFrame())
    edges = network.get("edges", pd.DataFrame())

    # Apply match filter to nodes and edges
    nodes = filter_pass_network(
        nodes,
        match_id=match_id
    )

    edges = filter_pass_network(
        edges,
        match_id=match_id
    )

    # Apply player filter to edges first
    if player is not None:

        edges = edges[
            (edges["source"] == player)
            | (edges["target"] == player)
        ].copy()

        # Keep all players involved in the filtered connections
        connected_players = set(edges["source"]) | set(edges["target"])

        nodes = nodes[
            nodes["player"].isin(connected_players)
        ].copy()

    if match_id is not None:
        edges = filter_pass_network(
            edges,
            match_id=match_id
        )

    if player is not None:

        if not edges.empty:
            edges = edges[
                (edges["source"] == player)
                | (edges["target"] == player)
            ].copy()

    if nodes.empty:
        st.info("No player connections available for these filters.")
        return

    # Filtered network
    filtered_network = {
        "nodes": nodes,
        "edges": edges
    }

    # KPIs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Successful Passes",
            f"{int(edges['passes'].sum()):,}"
            if "passes" in edges.columns else "0"
        )

    with col2:
        st.metric(
            "Players",
            f"{nodes['player'].nunique():,}"
        )

    with col3:
        st.metric(
            "Connections",
            f"{len(edges):,}"
        )

    # Pass network pitch
    fig = create_pass_network_pitch(filtered_network)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="collective_pass_network"
    )

    # Strongest connections
    if not edges.empty:

        st.markdown("#### Strongest passing connections")

        st.dataframe(
            edges.sort_values(
                "passes",
                ascending=False
            ).head(15),
            use_container_width=True,
            hide_index=True
        )

def display_player_heatmaps(results):

    st.subheader("Player Activity Heatmaps")

    st.caption(
        "Spatial distribution of actions performed by each player."
    )

    heatmaps = results.get("heatmaps", {})

    if not isinstance(heatmaps, dict):
        st.info("Heatmap data unavailable.")
        return

    events = heatmaps.get("events", pd.DataFrame())
    zones = heatmaps.get("zones", pd.DataFrame())

    if events.empty or zones.empty:
        st.info("No player activity data available.")
        return

    players = sorted(
        events["player_name"].dropna().unique().tolist()
    )

    if not players:
        st.info("No players available.")
        return

    selected_player = st.selectbox(
        "Select player",
        players,
        key="collective_heatmap_player"
    )

    player_events = events[
        events["player_name"] == selected_player
    ].copy()

    player_zones = zones[
        zones["player_name"] == selected_player
    ].copy()

    if player_events.empty:
        st.info("No activity for this player.")
        return

    # Clean coordinates
    player_events["location_x"] = pd.to_numeric(
        player_events["location_x"],
        errors="coerce"
    )

    player_events["location_y"] = pd.to_numeric(
        player_events["location_y"],
        errors="coerce"
    )

    player_events = player_events.dropna(
        subset=["location_x", "location_y"]
    )

    if player_events.empty:
        st.info("No valid player coordinates available.")
        return

    fig = go.Figure()

    # Activity heatmap
    fig.add_trace(
        go.Histogram2d(
            x=player_events["location_x"],
            y=player_events["location_y"],
            xbins=dict(start=0, end=120, size=5),
            ybins=dict(start=0, end=80, size=5),
            colorscale=[
                [0, "rgba(46,125,50,0)"],
                [0.01, "rgba(245,203,120,0.45)"],
                [0.35, "rgba(240,164,58,0.75)"],
                [1, "rgba(192,57,43,0.95)"]
            ],
            colorbar=dict(title="Actions"),
            hovertemplate=(
                "X: %{x}<br>"
                "Y: %{y}<br>"
                "Actions: %{z}"
                "<extra></extra>"
            )
        )
    )

    # Add the same pitch used in the other visualizations.
    add_football_pitch(
        fig,
        height=600,
        x_range=(0, 120),
        y_range=(0, 80),
        invert_y=True,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True
        },
        key="collective_player_heatmap"
    )

    st.metric(
        "Player Actions",
        f"{len(player_events):,}"
    )

    if not player_zones.empty:
        st.dataframe(
            player_zones.sort_values(
                "events",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

def display_space_occupation(results):

    st.subheader("Space Occupation")

    st.caption(
        "Distribution of actions across longitudinal zones "
        "and the five pitch lanes."
    )

    if not isinstance(results, dict):
        st.info("Space occupation data unavailable.")
        return

    occupation = results.get("space_occupation", {})
    if not isinstance(occupation, dict):
        st.info("Space occupation data unavailable.")
        return

    summary = occupation.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    zones = occupation.get("zones", pd.DataFrame())

    if not isinstance(zones, pd.DataFrame) or zones.empty:
        st.info("No space occupation data available.")
        return

    st.metric(
        "Total Actions",
        f"{summary.get('events', 0):,}"
    )

    # Columns actually returned by global_analysis.py
    required = {
        "longitudinal_zone",
        "lateral_zone",
        "events"
    }

    if not required.issubset(zones.columns):
        st.warning(
            f"Unexpected space occupation data structure. "
            f"Columns found: {list(zones.columns)}"
        )
        st.dataframe(zones, use_container_width=True)
        return

    # Pitch zone order
    x_order = ["Defensive", "Middle", "Attacking"]

    y_order = [
        "Couloir haut",
        "Demi-espace haut",
        "Axe",
        "Demi-espace bas",
        "Couloir bas"
    ]

    # Conversion en tableau 3 x 5
    pivot = zones.pivot_table(
        index="lateral_zone",
        columns="longitudinal_zone",
        values="events",
        aggfunc="sum",
        fill_value=0,
        observed=False
    )

    pivot = pivot.reindex(
        index=y_order,
        columns=x_order,
        fill_value=0
    )

    # Center coordinates of the 15 zones
    x_centers = [20, 60, 100]
    y_centers = [8, 24, 40, 56, 72]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=x_centers,
            y=y_centers,
            colorscale="YlGn",
            showscale=True,
            colorbar=dict(
                title="Actions"
            ),
            text=pivot.values,
            texttemplate="%{text}",
            textfont=dict(
                color="#172B4D",
                size=14
            ),
            hovertemplate=(
                "Actions : %{z}<br>"
                "Position : (%{x}, %{y})"
                "<extra></extra>"
            )
        )
    )

    # Add the shared pitch
    add_football_pitch(
        fig,
        invert_y=True,
        height=500,
        show_legend=False
    )

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        xaxis=dict(
            range=[-2, 122],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[82, -2],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="collective_space_occupation"
    )

    # Detailed table kept below the pitch
    st.markdown("#### Space occupation details")

    st.dataframe(
        zones.sort_values(
            "events",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

def display_chance_creation(results):

    st.subheader("Chance Creation")

    st.caption(
        "Shot-assisted passes and their xG contribution."
    )

    chance = results.get("chance_creation", {})

    if not isinstance(chance, dict):
        st.info("Chance creation data unavailable.")
        return

    summary = chance.get("summary", {})
    passes = chance.get("passes", pd.DataFrame())
    zones = chance.get("zones", pd.DataFrame())
    shots = chance.get("shots", pd.DataFrame())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Chance-Creating Passes",
            f"{summary.get('chance_creating_passes', 0):,}"
        )

    with col2:
        st.metric(
            "Linked Shots",
            f"{summary.get('linked_shots', 0):,}"
        )

    with col3:
        st.metric(
            "xG Created",
            f"{summary.get('xg_created', 0):.2f}"
        )

    if not zones.empty:

        st.markdown("#### Chance creation zones")

        fig = go.Figure(
            go.Bar(
                x=zones["x_zone"].astype(str)
                    + " / "
                    + zones["y_zone"].astype(str),
                y=zones["chance_passes"],
                marker_color="#159B72",
                text=zones["chance_passes"],
                textposition="outside",
                hovertemplate=(
                    "%{x}<br>"
                    "Chance passes: %{y}"
                    "<extra></extra>"
                )
            )
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            height=380,
            xaxis_title="Pitch zone",
            yaxis_title="Chance-creating passes"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="collective_chance_zones"
        )

    if not passes.empty:

        with st.expander("View chance-creating passes"):
            st.dataframe(
                passes,
                use_container_width=True,
                hide_index=True
            )

    if not shots.empty:

        with st.expander("View linked shots"):
            st.dataframe(
                shots,
                use_container_width=True,
                hide_index=True
            )

def display_attacking_patterns(results):

    st.subheader("Attacking Patterns")

    st.caption(
        "Attacking sequences and passing combinations "
        "between AS Monaco players."
    )

    patterns = results.get("attacking_patterns", {})

    if not isinstance(patterns, dict):
        st.info("Attacking patterns unavailable.")
        return

    summary = patterns.get("summary", {})
    combinations = patterns.get(
        "combinations",
        pd.DataFrame()
    )
    sequences = patterns.get(
        "sequences",
        pd.DataFrame()
    )

    if not combinations.empty:

        st.markdown("#### Most frequent combinations")

        st.dataframe(
            combinations.sort_values(
                "passes",
                ascending=False
            ).head(15),
            use_container_width=True,
            hide_index=True
        )

    if not sequences.empty:

        st.markdown("#### Attacking sequences")

        st.dataframe(
            sequences.tail(20),
            use_container_width=True,
            hide_index=True
        )

def display_collective_analysis():

    st.markdown(
        '<div class="section-label">'
        'COLLECTIVE ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("Collective Performance")

    st.caption(
        f"Collective analysis | {TEAM_NAME}"
    )

    try:
        with st.spinner(
            "Calculating collective analysis..."
        ):
            results = load_collective_analysis()

    except Exception as error:
        st.error(
            f"Unable to load collective analysis: {error}"
        )
        return

    # 01 — KPIs
    st.markdown(
        '<div class="section-label">01 / Overview</div>',
        unsafe_allow_html=True
    )

    display_collective_kpis(results)

    st.divider()

    # 02 — Passing network
    st.markdown(
        '<div class="section-label">02 / Passing connections</div>',
        unsafe_allow_html=True
    )

    selected_player, selected_match = (
        display_pass_network_filters(results)
    )

    display_pass_network(
        results,
        player=selected_player,
        match_id=selected_match
    )

    st.divider()

    # 03 — Player heatmaps
    st.markdown(
        '<div class="section-label">03 / Player activity</div>',
        unsafe_allow_html=True
    )

    display_player_heatmaps(results)

    st.divider()

    # 04 — Space occupation
    st.markdown(
        '<div class="section-label">04 / Space occupation</div>',
        unsafe_allow_html=True
    )

    display_space_occupation(results)

    st.divider()

    # 05 — Chance creation
    st.markdown(
        '<div class="section-label">05 / Chance creation</div>',
        unsafe_allow_html=True
    )

    display_chance_creation(results)

    st.divider()

    # 06 — Attacking patterns
    st.markdown(
        '<div class="section-label">06 / Attacking patterns</div>',
        unsafe_allow_html=True
    )

    display_attacking_patterns(results)

    st.divider()

    st.caption(
        "FOOTBALL INTELLIGENCE PLATFORM | "
        "COLLECTIVE ANALYSIS ENGINE"
    )

    st.caption(
        f"Team ID: {TEAM_ID} | Database: {DB_PATH.name}"
    )

def display_progression_analysis():

    st.markdown(
        '<div class="section-label">'
        'PROGRESSION ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("Ball Progression")

    st.caption(
        f"Progression analysis | {TEAM_NAME}"
    )

    try:

        with st.spinner(
            "Calculating progression analysis..."
        ):

            results = load_progression_analysis()

    except Exception as error:

        st.error(
            f"Unable to load progression analysis: {error}"
        )

        return

    display_progression_kpis(results)

    st.divider()

    selected_player, selected_match = (
        display_progression_filters(results)
    )

    st.divider()

    display_progression_pitch(
        results,
        player=selected_player,
        match_id=selected_match
    )

    st.divider()

    display_final_third_analysis(
        results,
        player=selected_player,
        match_id=selected_match
    )

    st.divider()

    display_between_lines_analysis(
        results,
        player=selected_player,
        match_id=selected_match
    )

# =========================================
# NAVIGATION
# =========================================

def display_navigation():

    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    if st.session_state.current_page != "home":

        if st.button("← Back to Home"):
            st.session_state.current_page = "home"
            st.rerun()

def display_home():

    st.markdown(
        f"""
        <div class="dashboard-header">
            <div class="dashboard-subtitle">
                FOOTBALL INTELLIGENCE PLATFORM
            </div>
            <h1>AS Monaco</h1>
            <p class="dashboard-description">
                Select an analysis to explore.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:

        st.subheader("⚽ Offensive")

        st.caption(
            "Shooting performance, xG, shot distribution "
            "and chance creation."
        )

        if st.button(
            "Open Offensive Analysis",
            use_container_width=True
        ):

            st.session_state.current_page = "offensive"
            st.rerun()

    with col2:

        st.subheader("📈 Progression")

        st.caption(
            "Progressive passes, carries, final third entries "
            "and between-the-lines receptions."
        )

        if st.button(
            "Open Progression Analysis",
            use_container_width=True
        ):

            st.session_state.current_page = "progression"
            st.rerun()

    with col3:

        st.subheader("🛡️ Defensive")

        st.caption(
            "Interceptions, tackles, pressures, recoveries, "
            "defensive positioning and pressing intensity."
        )

        if st.button(
            "Open Defensive Analysis",
            use_container_width=True
        ):

            st.session_state.current_page = "defensive"
            st.rerun()

    with col4:

        st.subheader("🔗 Collective")

        st.caption(
            "Pass networks, player heatmaps, space occupation, "
            "attack creations and patterns."
        )

        if st.button(
            "Open Collective Analysis",
            use_container_width=True
        ):

            st.session_state.current_page = "collective"
            st.rerun()

# =========================================
# APPLICATION NAVIGATION
# =========================================

display_navigation()

if st.session_state.current_page == "home":

    display_home()

elif st.session_state.current_page == "offensive":

    # =========================================
    # OFFENSIVE ANALYSIS
    # =========================================

    try:

        with st.spinner("Calculating offensive analysis..."):
            results = load_offensive_analysis()

    except Exception as error:

        st.error(
            f"Unable to load offensive analysis: {error}"
        )

        st.stop()

    # =========================================
    # 1. OFFENSIVE INDICATORS
    # =========================================

    st.markdown(
        '<div class="section-label">01 / Performance</div>',
        unsafe_allow_html=True
    )

    st.subheader("Offensive indicators")

    metrics = results.get("metrics", {})

    if metrics:

        metric_items = list(metrics.items())

        columns = st.columns(min(len(metric_items), 4))

        for index, (metric, value) in enumerate(metric_items):

            with columns[index % len(columns)]:

                if isinstance(value, Real):

                    if pd.isna(value):
                        display_value = "N/A"

                    elif metric == "conversion_rate":
                        display_value = f"{value:.1f}%"

                    elif isinstance(value, int):
                        display_value = f"{value:,}"

                    else:
                        display_value = f"{value:.3f}"

                else:
                    display_value = str(value)

                st.metric(
                    label=metric.replace("_", " ").upper(),
                    value=display_value
                )

    else:

        st.info("No offensive indicators available.")

    st.divider()

    # =========================================
    # 2. SHOT DISTRIBUTION
    # =========================================

    st.markdown(
        '<div class="section-label">02 / Shot analysis</div>',
        unsafe_allow_html=True
    )

    st.subheader("Shot distribution by zone")

    zones = results.get("zones", pd.DataFrame())

    if isinstance(zones, pd.DataFrame) and not zones.empty:

        st.subheader("Shot zones on football pitch")

        pitch_fig = create_shot_zone_pitch(zones)

        st.plotly_chart(
            pitch_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )

        display_shot_zone_legend(zones)

    else:

        st.info("No shot distribution data available.")

    # =========================================
    # 3. ASSIST ANALYSIS
    # =========================================

    st.markdown(
        '<div class="section-label">03 / Chance creation</div>',
        unsafe_allow_html=True
    )

    st.subheader("Passes associated with shots")

    assists = results.get("assists", pd.DataFrame())

    if isinstance(assists, pd.DataFrame) and not assists.empty:

        st.dataframe(
            assists,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No shot-assisted pass data available.")

    # =========================================
    # FOOTER
    # =========================================

    st.divider()

    st.caption(
        "FOOTBALL INTELLIGENCE PLATFORM | "
        "OFFENSIVE ANALYSIS ENGINE"
    )

    st.caption(
        f"Team ID: {TEAM_ID} | Database: {DB_PATH.name}"
    )

elif st.session_state.current_page == "progression":

    display_progression_analysis()

elif st.session_state.current_page == "defensive":

    display_defensive_analysis()

elif st.session_state.current_page == "collective":

    display_collective_analysis()
