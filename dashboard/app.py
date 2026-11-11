import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from numbers import Real

# =========================================
# PROJECT PATH
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================
# IMPORTS
# =========================================

from config import (
    DB_PATH,
    APP_TITLE,
    COLORS,
    TEAM_ID,
    TEAM_NAME
)

from src.global_analysis import (
    run_offensive_analysis,
    run_progression_analysis
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
            "description": "Zone extérieure à la surface de réparation."
        },
        "central outside area": {
            "color": "#F5CB78",
            "description": "Couloir central devant la surface."
        },
        "penalty area": {
            "color": "#3976B9",
            "description": "Ensemble de la surface de réparation."
        },
        "central penalty area": {
            "color": "#159B72",
            "description": "Zone centrale à l'intérieur de la surface."
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
                "description": "Zone de tir."
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

def create_shot_zone_pitch(zones_df=None):

    fig = go.Figure()

    pitch_line = "#263746"

    zone_colors = {
        "outside penalty area": "#F0A43A",
        "central outside area": "#F5CB78",
        "penalty area": "#3976B9",
        "central penalty area": "#159B72"
    }

    # Retrieve shot counts
    zone_counts = {}

    if zones_df is not None and not zones_df.empty:

        zone_counts = {
            str(row["shot_zone"]).strip().casefold():
                int(row["total_shots"])
            for _, row in zones_df.iterrows()
        }

    # Pitch background
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=120, y1=80,
        fillcolor="#FFFFFF",
        line=dict(color=pitch_line, width=2),
        layer="below"
    )

    # Four non-overlapping zones
    zones = [

        {
            "name": "Outside penalty area",
            "x0": 80, "x1": 120,
            "y0": 0, "y1": 18,
            "color": zone_colors["outside penalty area"],
            "label_y": 69
        },

        {
            "name": "Outside penalty area",
            "x0": 80, "x1": 120,
            "y0": 62, "y1": 80,
            "color": zone_colors["outside penalty area"],
            "label_y": 69
        },

        {
            "name": "Central outside area",
            "x0": 80, "x1": 102,
            "y0": 18, "y1": 62,
            "color": zone_colors["central outside area"],
            "label_y": 40
        },

        {
            "name": "Penalty area",
            "x0": 102, "x1": 120,
            "y0": 18, "y1": 62,
            "color": zone_colors["penalty area"],
            "label_y": 24
        },

        {
            "name": "Central penalty area",
            "x0": 102, "x1": 120,
            "y0": 30, "y1": 50,
            "color": zone_colors["central penalty area"],
            "label_y": 40
        }
    ]

    # Draw zones and annotations
    for zone in zones:

        key = zone["name"].casefold()

        count = zone_counts.get(key, 0)

        center_x = (
            zone["x0"] + zone["x1"]
        ) / 2

        fig.add_shape(
            type="rect",
            x0=zone["x0"],
            y0=zone["y0"],
            x1=zone["x1"],
            y1=zone["y1"],
            fillcolor=zone["color"],
            opacity=0.75,
            line=dict(
                color="#FFFFFF",
                width=1
            ),
            layer="below"
        )

        fig.add_annotation(
            x=center_x,
            y=zone["label_y"],
            text=f"<b>{count}</b><br>SHOTS",
            showarrow=False,
            font=dict(
                size=14,
                color="#263746"
            ),
            bgcolor="rgba(255,255,255,0.88)",
            borderpad=5,
            align="center"
        )

    # Pitch markings
    fig.add_shape(
        type="line",
        x0=60, y0=0,
        x1=60, y1=80,
        line=dict(color=pitch_line, width=2)
    )

    fig.add_shape(
        type="circle",
        x0=50, y0=30,
        x1=70, y1=50,
        line=dict(color=pitch_line, width=2),
        fillcolor="rgba(0,0,0,0)"
    )

    # Penalty areas
    for x0, x1 in [(0, 18), (102, 120)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=18,
            x1=x1, y1=62,
            line=dict(color=pitch_line, width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    # Six-yard boxes
    for x0, x1 in [(0, 6), (114, 120)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=30,
            x1=x1, y1=50,
            line=dict(color=pitch_line, width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    # Spots
    fig.add_trace(
        go.Scatter(
            x=[12, 108, 60],
            y=[40, 40, 40],
            mode="markers",
            marker=dict(
                size=5,
                color=pitch_line
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # Goals
    for x0, x1 in [(-2, 0), (120, 122)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=36,
            x1=x1, y1=44,
            line=dict(color=pitch_line, width=3),
            fillcolor="rgba(0,0,0,0)"
        )

    # Layout
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor="#FFFFFF",
        height=600,
        margin=dict(l=5, r=5, t=10, b=10),
        showlegend=False
    )

    fig.update_xaxes(
        range=[-4, 124],
        visible=False,
        fixedrange=True
    )

    fig.update_yaxes(
        range=[-2, 82],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True
    )

    return fig




def create_progression_pitch():

    fig = go.Figure()

    pitch_color = "#263746"

    # Pitch background
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=120, y1=80,
        fillcolor="#FFFFFF",
        line=dict(color=pitch_color, width=2),
        layer="below"
    )

    # Halfway line
    fig.add_shape(
        type="line",
        x0=60, y0=0,
        x1=60, y1=80,
        line=dict(color=pitch_color, width=2)
    )

    # Center circle
    fig.add_shape(
        type="circle",
        x0=50, y0=30,
        x1=70, y1=50,
        line=dict(color=pitch_color, width=2),
        fillcolor="rgba(0,0,0,0)"
    )

    # Penalty areas
    for x0, x1 in [(0, 18), (102, 120)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=18,
            x1=x1, y1=62,
            line=dict(color=pitch_color, width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    # Six-yard boxes
    for x0, x1 in [(0, 6), (114, 120)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=30,
            x1=x1, y1=50,
            line=dict(color=pitch_color, width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    # Penalty spots
    fig.add_trace(
        go.Scatter(
            x=[12, 108, 60],
            y=[40, 40, 40],
            mode="markers",
            marker=dict(
                size=5,
                color=pitch_color
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # Layout
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor="#FFFFFF",
        height=600,
        margin=dict(l=5, r=5, t=10, b=10),
        showlegend=False
    )

    fig.update_xaxes(
        range=[-2, 122],
        visible=False,
        fixedrange=True
    )

    fig.update_yaxes(
        range=[-2, 82],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True
    )

    return fig



def create_final_third_pitch():

    fig = go.Figure()

    pitch_color = "#263746"

    # Pitch background
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=120, y1=80,
        fillcolor="#FFFFFF",
        line=dict(color=pitch_color, width=2),
        layer="below"
    )

    # Highlight final third
    fig.add_shape(
        type="rect",
        x0=80, y0=0,
        x1=120, y1=80,
        fillcolor="rgba(21,155,114,0.13)",
        line=dict(
            color="#159B72",
            width=2,
            dash="dash"
        ),
        layer="below"
    )

    # Halfway line
    fig.add_shape(
        type="line",
        x0=60, y0=0,
        x1=60, y1=80,
        line=dict(color=pitch_color, width=2)
    )

    # Center circle
    fig.add_shape(
        type="circle",
        x0=50, y0=30,
        x1=70, y1=50,
        line=dict(color=pitch_color, width=2),
        fillcolor="rgba(0,0,0,0)"
    )

    # Penalty areas
    for x0, x1 in [(0, 18), (102, 120)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=18,
            x1=x1, y1=62,
            line=dict(color=pitch_color, width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    # Six-yard boxes
    for x0, x1 in [(0, 6), (114, 120)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=30,
            x1=x1, y1=50,
            line=dict(color=pitch_color, width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    # Layout
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor="#FFFFFF",
        height=550,
        margin=dict(l=5, r=5, t=10, b=10),
        showlegend=False
    )

    fig.update_xaxes(
        range=[-2, 122],
        visible=False,
        fixedrange=True
    )

    fig.update_yaxes(
        range=[-2, 82],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True
    )

    return fig



def create_between_lines_pitch():

    fig = go.Figure()

    pitch_color = "#263746"

    # Pitch background
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=120, y1=80,
        fillcolor="#FFFFFF",
        line=dict(color=pitch_color, width=2),
        layer="below"
    )

    # Halfway line
    fig.add_shape(
        type="line",
        x0=60, y0=0,
        x1=60, y1=80,
        line=dict(color=pitch_color, width=2)
    )

    # Center circle
    fig.add_shape(
        type="circle",
        x0=50, y0=30,
        x1=70, y1=50,
        line=dict(color=pitch_color, width=2),
        fillcolor="rgba(0,0,0,0)"
    )

    # Penalty areas
    for x0, x1 in [(0, 18), (102, 120)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=18,
            x1=x1, y1=62,
            line=dict(color=pitch_color, width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    # Six-yard boxes
    for x0, x1 in [(0, 6), (114, 120)]:

        fig.add_shape(
            type="rect",
            x0=x0, y0=30,
            x1=x1, y1=50,
            line=dict(color=pitch_color, width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor="#FFFFFF",
        height=550,
        margin=dict(l=5, r=5, t=10, b=10),
        showlegend=False
    )

    fig.update_xaxes(
        range=[-2, 122],
        visible=False,
        fixedrange=True
    )

    fig.update_yaxes(
        range=[-2, 82],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True
    )

    return fig




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




def display_progression_pitch(
    results,
    player=None,
    match_id=None
):

    st.subheader("Progression map")

    st.caption(
        "Visualisation des passes et conduites progressives "
        "de l'AS Monaco."
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
    players = sorted(
        combined["player_name"]
        .dropna()
        .unique()
        .tolist()
    )

    with col1:

        selected_player = st.selectbox(
            "Select player",
            options=["All players"] + players
        )

    # Match filter
    matches = sorted(
        combined["match_id"]
        .dropna()
        .unique()
        .tolist()
    )

    with col2:

        selected_match = st.selectbox(
            "Select match",
            options=["All matches"] + matches
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
        "Analyse des passes arrivant dans le dernier tiers offensif."
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
        "Analyse des zones de réception entre les lignes adverses."
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


# =========================================
# DATA LOADING
# =========================================

@st.cache_data(ttl=300)
def load_offensive_analysis():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Base de données introuvable : {DB_PATH}"
        )

    return run_offensive_analysis(
        database_path=DB_PATH,
        team_id=TEAM_ID
    )



@st.cache_data(ttl=300)
def load_progression_analysis():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Base de données introuvable : {DB_PATH}"
        )

    return run_progression_analysis(
        database_path=DB_PATH,
        team_id=TEAM_ID
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
# HEADER
# =========================================

st.markdown(
    """
    <style>
    .dashboard-header {
        background: #FFFFFF;
        border: 1px solid #DFE6EE;
        border-radius: 14px;
        padding: 25px 30px;
        margin-bottom: 25px;
    }

    .dashboard-subtitle {
        color: #718096;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    .team-name {
        color: #159B72;
        font-size: 18px;
        font-weight: 700;
    }

    .dashboard-description {
        color: #718096;
        font-size: 14px;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="dashboard-header">
        <div class="dashboard-subtitle">
            FOOTBALL INTELLIGENCE PLATFORM
        </div>
        <h1>Offensive Analysis</h1>
        <div class="team-name">{TEAM_NAME}</div>
        <p class="dashboard-description">
            Analyse des performances offensives et des situations
            de création d'occasions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


def display_navigation():

    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    if st.session_state.current_page != "home":

        if st.button("← Retour à l'accueil"):
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

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⚽ Offensive Analysis")

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

        st.subheader("📈 Progression Analysis")

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

        with st.spinner("Calcul de l'analyse offensive..."):
            results = load_offensive_analysis()

    except Exception as error:

        st.error(
            f"Impossible de charger l'analyse offensive : {error}"
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

        st.info("Aucun indicateur offensif disponible.")


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

        st.info("Aucune donnée de répartition des tirs disponible.")


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

        st.info("Aucune donnée sur les passes associées aux tirs.")


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

    pass


elif st.session_state.current_page == "progression":

    # =========================================
    # PROGRESSION ANALYSIS
    # =========================================

    display_progression_analysis()
