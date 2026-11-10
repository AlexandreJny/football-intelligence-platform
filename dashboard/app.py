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

from src.global_analysis import run_offensive_analysis


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


# =========================================
# ANALYSIS
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