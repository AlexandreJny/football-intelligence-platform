
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = ROOT_DIR / "database" / "football.duckdb"

APP_TITLE = "Football Intelligence Platform"

# Shared application settings
CACHE_TTL_SECONDS = 300

# StatsBomb pitch coordinate system
PITCH_LENGTH = 120
PITCH_WIDTH = 80

TEAM_ID = 129
TEAM_NAME = "AS Monaco"

COLORS = {
    "background": "#F3F6FA",
    "card": "#FFFFFF",
    "primary": "#159B72",
    "primary_light": "#E3F5EE",
    "secondary": "#3976B9",
    "accent": "#E6A23C",
    "text": "#172B45",
    "muted": "#718096",
    "border": "#DFE6EE",
    "chart_grid": "#E8EDF3",
    "danger": "#D95D5D",
}
