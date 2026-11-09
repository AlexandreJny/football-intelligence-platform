
import pandas as pd
import pytest


@pytest.fixture
def sample_events():
    """Jeu de données StatsBomb fictif."""

    return pd.DataFrame({
        "match_id": [1001, 1001, 1002],
        "id": ["event_1", "event_2", "event_3"],
        "index": [1, 2, 1],
        "period": [1, 1, 1],
        "timestamp": [
            "00:01:10.000",
            "00:02:20.000",
            "00:01:30.000"
        ],
        "event_type_name": [
            "Pass",
            "Shot",
            "Pass"
        ],
        "team_id": [10, 10, 20],
        "team_name": [
            "AS Monaco",
            "AS Monaco",
            "PSG"
        ],
        "player_id": [101, 102, 201],
        "player_name": [
            "Joueur A",
            "Joueur B",
            "Joueur C"
        ],
        "player_position_name": [
            "Midfielder",
            "Forward",
            "Defender"
        ],
        "location_x": [50.0, 100.0, 30.0],
        "location_y": [25.0, 40.0, 60.0]
    })


@pytest.fixture
def sample_csv(tmp_path):
    """Crée un fichier CSV temporaire."""

    df = pd.DataFrame({
        "match_id": [1001, 1002],
        "event_id": ["event_1", "event_2"]
    })

    file_path = tmp_path / "events.csv"
    df.to_csv(file_path, index=False)

    return file_path
