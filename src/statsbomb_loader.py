
from pathlib import Path
import pandas as pd


def load_csv(file_path):
    """
    Load a StatsBomb CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded event data.
    """

    # Convertir le chemin en objet Path
    file_path = Path(file_path)

    # Vérifier que le fichier existe
    if not file_path.exists():
        raise FileNotFoundError(
            f"Le fichier est introuvable : {file_path}"
        )

    # Vérifier qu'il s'agit bien d'un CSV
    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Le fichier doit être au format CSV : {file_path}"
        )

    # Charger le fichier
    try:
        df = pd.read_csv(file_path)

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"Le fichier CSV est vide : {file_path}"
        ) from error

    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
        raise RuntimeError(
            f"Erreur lors de la lecture du fichier : {file_path}"
        ) from error

    return df



def load_season(data_directory):
    """
    Load and combine all CSV files from the AS Monaco season.

    Parameters
    ----------
    data_directory : str or Path
        Directory containing the CSV files.

    Returns
    -------
    pandas.DataFrame
        Combined season event data.
    """

    data_directory = Path(data_directory)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {data_directory}"
        )

    files = sorted(
        data_directory.glob("Ligue 1 2025_2026 AS Monaco p*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier de saison trouvé dans {data_directory}"
        )

    dataframes = []

    for file in files:
        df = load_csv(file)

        # Ajouter une colonne pour identifier le fichier source
        df = df.assign(source_file=file.name)

        dataframes.append(df)

    # Fusionner les DataFrames
    season_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return season_df



def standardize_events(df):
    """
    Standardize StatsBomb event columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Original event data.

    Returns
    -------
    pandas.DataFrame
        Standardized event data.
    """

    # Copier le DataFrame pour ne jamais modifier l'original
    events = df.copy()

    # Définir les correspondances entre colonnes
    column_mapping = {
        "id": "event_id",
        "event_type_name": "type",
        "team_name": "team",
        "player_name": "player"
    }

    # Renommer les colonnes
    events = events.rename(columns=column_mapping)

    # Définir les colonnes obligatoires
    required_columns = [
        "match_id",
        "event_id",
        "index",
        "period",
        "timestamp",
        "type",
        "team",
        "player",
        "location_x",
        "location_y"
    ]

    # Ajouter les colonnes absentes
    for column in required_columns:
        if column not in events.columns:
            events[column] = pd.NA

    return events




def validate_events(df):
    """
    Validate StatsBomb event data.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    bool
        True if validation succeeds.
    """

    required_columns = [
        "match_id",
        "event_id",
        "index",
        "period",
        "timestamp",
        "type",
        "team",
        "player",
        "location_x",
        "location_y"
    ]

    # Vérifier que le DataFrame n'est pas vide
    if df.empty:
        raise ValueError("Le DataFrame est vide.")

    # Vérifier les colonnes obligatoires
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    # Vérifier que les identifiants de matchs sont renseignés
    if df["match_id"].isna().any():
        raise ValueError(
            "Certains événements n'ont pas de match_id."
        )

    # Vérifier que les identifiants d'événements sont renseignés
    if df["event_id"].isna().any():
        raise ValueError(
            "Certains événements n'ont pas de event_id."
        )

    print("Validation des données réussie.")

    return True
