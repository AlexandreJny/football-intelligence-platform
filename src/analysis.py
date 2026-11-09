
import pandas as pd


def explore_events(events_df):
    """
    Perform exploratory analysis on StatsBomb events.

    Parameters
    ----------
    events_df : pandas.DataFrame
        Standardized Events table.

    Returns
    -------
    dict
        Results of exploratory analyses.
    """

    results = {}

    # 1. Nombre d'événements par match
    results["events_per_match"] = (
        events_df.groupby("match_id")
        .size()
        .sort_values(ascending=False)
        .rename("event_count")
    )

    # 2. Nombre d'événements par type
    results["events_per_type"] = (
        events_df["type"]
        .value_counts(dropna=False)
        .rename_axis("event_type")
        .reset_index(name="event_count")
    )

    # 3. Nombre d'événements par équipe
    results["events_per_team"] = (
        events_df.groupby(
            ["team_id", "team_name"],
            dropna=False
        )
        .size()
        .reset_index(name="event_count")
        .sort_values("event_count", ascending=False)
    )

    # 4. Nombre d'événements par période
    results["events_per_period"] = (
        events_df["period"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("period")
        .reset_index(name="event_count")
    )

    # 5. Nombre de joueurs distincts par match
    results["players_per_match"] = (
        events_df.groupby("match_id")
        ["player_id"]
        .nunique()
        .rename("distinct_players")
        .reset_index()
    )

    # 6. Top 10 des joueurs ayant réalisé le plus d'événements
    player_events = (
        events_df.dropna(subset=["player_id"])
        .groupby("player_id")
        .agg(
            player_name=("player_name", "first"),
            event_count=("id", "count")
        )
        .sort_values("event_count", ascending=False)
        .head(10)
        .reset_index()
    )

    results["top_10_players"] = player_events

    return results


def display_results(results):
    """
    Display exploratory analysis results.
    """

    print("\n===== 1. ÉVÉNEMENTS PAR MATCH =====")
    print(results["events_per_match"])

    print("\n===== 2. ÉVÉNEMENTS PAR TYPE =====")
    print(results["events_per_type"].to_string(index=False))

    print("\n===== 3. ÉVÉNEMENTS PAR ÉQUIPE =====")
    print(results["events_per_team"].to_string(index=False))

    print("\n===== 4. ÉVÉNEMENTS PAR PÉRIODE =====")
    print(results["events_per_period"].to_string(index=False))

    print("\n===== 5. JOUEURS DISTINCTS PAR MATCH =====")
    print(results["players_per_match"].to_string(index=False))

    print("\n===== 6. TOP 10 DES JOUEURS =====")
    print(results["top_10_players"].to_string(index=False))
