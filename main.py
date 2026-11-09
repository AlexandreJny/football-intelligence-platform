
from pathlib import Path

from src.statsbomb_loader import load_season
from src.events import build_events
from src.enrichment import (
    build_events_enriched,
    save_enriched_tables
)
from src.relational import build_relational_tables

from src.database import (
    create_database,
    load_tables,
    check_primary_keys,
    validate_foreign_keys,
    display_database_summary
)

from src.global_analysis import run_offensive_analysis


def main():

    print("\n===== FOOTBALL INTELLIGENCE PLATFORM =====")

    # 1. Charger les données brutes
    data_directory = Path("data/raw")

    season_df = load_season(data_directory)

    print("\n[OK] Données brutes chargées.")
    print("Dimensions :", season_df.shape)

    # 2. Construire Events
    events = build_events(season_df)

    print("\n[OK] Table Events construite.")
    print("Dimensions :", events.shape)

    # 3. Construire les tables relationnelles
    tables = build_relational_tables(events)

    # 4. Exporter les CSV
    processed_directory = Path("data/processed")

    processed_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n===== EXPORT DES TABLES =====")

    for name, df in tables.items():

        output_path = (
            processed_directory /
            f"{name}_2025_2026.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(f"[OK] {name} : {len(df)} lignes")

    # 5. Créer / ouvrir DuckDB
    database_path = Path("database/football.duckdb")

    connection = create_database(database_path)

    try:

        # 6. Importer les tables relationnelles
        print("\n===== IMPORT DANS DUCKDB =====")

        load_tables(
            connection,
            processed_directory
        )
        # 7. Contrôler les clés primaires
        check_primary_keys(connection)

        # 8. Vérifier les relations
        validate_foreign_keys(connection)

        # 9. Résumé de la base
        display_database_summary(connection)

        # 10. Construction des tables enrichies

        print("\n===== CONSTRUCTION EVENTS ENRICHED =====")

        enriched_tables = build_events_enriched(season_df)

        connection = create_database(database_path)

        try:
            save_enriched_tables(
                connection,
                enriched_tables
            )

        finally:
            connection.close()

    finally:
        connection.close()

    # 11. Analyse offensive
    print("\n===== ANALYSE OFFENSIVE AS MONACO =====")

    results = run_offensive_analysis(
        database_path=database_path,
        team_id=129
    )

    print("\n--- INDICATEURS OFFENSIFS ---")

    for metric, value in results["metrics"].items():

        if isinstance(value, float):
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value}")

    print("\n--- RÉPARTITION DES TIRS PAR ZONE ---")

    print(results["zones"].to_string(index=False))

    print("\n--- ANALYSE DES PASSES ASSOCIÉES AUX TIRS ---")

    print(
        results["assists"].to_string(index=False)
    )

    print("\n===== PIPELINE TERMINÉ =====")


if __name__ == "__main__":
    main()
