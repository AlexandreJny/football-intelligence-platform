import duckdb
import pandas as pd
import pytest

from src.global_analysis import (
    MONACO_TEAM_ID,
    load_events,
    get_passes,
    get_carries,
    get_shots,
    prepare_line_breaking_passes,
    detect_line_breaking_passes,
    prepare_between_lines_receptions,
    detect_between_lines_receptions,
    calculate_shot_metrics,
    analyze_shot_zones,
    analyze_shot_assists,
    analyze_progressive_passes,
    analyze_progressive_carries,
    analyze_final_third_passes,
    analyze_line_breaking_passes,
    analyze_between_lines_receptions,
    run_offensive_analysis,
    run_progression_analysis,
    get_interceptions,
    get_tackles,
    get_pressures,
    get_recoveries,
    get_blocks,
    get_fouls,
    analyze_defensive_events,
    analyze_recovery_zones,
    calculate_ppda,
    run_defensive_analysis,
)


@pytest.fixture
def sample_events():
    """Small event dataset containing shots, passes, carries and receptions."""
    return pd.DataFrame({
        "event_id": ["s1", "s2", "p1", "p2", "c1", "c2", "r1", "r2"],
        "match_id": [1, 1, 1, 1, 1, 1, 1, 2],
        "period": [1] * 8,
        "team_id": [129, 129, 129, 130, 129, 129, 129, 129],
        "player_id": [10, 11, 10, 20, 10, 11, 12, 12],
        "player_name": [
            "Player A", "Player B", "Player A", "Opponent",
            "Player A", "Player B", "Player C", "Player C",
        ],
        "type": [
            "Shot", "Shot", "Pass", "Pass", "Carry", "Carries",
            "Ball Receipt*", "Ball Receipt*",
        ],
        "statsbomb_xg": [0.5, 0.2, None, None, None, None, None, None],
        "outcome_name": ["Goal", "Saved", None, None, None, None, None, None],
        "location_x": [110, 90, 70, 75, 50, 90, 90, 85],
        "location_y": [40, 30, 35, 35, 40, 40, 40, 65],
        "end_location_x": [None, None, 90, 100, 90, 105, None, None],
        "end_location_y": [None, None, 35, 35, 40, 40, None, None],
        "assisted_shot_id": [None, None, "s1", None, None, None, None, None],
    })


@pytest.fixture
def shot_events(sample_events):
    return get_shots(sample_events)


@pytest.fixture
def pass_events(sample_events):
    return get_passes(sample_events)


@pytest.fixture
def carry_events(sample_events):
    return get_carries(sample_events)

@pytest.fixture
def defensive_events():
    """Dataset containing several defensive event types."""
    return pd.DataFrame({
        "event_id": [
            "i1", "t1", "t2", "pr1",
            "r1", "b1", "f1", "p1", "p2"
        ],
        "match_id": [1] * 9,
        "type": [
            "Interception",
            "Duel",
            "Duel",
            "Pressure",
            "Ball Recovery",
            "Block",
            "Foul Committed",
            "Pass",
            "Pass",
        ],
        "type_name": [
            None,
            "Tackle",
            "Aerial",
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        "team_id": [
            129, 129, 129, 129, 129, 129, 129, 130, 130
        ],
        "player_id": [
            10, 11, 12, 10, 11, 12, 10, 20, 20
        ],
        "player_name": [
            "Player A", "Player B", "Player C",
            "Player A", "Player B", "Player C",
            "Player A", "Opponent", "Opponent"
        ],
        "location_x": [
            75, 70, 65, 80, 50, 90, 65, 50, 80
        ],
        "location_y": [
            40, 40, 30, 40, 35, 40, 45, 40, 40
        ],
        "outcome_name": [
            "Won", "Won", "Lost", None,
            "Success", None, None, None, None
        ],
        "location_bucket": [
            "Attacking Third",
            "Attacking Third",
            "Middle Third",
            "Attacking Third",
            "Middle Third",
            "Defensive Third",
            "Middle Third",
            "Middle Third",
            "Attacking Third",
        ],
        "attacking_direction": [
            "left_to_right",
            "left_to_right",
            "left_to_right",
            "left_to_right",
            "left_to_right",
            "left_to_right",
            "left_to_right",
            "left_to_right",
            "left_to_right",
        ],
    })


def test_load_events_reads_only_matches_involving_team(tmp_path):
    database = tmp_path / "events.duckdb"
    con = duckdb.connect(str(database))
    con.execute("""
        CREATE TABLE events_enriched (
            event_id VARCHAR, match_id INTEGER, "index" INTEGER, team_id INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE match_teams (match_id INTEGER, team_id INTEGER)
    """)
    con.execute("""
        INSERT INTO events_enriched VALUES
        ('a', 1, 1, 129), ('b', 1, 2, 130), ('c', 2, 1, 130)
    """)
    con.execute("""
        INSERT INTO match_teams VALUES (1, 129), (2, 130)
    """)
    con.close()

    result = load_events(database)

    assert result["event_id"].tolist() == ["a", "b"]
    assert set(result["team_id"]) == {129, 130}


def test_load_events_uses_custom_team_id(tmp_path):
    database = tmp_path / "events.duckdb"
    con = duckdb.connect(str(database))
    con.execute("""
        CREATE TABLE events_enriched (
            event_id VARCHAR, match_id INTEGER, "index" INTEGER, team_id INTEGER
        )
    """)
    con.execute("CREATE TABLE match_teams (match_id INTEGER, team_id INTEGER)")
    con.execute("INSERT INTO events_enriched VALUES ('x', 3, 1, 7)")
    con.execute("INSERT INTO match_teams VALUES (3, 7)")
    con.close()

    assert load_events(database, team_id=7)["event_id"].tolist() == ["x"]


def test_load_events_raises_for_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="Base de données introuvable"):
        load_events(tmp_path / "missing.duckdb")


@pytest.mark.parametrize("extractor,event_type,expected", [
    (get_shots, "Shot", 2),
    (get_passes, "Pass", 2),
    # The implementation currently selects the plural event type "Carries".
    (get_carries, "Carries", 1),
])
def test_event_extractors_filter_event_type(sample_events, extractor, event_type, expected):
    result = extractor(sample_events)
    assert len(result) == expected
    assert result["type"].str.lower().eq(event_type.lower()).all()


@pytest.mark.parametrize("extractor,required", [
    (get_shots, ["type", "event_id", "match_id", "team_id", "player_id",
                 "statsbomb_xg", "outcome_name", "location_x", "location_y"]),
    (get_passes, ["type", "event_id", "match_id", "team_id", "player_id",
                  "player_name", "location_x", "location_y",
                  "end_location_x", "end_location_y"]),
    (get_carries, ["type", "event_id", "match_id", "team_id", "player_id",
                   "player_name", "location_x", "location_y",
                   "end_location_x", "end_location_y"]),
])
def test_extractors_raise_for_missing_columns(extractor, required):
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        extractor(pd.DataFrame(columns=[c for c in required if c != "type"]))


def test_get_passes_coerces_coordinates_and_drops_invalid(sample_events):
    events = pd.concat([
        sample_events,
        pd.DataFrame([{
            "event_id": "bad", "match_id": 3, "team_id": 129,
            "player_id": 1, "player_name": "Bad", "type": "Pass",
            "location_x": "not-a-number", "location_y": 20,
            "end_location_x": 90, "end_location_y": 20,
        }]),
    ], ignore_index=True)
    result = get_passes(events)
    assert "bad" not in result["event_id"].tolist()
    assert pd.api.types.is_numeric_dtype(result["location_x"])


def test_get_carries_drops_invalid_coordinates(sample_events):
    events = pd.concat([
        sample_events,
        pd.DataFrame([{
            "event_id": "bad", "match_id": 3, "team_id": 129,
            "player_id": 1, "player_name": "Bad", "type": "Carries",
            "location_x": 10, "location_y": 20,
            "end_location_x": None, "end_location_y": 20,
        }]),
    ], ignore_index=True)
    assert "bad" not in get_carries(events)["event_id"].tolist()


def test_prepare_line_breaking_passes_filters_team_and_valid_pitch(pass_events):
    extra = pass_events.iloc[[0]].copy()
    extra["event_id"] = "outside"
    extra["team_id"] = 129
    extra["location_x"] = 121
    result = prepare_line_breaking_passes(
        pd.concat([pass_events, extra], ignore_index=True), 129
    )

    assert set(result["team_id"]) == {129}
    assert "outside" not in result["event_id"].tolist()
    assert "progression_x" in result.columns
    assert "pass_distance" in result.columns
    assert result.iloc[0]["progression_x"] == 20


def test_prepare_line_breaking_passes_raises_for_missing_columns(pass_events):
    with pytest.raises(ValueError, match="Missing columns"):
        prepare_line_breaking_passes(pass_events.drop(columns="period"), 129)


def test_detect_line_breaking_passes_applies_thresholds():
    prepared = pd.DataFrame({
        "event_id": ["yes", "short", "already_final", "not_reaching"],
        "progression_x": [12, 9, 15, 12],
        "location_x": [70, 70, 85, 70],
        "end_location_x": [82, 82, 100, 79],
    })
    result = detect_line_breaking_passes(prepared)
    assert result["event_id"].tolist() == ["yes"]


def test_detect_line_breaking_passes_accepts_custom_thresholds():
    prepared = pd.DataFrame({
        "event_id": ["a", "b"],
        "progression_x": [8, 8],
        "location_x": [60, 60],
        "end_location_x": [80, 80],
    })
    result = detect_line_breaking_passes(
        prepared, min_progression=8, final_third_start=80
    )
    assert result["event_id"].tolist() == ["a", "b"]


def test_prepare_between_lines_receptions_filters_team_and_coordinates(sample_events):
    extra = sample_events.iloc[[6]].copy()
    extra["event_id"] = "bad_coord"
    extra["location_x"] = 121
    result = prepare_between_lines_receptions(
        pd.concat([sample_events, extra], ignore_index=True), 129
    )
    assert result["event_id"].tolist() == ["r1", "r2"]
    assert result["location_x"].between(0, 120).all()


def test_prepare_between_lines_receptions_raises_for_missing_columns(sample_events):
    with pytest.raises(ValueError, match="Missing columns"):
        prepare_between_lines_receptions(sample_events.drop(columns="period"), 129)


def test_detect_between_lines_receptions_uses_final_third_and_central_corridor():
    prepared = pd.DataFrame({
        "event_id": ["inside", "too_deep", "too_wide", "before"],
        "location_x": [80, 90, 90, 79],
        "location_y": [40, 61, 19, 40],
    })
    result = detect_between_lines_receptions(prepared)
    assert result["event_id"].tolist() == ["inside"]


def test_detect_between_lines_receptions_accepts_custom_bounds():
    prepared = pd.DataFrame({
        "event_id": ["inside"],
        "location_x": [75],
        "location_y": [50],
    })
    result = detect_between_lines_receptions(
        prepared, final_third_start=75, central_y_min=45, central_y_max=55
    )
    assert result["event_id"].tolist() == ["inside"]


def test_calculate_shot_metrics(shot_events):
    metrics = calculate_shot_metrics(shot_events)
    assert metrics["total_shots"] == 2
    assert metrics["matches"] == 1
    assert metrics["shots_per_match"] == pytest.approx(2)
    assert metrics["total_xg"] == pytest.approx(0.7)
    # average_xg is average xG per match (not per shot).
    assert metrics["average_xg"] == pytest.approx(0.7)
    assert metrics["xg_per_shot"] == pytest.approx(0.35)
    assert metrics["goals"] == 1
    assert metrics["conversion_rate"] == pytest.approx(50)
    assert metrics["goals_minus_xg"] == pytest.approx(0.3)


def test_calculate_shot_metrics_empty_dataframe():
    empty = pd.DataFrame(columns=[
        "statsbomb_xg", "match_id", "outcome_name"
    ])
    metrics = calculate_shot_metrics(empty)
    assert metrics["total_shots"] == 0
    assert metrics["matches"] == 0
    assert metrics["shots_per_match"] == 0
    assert metrics["total_xg"] == 0
    assert metrics["average_xg"] == 0
    assert metrics["xg_per_shot"] == 0
    assert metrics["goals"] == 0
    assert metrics["conversion_rate"] == 0
    assert metrics["goals_minus_xg"] == 0


def test_calculate_shot_metrics_average_xg_across_matches():
    shots = pd.DataFrame({
        "statsbomb_xg": [0.5, 0.3, 0.2],
        "match_id": [1, 1, 2],
        "outcome_name": ["Goal", "Saved", "Off T"],
    })
    metrics = calculate_shot_metrics(shots)
    assert metrics["matches"] == 2
    assert metrics["average_xg"] == pytest.approx(0.5)
    assert metrics["shots_per_match"] == pytest.approx(1.5)


def test_analyze_shot_zones_assigns_all_zone_categories():
    shots = pd.DataFrame({
        "event_id": ["wide_out", "central_out", "wide_box", "central_box", "bad"],
        "statsbomb_xg": [0.1, 0.2, 0.3, 0.4, 0.1],
        "location_x": [80, 80, 110, 110, 130],
        "location_y": [20, 40, 20, 40, 40],
    })
    zones = analyze_shot_zones(shots)
    counts = dict(zip(zones["shot_zone"], zones["total_shots"]))
    assert counts == {
        "Central outside area": 1,
        "Central penalty area": 1,
        "Outside penalty area": 1,
        "Penalty area": 1,
        "Unknown": 1,
    }
    assert zones["total_shots"].sum() == 5
    assert zones["shot_percentage"].sum() == pytest.approx(100)


def test_analyze_shot_zones_empty_input():
    shots = pd.DataFrame(columns=[
        "event_id", "statsbomb_xg", "location_x", "location_y"
    ])
    result = analyze_shot_zones(shots)
    assert "shot_zone" in result.columns
    assert result.empty


def test_analyze_shot_assists_links_same_match_and_team_only():
    events = pd.DataFrame({
        "event_id": ["p1", "p2", "p3"],
        "match_id": [1, 1, 2],
        "team_id": [129, 130, 129],
        "player_name": ["Passer A", "Passer B", "Passer C"],
        "type": ["Pass", "Pass", "Pass"],
        "assisted_shot_id": ["s1", "s1", "s2"],
    })
    shots = pd.DataFrame({
        "event_id": ["s1", "s2"],
        # s2 deliberately belongs to another match than its linked pass p3.
        "match_id": [1, 3],
        "team_id": [129, 129],
        "player_name": ["Shooter A", "Shooter B"],
        "statsbomb_xg": [0.4, 0.2],
        "outcome_name": ["Goal", "Saved"],
    })
    result = analyze_shot_assists(events, shots)
    assert len(result) == 1
    assert result.iloc[0]["passer_name"] == "Passer A"
    assert result.iloc[0]["shooter_name"] == "Shooter A"
    assert "event_id" not in result.columns
    assert "assisted_shot_id" not in result.columns


def test_analyze_progressive_passes_returns_events_and_summary():
    passes = pd.DataFrame({
        "team_id": [129, 129, 130],
        "location_x": [50, 50, 50],
        "location_y": [40, 40, 40],
        "end_location_x": [90, 60, 110],
        "end_location_y": [40, 40, 40],
    })
    result = analyze_progressive_passes(passes, 129)
    assert result["summary"]["total_passes"] == 2
    assert result["summary"]["total_progressive_passes"] == 1
    assert result["summary"]["progressive_passes_percentage"] == pytest.approx(50)
    assert len(result["events"]) == 1


def test_analyze_progressive_passes_empty_team():
    passes = pd.DataFrame(columns=[
        "team_id", "location_x", "location_y", "end_location_x", "end_location_y"
    ])
    result = analyze_progressive_passes(passes, 129)
    assert result["summary"]["total_passes"] == 0
    assert result["summary"]["total_progressive_passes"] == 0
    assert result["summary"]["progressive_passes_percentage"] == 0
    assert result["summary"]["average_progression"] == 0
    assert result["summary"]["average_pass_distance"] == 0


def test_analyze_progressive_carries_returns_events_and_summary():
    carries = pd.DataFrame({
        "team_id": [129, 129, 130],
        "location_x": [50, 50, 50],
        "location_y": [40, 40, 40],
        "end_location_x": [90, 55, 110],
        "end_location_y": [40, 40, 40],
    })
    result = analyze_progressive_carries(carries, 129)
    assert result["summary"]["total_carries"] == 2
    assert result["summary"]["total_progressive_carries"] == 1
    assert result["summary"]["progressive_carries_percentage"] == pytest.approx(50)
    assert len(result["events"]) == 1


def test_analyze_progressive_carries_empty_team():
    carries = pd.DataFrame(columns=[
        "team_id", "location_x", "location_y", "end_location_x", "end_location_y"
    ])
    result = analyze_progressive_carries(carries, 129)
    assert result["summary"]["total_carries"] == 0
    assert result["summary"]["total_progressive_carries"] == 0
    assert result["summary"]["progressive_carries_percentage"] == 0
    assert result["summary"]["average_progression"] == 0
    assert result["summary"]["average_carry_distance"] == 0


def test_analyze_final_third_passes_counts_entries_and_summary():
    passes = pd.DataFrame({
        "team_id": [129, 129, 130],
        "location_x": [70, 85, 70],
        "end_location_x": [85, 100, 100],
        "match_id": [1, 1, 1],
        "player_id": [10, 11, 20],
    })
    result = analyze_final_third_passes(passes, 129)
    assert result["summary"]["total_passes"] == 2
    assert result["summary"]["total_final_third_passes"] == 1
    assert result["summary"]["final_third_passes_percentage"] == pytest.approx(50)
    assert result["summary"]["matches"] == 1
    assert result["summary"]["players_involved"] == 1
    assert len(result["events"]) == 1


def test_analyze_final_third_passes_empty_team():
    passes = pd.DataFrame(columns=[
        "team_id", "location_x", "end_location_x", "match_id", "player_id"
    ])
    result = analyze_final_third_passes(passes, 129)
    assert result["summary"]["total_passes"] == 0
    assert result["summary"]["total_final_third_passes"] == 0
    assert result["summary"]["final_third_passes_percentage"] == 0
    assert result["summary"]["average_progression"] == 0
    assert result["summary"]["matches"] == 0
    assert result["summary"]["players_involved"] == 0


def test_analyze_line_breaking_passes_integration(pass_events):
    result = analyze_line_breaking_passes(pass_events, 129)
    assert result["summary"]["total_passes"] == 1
    assert result["summary"]["total_potential_line_breaks"] == 1
    assert result["summary"]["potential_line_breaks_percentage"] == pytest.approx(100)
    assert len(result["events"]) == 1


def test_analyze_line_breaking_passes_empty_input(pass_events):
    result = analyze_line_breaking_passes(pass_events.iloc[0:0], 129)
    assert result["summary"]["total_passes"] == 0
    assert result["summary"]["total_potential_line_breaks"] == 0
    assert result["summary"]["potential_line_breaks_percentage"] == 0
    assert result["summary"]["matches"] == 0
    assert result["summary"]["players_involved"] == 0


def test_analyze_between_lines_receptions_integration(sample_events):
    result = analyze_between_lines_receptions(sample_events, 129)
    assert result["summary"]["total_receptions"] == 2
    assert result["summary"]["total_between_lines_receptions"] == 1
    assert result["summary"]["between_lines_percentage"] == pytest.approx(50)
    assert result["summary"]["matches"] == 1
    assert result["summary"]["players_involved"] == 1
    assert result["events"]["event_id"].tolist() == ["r1"]


def test_analyze_between_lines_receptions_empty_input(sample_events):
    empty = sample_events.iloc[0:0].copy()
    result = analyze_between_lines_receptions(empty, 129)
    assert result["summary"]["total_receptions"] == 0
    assert result["summary"]["total_between_lines_receptions"] == 0
    assert result["summary"]["between_lines_percentage"] == 0
    assert result["summary"]["matches"] == 0
    assert result["summary"]["players_involved"] == 0


def test_run_offensive_analysis_orchestrates_pipeline(monkeypatch, sample_events):
    import src.global_analysis as analysis

    monkeypatch.setattr(analysis, "load_events", lambda path, team_id: sample_events)
    result = run_offensive_analysis("unused.db", team_id=129)
    assert {"metrics", "shots", "zones", "assists"} == set(result)
    assert len(result["shots"]) == 2
    assert result["metrics"]["total_shots"] == 2


def test_run_progression_analysis_orchestrates_pipeline(monkeypatch, sample_events):
    import src.global_analysis as analysis

    monkeypatch.setattr(analysis, "load_events", lambda path, team_id: sample_events)
    result = run_progression_analysis("unused.db", team_id=129)
    assert {
        "progressive_passes", "progressive_carries", "final_third_passes",
        "line_breaking_passes", "between_lines_receptions",
    } == set(result)
    for section in result.values():
        assert set(section) == {"summary", "events"}

def test_get_interceptions(defensive_events):
    result = get_interceptions(defensive_events)

    assert len(result) == 1
    assert result.iloc[0]["event_id"] == "i1"


def test_get_tackles_only_extracts_tackle_duels(defensive_events):
    result = get_tackles(defensive_events)

    assert len(result) == 1
    assert result.iloc[0]["event_id"] == "t1"
    assert result.iloc[0]["type_name"] == "Tackle"


def test_get_tackles_raises_without_type_name(defensive_events):
    events = defensive_events.drop(columns=["type_name"])

    with pytest.raises(ValueError, match="type_name"):
        get_tackles(events)


@pytest.mark.parametrize(
    "extractor,event_type,event_id",
    [
        (get_pressures, "Pressure", "pr1"),
        (get_recoveries, "Ball Recovery", "r1"),
        (get_blocks, "Block", "b1"),
        (get_fouls, "Foul Committed", "f1"),
    ],
)
def test_defensive_extractors(
    defensive_events, extractor, event_type, event_id
):
    result = extractor(defensive_events)

    assert len(result) == 1
    assert result.iloc[0]["type"] == event_type
    assert result.iloc[0]["event_id"] == event_id


def test_defensive_extractor_drops_invalid_coordinates(defensive_events):
    events = defensive_events.copy()

    invalid = events.iloc[[0]].copy()
    invalid["event_id"] = "invalid"
    invalid["location_x"] = None

    events = pd.concat([events, invalid], ignore_index=True)

    result = get_interceptions(events)

    assert "invalid" not in result["event_id"].tolist()

def test_analyze_defensive_events_filters_selected_team(defensive_events):
    interceptions = get_interceptions(defensive_events)

    result = analyze_defensive_events(interceptions, 129)

    assert result["summary"]["total_events"] == 1
    assert result["summary"]["matches"] == 1
    assert result["summary"]["players_involved"] == 1
    assert result["events"]["team_id"].eq(129).all()


def test_analyze_defensive_events_counts_outcomes(defensive_events):
    tackles = get_tackles(defensive_events)

    result = analyze_defensive_events(tackles, 129)

    assert result["summary"]["outcomes"] == {"Won": 1}


def test_analyze_defensive_events_empty_input(defensive_events):
    empty = get_interceptions(defensive_events).iloc[0:0].copy()

    result = analyze_defensive_events(empty, 129)

    assert result["summary"]["total_events"] == 0
    assert result["summary"]["matches"] == 0
    assert result["summary"]["players_involved"] == 0
    assert result["events"].empty


def test_analyze_recovery_zones_only_counts_selected_team(defensive_events):
    recoveries = get_recoveries(defensive_events)

    result = analyze_recovery_zones(recoveries, 129)

    assert result["summary"]["total_recoveries"] == 1
    assert result["zones"]["recoveries"].sum() == 1
    assert result["events"]["team_id"].eq(129).all()


def test_analyze_recovery_zones_empty_input(defensive_events):
    recoveries = get_recoveries(defensive_events).iloc[0:0].copy()

    result = analyze_recovery_zones(recoveries, 129)

    assert result["summary"]["total_recoveries"] == 0
    assert result["zones"].empty
    assert result["events"].empty


def test_analyze_recovery_zones_calculates_percentages(defensive_events):
    recoveries = get_recoveries(defensive_events)

    extra = recoveries.copy()
    extra["event_id"] = "r2"
    extra["location_bucket"] = "Attacking Third"

    recoveries = pd.concat([recoveries, extra], ignore_index=True)

    result = analyze_recovery_zones(recoveries, 129)

    assert result["summary"]["total_recoveries"] == 2
    assert result["zones"]["recoveries"].sum() == 2
    assert result["zones"]["percentage"].sum() == pytest.approx(100)


def test_calculate_ppda_counts_passes_and_defensive_actions(defensive_events):
    result = calculate_ppda(defensive_events, 129)

    assert result["summary"]["total_opponent_passes"] == 1
    assert result["summary"]["total_defensive_actions"] == 3
    assert result["summary"]["ppda"] == pytest.approx(1 / 3)

    assert len(result["matches"]) == 1
    assert result["matches"].iloc[0]["match_id"] == 1


def test_calculate_ppda_excludes_defensive_actions_outside_pressing_zone(
    defensive_events
):
    events = defensive_events.copy()

    # Move all Monaco defensive actions outside the pressing zone.
    mask = (
        (events["team_id"] == 129)
        & (events["type"].isin([
            "Duel", "Interception", "Foul Committed"
        ]))
    )

    events.loc[mask, "location_x"] = 40

    result = calculate_ppda(events, 129)

    assert result["summary"]["total_defensive_actions"] == 0
    assert result["summary"]["ppda"] is None


def test_calculate_ppda_returns_empty_matches_with_empty_input():
    events = pd.DataFrame(columns=[
        "type",
        "type_name",
        "team_id",
        "match_id",
        "location_x",
        "attacking_direction",
    ])

    result = calculate_ppda(events, 129)

    assert result["summary"]["total_opponent_passes"] == 0
    assert result["summary"]["total_defensive_actions"] == 0
    assert result["summary"]["ppda"] is None
    assert result["matches"].empty

    assert list(result["matches"].columns) == [
        "match_id",
        "opponent_passes",
        "defensive_actions",
        "ppda",
    ]


def test_calculate_ppda_handles_matches_without_defensive_actions():
    events = pd.DataFrame({
        "type": ["Pass"],
        "type_name": [None],
        "team_id": [130],
        "match_id": [1],
        "location_x": [50],
        "attacking_direction": ["left_to_right"],
    })

    result = calculate_ppda(events, 129)

    assert result["summary"]["total_opponent_passes"] == 1
    assert result["summary"]["total_defensive_actions"] == 0
    assert result["summary"]["ppda"] is None
    assert result["matches"].iloc[0]["ppda"] is None


def test_calculate_ppda_raises_for_missing_columns():
    events = pd.DataFrame({
        "type": ["Pass"],
        "team_id": [130],
    })

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        calculate_ppda(events, 129)


def test_run_defensive_analysis_orchestrates_pipeline(
    monkeypatch, defensive_events
):
    import src.global_analysis as analysis

    monkeypatch.setattr(
        analysis,
        "load_events",
        lambda path, team_id: defensive_events
    )

    result = run_defensive_analysis("unused.db", team_id=129)

    assert {
        "interceptions",
        "tackles",
        "pressures",
        "recoveries",
        "blocks",
        "fouls",
        "recovery_zones",
        "ppda",
    } == set(result)

    for category in [
        "interceptions",
        "tackles",
        "pressures",
        "recoveries",
        "blocks",
        "fouls",
    ]:
        assert "summary" in result[category]
        assert "events" in result[category]

    assert "zones" in result["recovery_zones"]
    assert "summary" in result["ppda"]
    assert "matches" in result["ppda"]