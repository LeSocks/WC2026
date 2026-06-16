from src.data.seed_top10_ratings import (
    build_player_ratings_dataframe,
    build_rankings_dataframe,
)
from src.data.scraper import extract_table, flatten_columns


def test_top10_rankings_are_present() -> None:
    rankings = build_rankings_dataframe()

    assert len(rankings) == 10
    assert rankings.iloc[0]["team"] == "Argentina"
    assert rankings.iloc[0]["fifa_rank"] == 1
    assert set(rankings["fifa_rank"]) == set(range(1, 11))


def test_player_seed_schema_and_ranges() -> None:
    ratings = build_player_ratings_dataframe()
    required_columns = {
        "team",
        "fifa_rank",
        "player_name",
        "position",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physical",
        "overall",
        "playstyle",
    }
    stat_columns = [
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physical",
        "overall",
    ]

    assert required_columns.issubset(ratings.columns)
    assert len(ratings) == 110
    assert ratings.groupby("team").size().eq(11).all()

    for column in stat_columns:
        assert ratings[column].between(0, 100).all()


def test_fbref_table_parser_flattens_columns() -> None:
    html = """
    <table id="stats_standard_9">
      <thead>
        <tr><th>Player</th><th>Nation</th><th>90s</th></tr>
      </thead>
      <tbody>
        <tr><td>Example Player</td><td>fr FRA</td><td>12.4</td></tr>
      </tbody>
    </table>
    """

    dataframe = extract_table(html, "stats_standard_9")

    assert flatten_columns([("Standard", "Player"), ("Playing Time", "90s")]) == [
        "Standard_Player",
        "Playing Time_90s",
    ]
    assert dataframe.iloc[0]["Player"] == "Example Player"
