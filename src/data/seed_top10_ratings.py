from __future__ import annotations

from pathlib import Path

import pandas as pd


RANKING_DATE = "2026-06-11"
RANKING_SOURCE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
RATINGS_SOURCE = "manual_seed_estimate"

TOP10_RANKINGS = [
    {"fifa_rank": 1, "team": "Argentina", "confederation": "CONMEBOL"},
    {"fifa_rank": 2, "team": "Spain", "confederation": "UEFA"},
    {"fifa_rank": 3, "team": "France", "confederation": "UEFA"},
    {"fifa_rank": 4, "team": "England", "confederation": "UEFA"},
    {"fifa_rank": 5, "team": "Portugal", "confederation": "UEFA"},
    {"fifa_rank": 6, "team": "Brazil", "confederation": "CONMEBOL"},
    {"fifa_rank": 7, "team": "Morocco", "confederation": "CAF"},
    {"fifa_rank": 8, "team": "Netherlands", "confederation": "UEFA"},
    {"fifa_rank": 9, "team": "Belgium", "confederation": "UEFA"},
    {"fifa_rank": 10, "team": "Germany", "confederation": "UEFA"},
]

FBREF_TEAM_SOURCES = [
    {
        "team": "France",
        "fbref_squad_id": "361ca564",
        "fbref_url": "https://fbref.com/en/squads/361ca564/France-Stats",
        "status": "known_from_tdd",
    },
    {
        "team": "Argentina",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
    {
        "team": "Spain",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
    {
        "team": "England",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
    {
        "team": "Portugal",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
    {
        "team": "Brazil",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
    {
        "team": "Morocco",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
    {
        "team": "Netherlands",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
    {
        "team": "Belgium",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
    {
        "team": "Germany",
        "fbref_squad_id": "",
        "fbref_url": "",
        "status": "todo_verify_before_scrape",
    },
]


def player(
    team: str,
    name: str,
    position: str,
    age: int,
    pace: int,
    shooting: int,
    passing: int,
    dribbling: int,
    defending: int,
    physical: int,
    overall: int,
    playstyle: str,
) -> dict[str, object]:
    return {
        "team": team,
        "player_name": name,
        "position": position,
        "age": age,
        "is_starter": True,
        "pace": pace,
        "shooting": shooting,
        "passing": passing,
        "dribbling": dribbling,
        "defending": defending,
        "physical": physical,
        "overall": overall,
        "playstyle": playstyle,
        "ratings_source": RATINGS_SOURCE,
    }


PLAYER_RATINGS = [
    player("Argentina", "Emiliano Martinez", "GK", 33, 52, 25, 72, 63, 88, 86, 88, "traditional_gk"),
    player("Argentina", "Nahuel Molina", "RB", 28, 83, 66, 78, 80, 80, 77, 82, "wing_back"),
    player("Argentina", "Cristian Romero", "CB", 28, 73, 43, 69, 69, 87, 86, 86, "stopper"),
    player("Argentina", "Nicolas Otamendi", "CB", 38, 55, 47, 69, 64, 84, 82, 82, "stopper"),
    player("Argentina", "Nicolas Tagliafico", "LB", 33, 74, 59, 76, 77, 82, 78, 82, "full_back"),
    player("Argentina", "Rodrigo De Paul", "CM", 32, 76, 75, 84, 82, 79, 82, 85, "box_to_box"),
    player("Argentina", "Enzo Fernandez", "CM", 25, 72, 76, 86, 83, 78, 78, 86, "deep_lying_pm"),
    player("Argentina", "Alexis Mac Allister", "CM", 27, 75, 78, 85, 84, 76, 76, 86, "box_to_box"),
    player("Argentina", "Lionel Messi", "RW", 38, 77, 88, 91, 94, 35, 65, 90, "creator"),
    player("Argentina", "Julian Alvarez", "ST", 26, 86, 84, 80, 85, 56, 78, 86, "pressing_forward"),
    player("Argentina", "Lautaro Martinez", "ST", 28, 82, 87, 75, 84, 55, 84, 88, "poacher"),
    player("Spain", "Unai Simon", "GK", 29, 55, 21, 77, 66, 85, 82, 85, "sweeper_keeper"),
    player("Spain", "Dani Carvajal", "RB", 34, 78, 64, 82, 82, 84, 82, 86, "full_back"),
    player("Spain", "Robin Le Normand", "CB", 29, 66, 39, 73, 70, 84, 82, 84, "ball_playing_cb"),
    player("Spain", "Aymeric Laporte", "CB", 32, 62, 50, 82, 76, 86, 80, 86, "ball_playing_cb"),
    player("Spain", "Marc Cucurella", "LB", 27, 78, 55, 78, 80, 82, 80, 83, "full_back"),
    player("Spain", "Rodri", "CDM", 29, 66, 80, 90, 84, 88, 86, 91, "deep_lying_pm"),
    player("Spain", "Pedri", "CM", 23, 78, 76, 88, 90, 68, 67, 87, "creator"),
    player("Spain", "Fabian Ruiz", "CM", 30, 70, 79, 86, 84, 77, 78, 86, "box_to_box"),
    player("Spain", "Nico Williams", "LW", 23, 93, 80, 78, 88, 45, 71, 85, "dribbler"),
    player("Spain", "Alvaro Morata", "ST", 33, 79, 84, 76, 80, 45, 79, 84, "target_man"),
    player("Spain", "Lamine Yamal", "RW", 18, 86, 82, 84, 90, 38, 62, 86, "dribbler"),
    player("France", "Mike Maignan", "GK", 30, 58, 23, 79, 68, 89, 84, 89, "sweeper_keeper"),
    player("France", "Jules Kounde", "RB", 27, 82, 45, 78, 77, 86, 82, 86, "full_back"),
    player("France", "William Saliba", "CB", 25, 79, 41, 76, 77, 88, 86, 88, "ball_playing_cb"),
    player("France", "Dayot Upamecano", "CB", 27, 82, 40, 73, 75, 86, 87, 86, "stopper"),
    player("France", "Theo Hernandez", "LB", 28, 93, 76, 81, 84, 79, 84, 87, "wing_back"),
    player("France", "Aurelien Tchouameni", "CDM", 26, 72, 76, 84, 81, 84, 86, 86, "ball_winner"),
    player("France", "Eduardo Camavinga", "CM", 23, 80, 73, 84, 86, 81, 82, 86, "box_to_box"),
    player("France", "Antoine Griezmann", "CAM", 35, 76, 86, 88, 88, 60, 73, 88, "creator"),
    player("France", "Ousmane Dembele", "RW", 29, 92, 83, 84, 90, 42, 64, 87, "dribbler"),
    player("France", "Kylian Mbappe", "LW", 27, 97, 90, 83, 93, 39, 78, 91, "complete_fwd"),
    player("France", "Marcus Thuram", "ST", 28, 88, 83, 78, 85, 47, 86, 85, "complete_fwd"),
    player("England", "Jordan Pickford", "GK", 32, 55, 24, 78, 65, 84, 83, 84, "traditional_gk"),
    player("England", "Kyle Walker", "RB", 36, 88, 60, 78, 78, 82, 82, 84, "full_back"),
    player("England", "John Stones", "CB", 32, 71, 55, 82, 80, 85, 80, 86, "ball_playing_cb"),
    player("England", "Marc Guehi", "CB", 25, 78, 38, 73, 74, 84, 82, 84, "stopper"),
    player("England", "Luke Shaw", "LB", 30, 77, 66, 83, 81, 81, 80, 84, "full_back"),
    player("England", "Declan Rice", "CDM", 27, 77, 76, 85, 82, 86, 88, 88, "ball_winner"),
    player("England", "Jude Bellingham", "CAM", 22, 82, 86, 86, 89, 79, 84, 90, "box_to_box"),
    player("England", "Phil Foden", "LW", 26, 84, 85, 86, 91, 58, 68, 89, "creator"),
    player("England", "Bukayo Saka", "RW", 24, 86, 84, 84, 89, 61, 72, 88, "dribbler"),
    player("England", "Harry Kane", "ST", 32, 69, 91, 85, 83, 50, 84, 90, "complete_fwd"),
    player("England", "Cole Palmer", "CAM", 24, 78, 86, 85, 88, 49, 68, 87, "creator"),
    player("Portugal", "Diogo Costa", "GK", 26, 60, 22, 78, 67, 86, 84, 86, "sweeper_keeper"),
    player("Portugal", "Diogo Dalot", "RB", 27, 83, 70, 79, 81, 80, 82, 84, "wing_back"),
    player("Portugal", "Ruben Dias", "CB", 29, 67, 42, 74, 70, 89, 88, 89, "stopper"),
    player("Portugal", "Antonio Silva", "CB", 22, 72, 39, 75, 72, 84, 82, 84, "ball_playing_cb"),
    player("Portugal", "Nuno Mendes", "LB", 24, 90, 63, 80, 84, 80, 78, 85, "wing_back"),
    player("Portugal", "Joao Palhinha", "CDM", 30, 64, 66, 76, 76, 88, 89, 86, "ball_winner"),
    player("Portugal", "Vitinha", "CM", 26, 75, 77, 88, 88, 74, 72, 87, "deep_lying_pm"),
    player("Portugal", "Bruno Fernandes", "CAM", 31, 74, 87, 90, 86, 69, 76, 88, "creator"),
    player("Portugal", "Rafael Leao", "LW", 26, 94, 84, 80, 89, 38, 81, 86, "dribbler"),
    player("Portugal", "Cristiano Ronaldo", "ST", 41, 78, 89, 76, 84, 34, 78, 86, "poacher"),
    player("Portugal", "Bernardo Silva", "RW", 31, 79, 81, 88, 91, 62, 68, 88, "creator"),
    player("Brazil", "Alisson", "GK", 33, 58, 22, 78, 68, 88, 86, 89, "sweeper_keeper"),
    player("Brazil", "Danilo", "RB", 34, 72, 63, 78, 77, 82, 82, 83, "full_back"),
    player("Brazil", "Marquinhos", "CB", 32, 76, 50, 79, 78, 87, 84, 87, "ball_playing_cb"),
    player("Brazil", "Gabriel Magalhaes", "CB", 28, 75, 45, 73, 71, 86, 87, 86, "stopper"),
    player("Brazil", "Guilherme Arana", "LB", 29, 80, 68, 79, 82, 77, 76, 82, "wing_back"),
    player("Brazil", "Bruno Guimaraes", "CM", 28, 72, 78, 85, 84, 80, 82, 86, "box_to_box"),
    player("Brazil", "Lucas Paqueta", "CM", 28, 76, 80, 84, 87, 73, 78, 85, "creator"),
    player("Brazil", "Rodrygo", "RW", 25, 88, 85, 82, 88, 39, 66, 87, "complete_fwd"),
    player("Brazil", "Vinicius Junior", "LW", 25, 96, 86, 82, 92, 36, 75, 90, "dribbler"),
    player("Brazil", "Endrick", "ST", 19, 86, 82, 72, 84, 40, 78, 82, "poacher"),
    player("Brazil", "Raphinha", "RW", 29, 88, 84, 83, 86, 55, 74, 86, "pressing_forward"),
    player("Morocco", "Yassine Bounou", "GK", 35, 50, 22, 73, 62, 86, 84, 85, "traditional_gk"),
    player("Morocco", "Achraf Hakimi", "RB", 27, 92, 76, 83, 86, 80, 78, 86, "wing_back"),
    player("Morocco", "Nayef Aguerd", "CB", 30, 75, 42, 73, 70, 82, 82, 82, "stopper"),
    player("Morocco", "Romain Saiss", "CB", 36, 55, 55, 72, 68, 80, 80, 80, "stopper"),
    player("Morocco", "Noussair Mazraoui", "LB", 28, 81, 68, 82, 84, 80, 77, 84, "full_back"),
    player("Morocco", "Sofyan Amrabat", "CDM", 29, 69, 65, 78, 78, 82, 83, 82, "ball_winner"),
    player("Morocco", "Azzedine Ounahi", "CM", 26, 76, 72, 80, 84, 69, 70, 80, "box_to_box"),
    player("Morocco", "Bilal El Khannouss", "CAM", 22, 75, 75, 82, 85, 56, 65, 81, "creator"),
    player("Morocco", "Brahim Diaz", "RW", 26, 82, 81, 82, 87, 47, 66, 84, "creator"),
    player("Morocco", "Hakim Ziyech", "LW", 33, 74, 82, 86, 86, 50, 66, 83, "creator"),
    player("Morocco", "Youssef En-Nesyri", "ST", 29, 79, 83, 68, 76, 44, 84, 82, "target_man"),
    player("Netherlands", "Bart Verbruggen", "GK", 23, 56, 20, 75, 65, 82, 82, 82, "sweeper_keeper"),
    player("Netherlands", "Denzel Dumfries", "RB", 30, 83, 75, 76, 78, 80, 86, 83, "wing_back"),
    player("Netherlands", "Virgil van Dijk", "CB", 34, 78, 60, 74, 72, 90, 88, 89, "stopper"),
    player("Netherlands", "Micky van de Ven", "CB", 25, 88, 43, 72, 74, 84, 84, 84, "stopper"),
    player("Netherlands", "Nathan Ake", "LB", 31, 76, 58, 79, 79, 84, 80, 84, "full_back"),
    player("Netherlands", "Frenkie de Jong", "CM", 29, 79, 76, 88, 88, 78, 76, 88, "deep_lying_pm"),
    player("Netherlands", "Tijjani Reijnders", "CM", 27, 78, 79, 84, 85, 74, 76, 85, "box_to_box"),
    player("Netherlands", "Xavi Simons", "CAM", 23, 84, 82, 84, 88, 55, 67, 85, "creator"),
    player("Netherlands", "Cody Gakpo", "LW", 27, 84, 84, 82, 85, 48, 78, 86, "complete_fwd"),
    player("Netherlands", "Memphis Depay", "ST", 32, 79, 84, 82, 85, 38, 74, 84, "false_nine"),
    player("Netherlands", "Jeremie Frimpong", "RW", 25, 94, 75, 79, 86, 72, 72, 84, "wing_back"),
    player("Belgium", "Thibaut Courtois", "GK", 34, 47, 20, 74, 60, 90, 86, 90, "traditional_gk"),
    player("Belgium", "Timothy Castagne", "RB", 30, 78, 63, 76, 78, 78, 78, 80, "full_back"),
    player("Belgium", "Wout Faes", "CB", 28, 67, 38, 70, 68, 80, 82, 80, "stopper"),
    player("Belgium", "Arthur Theate", "CB", 26, 70, 44, 72, 70, 80, 82, 80, "stopper"),
    player("Belgium", "Maxim De Cuyper", "LB", 25, 78, 68, 78, 80, 75, 73, 79, "wing_back"),
    player("Belgium", "Amadou Onana", "CDM", 24, 75, 70, 78, 79, 82, 88, 83, "ball_winner"),
    player("Belgium", "Youri Tielemans", "CM", 29, 67, 80, 84, 82, 74, 77, 83, "deep_lying_pm"),
    player("Belgium", "Kevin De Bruyne", "CAM", 34, 73, 87, 93, 88, 64, 74, 90, "creator"),
    player("Belgium", "Jeremy Doku", "LW", 24, 91, 78, 78, 90, 39, 69, 84, "dribbler"),
    player("Belgium", "Romelu Lukaku", "ST", 33, 79, 86, 76, 78, 42, 88, 85, "target_man"),
    player("Belgium", "Leandro Trossard", "RW", 31, 78, 82, 82, 86, 50, 68, 84, "creator"),
    player("Germany", "Marc-Andre ter Stegen", "GK", 34, 54, 22, 84, 70, 88, 84, 89, "sweeper_keeper"),
    player("Germany", "Joshua Kimmich", "RB", 31, 70, 76, 89, 84, 82, 78, 88, "inverted_full_back"),
    player("Germany", "Antonio Rudiger", "CB", 33, 82, 55, 72, 71, 87, 88, 87, "stopper"),
    player("Germany", "Jonathan Tah", "CB", 30, 76, 42, 74, 72, 85, 86, 85, "stopper"),
    player("Germany", "David Raum", "LB", 28, 82, 66, 81, 82, 76, 78, 82, "wing_back"),
    player("Germany", "Robert Andrich", "CDM", 31, 64, 72, 78, 77, 82, 86, 82, "ball_winner"),
    player("Germany", "Ilkay Gundogan", "CM", 35, 66, 81, 87, 86, 72, 72, 85, "deep_lying_pm"),
    player("Germany", "Jamal Musiala", "CAM", 23, 85, 84, 84, 91, 54, 69, 88, "dribbler"),
    player("Germany", "Florian Wirtz", "CAM", 23, 80, 84, 88, 90, 60, 70, 89, "creator"),
    player("Germany", "Leroy Sane", "RW", 30, 90, 84, 81, 87, 44, 72, 86, "dribbler"),
    player("Germany", "Kai Havertz", "ST", 27, 78, 83, 82, 84, 61, 80, 86, "false_nine"),
]


def build_rankings_dataframe() -> pd.DataFrame:
    rankings = pd.DataFrame(TOP10_RANKINGS)
    rankings["ranking_date"] = RANKING_DATE
    rankings["source_url"] = RANKING_SOURCE_URL
    rankings["source_note"] = "FIFA official ranking page and 11 June 2026 ranking article."
    return rankings


def build_player_ratings_dataframe() -> pd.DataFrame:
    rankings = pd.DataFrame(TOP10_RANKINGS)[["team", "fifa_rank"]]
    players = pd.DataFrame(PLAYER_RATINGS)
    players = players.merge(rankings, on="team", how="left")
    players["ranking_date"] = RANKING_DATE
    return players[
        [
            "team",
            "fifa_rank",
            "player_name",
            "position",
            "age",
            "is_starter",
            "pace",
            "shooting",
            "passing",
            "dribbling",
            "defending",
            "physical",
            "overall",
            "playstyle",
            "ratings_source",
            "ranking_date",
        ]
    ].sort_values(["fifa_rank", "team", "position", "player_name"])


def write_seed_data(output_dir: str | Path = "data/raw") -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rankings_path = output_path / f"fifa_rankings_top10_{RANKING_DATE}.csv"
    ratings_path = output_path / "fifa_style_player_ratings_top10.csv"
    fbref_sources_path = output_path / "fbref_team_sources.csv"

    build_rankings_dataframe().to_csv(rankings_path, index=False)
    build_player_ratings_dataframe().to_csv(ratings_path, index=False)
    pd.DataFrame(FBREF_TEAM_SOURCES).to_csv(fbref_sources_path, index=False)

    return {
        "rankings": rankings_path,
        "ratings": ratings_path,
        "fbref_sources": fbref_sources_path,
    }


if __name__ == "__main__":
    for label, path in write_seed_data().items():
        print(f"{label}: {path}")
