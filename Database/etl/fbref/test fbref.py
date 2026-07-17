import soccerdata as sd

understat = sd.Understat(
    leagues="ENG-Premier League",
    seasons="2025"
)

print(understat)

players = understat.read_player_season_stats()

print(players.head())

players.to_csv("understat_player_stats_2025.csv")

players.shape

print(players.columns.tolist())
