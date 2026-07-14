CREATE TABLE IF NOT EXISTS raw.player_history (

    player_id INTEGER,
    fixture_id INTEGER,
    gameweek INTEGER,

    opponent_team INTEGER,

    kickoff_time TIMESTAMP,

    was_home BOOLEAN,

    minutes INTEGER,

    goals_scored INTEGER,
    assists INTEGER,

    clean_sheets INTEGER,

    goals_conceded INTEGER,

    own_goals INTEGER,

    penalties_saved INTEGER,
    penalties_missed INTEGER,

    yellow_cards INTEGER,
    red_cards INTEGER,

    saves INTEGER,

    bonus INTEGER,
    bps INTEGER,

    influence REAL,
    creativity REAL,
    threat REAL,
    ict_index REAL,

    expected_goals REAL,
    expected_assists REAL,
    expected_goal_involvements REAL,
    expected_goals_conceded REAL,

    total_points INTEGER,

    value INTEGER
);