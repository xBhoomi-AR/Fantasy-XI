CREATE TABLE IF NOT EXISTS raw.players (

    id INTEGER PRIMARY KEY,

    code INTEGER,

    first_name TEXT,
    second_name TEXT,
    web_name TEXT,

    team INTEGER,
    team_code INTEGER,

    element_type INTEGER,

    now_cost INTEGER,
    cost_change_start INTEGER,
    cost_change_event INTEGER,

    total_points INTEGER,
    points_per_game REAL,

    minutes INTEGER,
    starts INTEGER,

    goals_scored INTEGER,
    assists INTEGER,

    clean_sheets INTEGER,
    goals_conceded INTEGER,

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

    selected_by_percent REAL,

    transfers_in INTEGER,
    transfers_out INTEGER,

    status CHAR(1),

    chance_of_playing_this_round INTEGER,
    chance_of_playing_next_round INTEGER,

    news TEXT,

    photo TEXT,

    team_join_date DATE
);