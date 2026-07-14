CREATE TABLE IF NOT EXISTS processed.teams (

    team_id INTEGER PRIMARY KEY,

    code INTEGER UNIQUE,

    pulse_id INTEGER UNIQUE,

    team_name TEXT NOT NULL,

    short_name TEXT NOT NULL,

    fbref_name TEXT,

    understat_name TEXT,

    position SMALLINT,

    played SMALLINT,

    wins SMALLINT,

    draws SMALLINT,

    losses SMALLINT,

    points SMALLINT,

    form DECIMAL(4,2),

    strength SMALLINT,

    strength_overall_home SMALLINT,

    strength_overall_away SMALLINT,

    strength_attack_home SMALLINT,

    strength_attack_away SMALLINT,

    strength_defence_home SMALLINT,

    strength_defence_away SMALLINT

);

CREATE TABLE IF NOT EXISTS processed.players (

    player_id INTEGER PRIMARY KEY,

    code INTEGER UNIQUE,

    team_id INTEGER NOT NULL,

    first_name TEXT,

    second_name TEXT,

    player_name TEXT NOT NULL,

    web_name TEXT,

    fbref_name TEXT,

    understat_name TEXT,

    fbref_player_id TEXT,

    understat_player_id TEXT,

    position TEXT NOT NULL,

    position_code SMALLINT NOT NULL,

    nationality TEXT,

    date_of_birth DATE,

    status CHAR(1),

    chance_of_playing_next_round SMALLINT,

    chance_of_playing_this_round SMALLINT,

    CONSTRAINT fk_team
        FOREIGN KEY (team_id)
        REFERENCES processed.teams(team_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT

);


CREATE TABLE IF NOT EXISTS processed.fixtures (

    fixture_id INTEGER PRIMARY KEY,

    code INTEGER UNIQUE,

    pulse_id INTEGER UNIQUE,

    season TEXT NOT NULL,

    gameweek SMALLINT NOT NULL,

    kickoff_time TIMESTAMP,

    home_team_id INTEGER NOT NULL,

    away_team_id INTEGER NOT NULL,

    team_h_score SMALLINT,

    team_a_score SMALLINT,

    started BOOLEAN,

    finished BOOLEAN,

    finished_provisional BOOLEAN,

    minutes SMALLINT,

    team_h_difficulty SMALLINT,

    team_a_difficulty SMALLINT,

    stats_available BOOLEAN,

    CONSTRAINT fk_home_team
        FOREIGN KEY (home_team_id)
        REFERENCES processed.teams(team_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,

    CONSTRAINT fk_away_team
        FOREIGN KEY (away_team_id)
        REFERENCES processed.teams(team_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT

);


CREATE TABLE IF NOT EXISTS processed.gameweeks (

    gameweek SMALLINT PRIMARY KEY,

    season TEXT NOT NULL,

    name TEXT,

    deadline_time TIMESTAMP,

    average_score SMALLINT,

    highest_score SMALLINT,

    highest_scoring_entry INTEGER,

    finished BOOLEAN,

    data_checked BOOLEAN,

    is_previous BOOLEAN,

    is_current BOOLEAN,

    is_next BOOLEAN,

    cup_leagues_created BOOLEAN,

    h2h_ko_matches_created BOOLEAN,

    can_enter BOOLEAN,

    can_manage BOOLEAN,

    released BOOLEAN,

    ranked_count BIGINT,

    transfers_made BIGINT,

    most_selected TEXT,

    most_transferred_in TEXT,

    top_element INTEGER,

    top_element_info TEXT

);

CREATE TABLE IF NOT EXISTS processed.player_match_stats (

    player_id INTEGER NOT NULL,

    fixture_id INTEGER NOT NULL,

    team_id INTEGER NOT NULL,

    opponent_team_id INTEGER NOT NULL,

    gameweek SMALLINT NOT NULL,

    season TEXT NOT NULL,

    was_home BOOLEAN,

    position VARCHAR(3),

    started BOOLEAN,

    minutes SMALLINT,

    total_points SMALLINT,

    goals_scored SMALLINT,

    assists SMALLINT,

    expected_goals DECIMAL(6,3),

    expected_assists DECIMAL(6,3),

    expected_goal_involvements DECIMAL(6,3),

    shots SMALLINT,

    key_passes SMALLINT,

    xg DECIMAL(6,3),

    xa DECIMAL(6,3),

    clean_sheets SMALLINT,

    goals_conceded SMALLINT,

    expected_goals_conceded DECIMAL(6,3),

    saves SMALLINT,

    recoveries SMALLINT,

    tackles SMALLINT,

    clearances_blocks_interceptions SMALLINT,

    defensive_contribution SMALLINT,

    yellow_cards SMALLINT,

    red_cards SMALLINT,

    own_goals SMALLINT,

    penalties_saved SMALLINT,

    penalties_missed SMALLINT,

    bonus SMALLINT,

    bps SMALLINT,

    influence DECIMAL(8,2),

    creativity DECIMAL(8,2),

    threat DECIMAL(8,2),

    ict_index DECIMAL(8,2),

    PRIMARY KEY (player_id, fixture_id),

    CONSTRAINT fk_pms_player
        FOREIGN KEY (player_id)
        REFERENCES processed.players(player_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,

    CONSTRAINT fk_pms_fixture
        FOREIGN KEY (fixture_id)
        REFERENCES processed.fixtures(fixture_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,

    CONSTRAINT fk_pms_team
        FOREIGN KEY (team_id)
        REFERENCES processed.teams(team_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,

    CONSTRAINT fk_pms_opponent
        FOREIGN KEY (opponent_team_id)
        REFERENCES processed.teams(team_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT

);

CREATE TABLE IF NOT EXISTS processed.player_market_history (

    player_id INTEGER NOT NULL,

    gameweek SMALLINT NOT NULL,

    season TEXT NOT NULL,

    value SMALLINT,

    cost_change_start SMALLINT,

    cost_change_event SMALLINT,

    selected_by_percent DECIMAL(5,2),

    transfers_in INTEGER,

    transfers_out INTEGER,

    transfers_in_event INTEGER,

    transfers_out_event INTEGER,

    status CHAR(1),

    chance_of_playing_this_round SMALLINT,

    chance_of_playing_next_round SMALLINT,

    news TEXT,

    news_added TIMESTAMP,

    PRIMARY KEY (player_id, gameweek, season),

    CONSTRAINT fk_pmh_player
        FOREIGN KEY (player_id)
        REFERENCES processed.players(player_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT

);

CREATE TABLE IF NOT EXISTS processed.player_fpl_season_stats (

    player_id INTEGER NOT NULL,

    season TEXT NOT NULL,

    minutes INTEGER,

    starts SMALLINT,

    total_points SMALLINT,

    goals_scored SMALLINT,

    assists SMALLINT,

    clean_sheets SMALLINT,

    goals_conceded SMALLINT,

    own_goals SMALLINT,

    penalties_saved SMALLINT,

    penalties_missed SMALLINT,

    yellow_cards SMALLINT,

    red_cards SMALLINT,

    saves SMALLINT,

    bonus SMALLINT,

    bps INTEGER,

    influence DECIMAL(8,2),

    creativity DECIMAL(8,2),

    threat DECIMAL(8,2),

    ict_index DECIMAL(8,2),

    expected_goals DECIMAL(8,3),

    expected_assists DECIMAL(8,3),

    expected_goal_involvements DECIMAL(8,3),

    expected_goals_conceded DECIMAL(8,3),

    recoveries SMALLINT,

    tackles SMALLINT,

    clearances_blocks_interceptions SMALLINT,

    defensive_contribution SMALLINT,

    PRIMARY KEY (player_id, season),

    CONSTRAINT fk_pfss_player
        FOREIGN KEY (player_id)
        REFERENCES processed.players(player_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT

);

CREATE TABLE IF NOT EXISTS processed.player_understat_season_stats (

    player_id INTEGER NOT NULL,

    season TEXT NOT NULL,

    matches_played SMALLINT,

    minutes INTEGER,

    goals SMALLINT,

    shots SMALLINT,

    xg DECIMAL(8,3),

    xg_per_shot DECIMAL(8,4),

    shots_per90 DECIMAL(8,3),

    goals_per90 DECIMAL(8,3),

    assists SMALLINT,

    key_passes SMALLINT,

    xa DECIMAL(8,3),

    xa_per90 DECIMAL(8,3),

    xg_chain DECIMAL(8,3),

    xg_chain_per90 DECIMAL(8,3),

    xg_buildup DECIMAL(8,3),

    xg_buildup_per90 DECIMAL(8,3),

    npg SMALLINT,

    npxg DECIMAL(8,3),

    PRIMARY KEY (player_id, season),

    CONSTRAINT fk_puss_player
        FOREIGN KEY (player_id)
        REFERENCES processed.players(player_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT

);

CREATE TABLE IF NOT EXISTS processed.team_match_stats (

    team_id INTEGER NOT NULL,

    fixture_id INTEGER NOT NULL,

    season TEXT NOT NULL,

    gameweek SMALLINT NOT NULL,

    was_home BOOLEAN,

    goals SMALLINT,

    goals_conceded SMALLINT,

    possession DECIMAL(5,2),

    shots SMALLINT,

    shots_on_target SMALLINT,

    shots_conceded SMALLINT,

    shots_on_target_conceded SMALLINT,

    xg DECIMAL(8,3),

    xga DECIMAL(8,3),

    xa DECIMAL(8,3),

    corners SMALLINT,

    fouls SMALLINT,

    yellow_cards SMALLINT,

    red_cards SMALLINT,

    offsides SMALLINT,

    saves SMALLINT,

    passes_attempted INTEGER,

    passes_completed INTEGER,

    pass_completion DECIMAL(5,2),

    touches INTEGER,

    tackles SMALLINT,

    interceptions SMALLINT,

    clearances SMALLINT,

    recoveries SMALLINT,

    blocks SMALLINT,

    aerials_won SMALLINT,

    aerials_lost SMALLINT,

    result CHAR(1),

    formation VARCHAR(10),

    PRIMARY KEY (team_id, fixture_id),

    CONSTRAINT fk_tms_team
        FOREIGN KEY (team_id)
        REFERENCES processed.teams(team_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,

    CONSTRAINT fk_tms_fixture
        FOREIGN KEY (fixture_id)
        REFERENCES processed.fixtures(fixture_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT

);

CREATE TABLE IF NOT EXISTS processed.team_season_stats (

    team_id INTEGER NOT NULL,

    season TEXT NOT NULL,

    matches_played SMALLINT,

    wins SMALLINT,

    draws SMALLINT,

    losses SMALLINT,

    points SMALLINT,

    league_position SMALLINT,

    goals_for SMALLINT,

    goals_against SMALLINT,

    goal_difference SMALLINT,

    expected_goal_difference DECIMAL(8,3),

    clean_sheets SMALLINT,

    possession DECIMAL(5,2),

    shots INTEGER,

    shots_on_target INTEGER,

    xg DECIMAL(8,3),

    xga DECIMAL(8,3),

    key_passes INTEGER,

    progressive_passes INTEGER,

    progressive_carries INTEGER,

    corners INTEGER,

    fouls INTEGER,

    yellow_cards SMALLINT,

    red_cards SMALLINT,

    passes_attempted INTEGER,

    passes_completed INTEGER,

    pass_completion DECIMAL(5,2),

    touches INTEGER,

    tackles INTEGER,

    interceptions INTEGER,

    clearances INTEGER,

    recoveries INTEGER,

    blocks INTEGER,

    aerials_won INTEGER,

    aerials_lost INTEGER,

    PRIMARY KEY (team_id, season),

    CONSTRAINT fk_tss_team
        FOREIGN KEY (team_id)
        REFERENCES processed.teams(team_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT

);