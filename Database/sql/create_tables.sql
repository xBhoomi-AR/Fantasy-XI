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