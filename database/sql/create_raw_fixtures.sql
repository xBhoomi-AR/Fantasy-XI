CREATE TABLE IF NOT EXISTS raw.fixtures (

    id INTEGER PRIMARY KEY,
    code BIGINT,
    pulse_id INTEGER,

    event INTEGER,

    kickoff_time TIMESTAMP,

    finished BOOLEAN,
    started BOOLEAN,

    minutes INTEGER,

    team_h INTEGER,
    team_a INTEGER,

    team_h_score INTEGER,
    team_a_score INTEGER,

    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER

);