CREATE TABLE IF NOT EXISTS raw.teams (

    id INTEGER PRIMARY KEY,
    code INTEGER,
    name TEXT,
    short_name TEXT,
    strength INTEGER,
    strength_overall_home INTEGER,
    strength_overall_away INTEGER,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER
);

CREATE TABLE IF NOT EXISTS raw.events (

    id INTEGER PRIMARY KEY,
    name TEXT,
    deadline_time TIMESTAMP,
    finished BOOLEAN,
    is_current BOOLEAN,
    is_next BOOLEAN,
    is_previous BOOLEAN
);

CREATE TABLE IF NOT EXISTS raw.element_types (

    id INTEGER PRIMARY KEY,
    plural_name TEXT,
    singular_name TEXT,
    singular_name_short TEXT,
    squad_select INTEGER,
    squad_min_play INTEGER,
    squad_max_play INTEGER
);