import requests
import psycopg2

from etl.common.config import (
    DATABASE,
    FPL_BASE_URL,
    HEADERS,
    REQUEST_TIMEOUT
)


connection = psycopg2.connect(
    host=DATABASE["host"],
    port=DATABASE["port"],
    database=DATABASE["database"],
    user=DATABASE["user"],
    password=DATABASE["password"]
)

cursor = connection.cursor()



print("Downloading bootstrap-static...")

response = requests.get(
    f"{FPL_BASE_URL}/bootstrap-static/",
    headers=HEADERS,
    timeout=REQUEST_TIMEOUT
)

response.raise_for_status()

bootstrap = response.json()

print("Download Complete.")



print("Inserting Teams...")

for team in bootstrap["teams"]:

    cursor.execute(
        """
        INSERT INTO processed.teams
        (
            team_id,
            code,
            pulse_id,
            team_name,
            short_name,
            fbref_name,
            understat_name,
            position,
            played,
            wins,
            draws,
            losses,
            points,
            form,
            strength,
            strength_overall_home,
            strength_overall_away,
            strength_attack_home,
            strength_attack_away,
            strength_defence_home,
            strength_defence_away
        )

        VALUES
        (
            %s,%s,%s,%s,%s,
            NULL,NULL,
            %s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s
        )

        ON CONFLICT (team_id)
        DO NOTHING;
        """,

        (
            team["id"],
            team["code"],
            team["pulse_id"],
            team["name"],
            team["short_name"],

            team["position"],
            team["played"],
            team["win"],
            team["draw"],
            team["loss"],
            team["points"],
            float(team["form"]) if team["form"] else None,

            team["strength"],
            team["strength_overall_home"],
            team["strength_overall_away"],
            team["strength_attack_home"],
            team["strength_attack_away"],
            team["strength_defence_home"],
            team["strength_defence_away"]
        )
    )

connection.commit()

print("Teams inserted successfully.")



print("Inserting Players...")

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

for player in bootstrap["elements"]:

    cursor.execute(
        """
        INSERT INTO processed.players
        (
            player_id,
            code,
            team_id,
            first_name,
            second_name,
            player_name,
            web_name,
            photo,
            fbref_name,
            understat_name,
            fbref_player_id,
            understat_player_id,
            position,
            position_code,
            nationality,
            date_of_birth,
            status,
            chance_of_playing_next_round,
            chance_of_playing_this_round,
            news,
            news_added
        )

        VALUES
        (
            %s,%s,%s,%s,%s,%s,%s,%s,
            NULL,NULL,NULL,NULL,
            %s,%s,
            NULL,NULL,
            %s,%s,%s,%s,%s
        )

        ON CONFLICT (player_id)
        DO NOTHING;
        """,

        (
            player["id"],
            player["code"],
            player["team"],

            player["first_name"],
            player["second_name"],

            f'{player["first_name"]} {player["second_name"]}',

            player["web_name"],

            player["photo"],

            POSITION_MAP[player["element_type"]],
            player["element_type"],

            player["status"],

            player["chance_of_playing_next_round"],
            player["chance_of_playing_this_round"],

            player["news"],
            player["news_added"]
        )
    )

connection.commit()

print(" Players inserted successfully.")


print("Inserting Gameweeks...")

CURRENT_SEASON = "2025/26"

for gw in bootstrap["events"]:

    cursor.execute(
        """
        INSERT INTO processed.gameweeks
        (
            gameweek,
            season,
            name,
            deadline_time,
            average_score,
            highest_score,
            highest_scoring_entry,
            finished,
            data_checked,
            is_previous,
            is_current,
            is_next,
            cup_leagues_created,
            h2h_ko_matches_created,
            can_enter,
            can_manage,
            released,
            ranked_count,
            transfers_made,
            most_selected,
            most_transferred_in,
            top_element,
            top_element_info
        )

        VALUES
        (
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
        )

        ON CONFLICT (gameweek)
        DO NOTHING;
        """,

        (
            gw["id"],
            CURRENT_SEASON,
            gw["name"],
            gw["deadline_time"],
            gw["average_entry_score"],
            gw["highest_score"],
            gw["highest_scoring_entry"],
            gw["finished"],
            gw["data_checked"],
            gw["is_previous"],
            gw["is_current"],
            gw["is_next"],
            gw["cup_leagues_created"],
            gw["h2h_ko_matches_created"],
            gw["can_enter"],
            gw["can_manage"],
            gw["released"],
            gw["ranked_count"],
            gw["transfers_made"],

            None,                  # most_selected
            None,                  # most_transferred_in

            gw["top_element"],

            str(gw["top_element_info"])
            if gw["top_element_info"] is not None
            else None
        )
    )

connection.commit()

print("Gameweeks inserted successfully.")

print("Downloading Fixtures...")

response = requests.get(
    f"{FPL_BASE_URL}/fixtures/",
    headers=HEADERS,
    timeout=REQUEST_TIMEOUT
)

response.raise_for_status()

fixtures = response.json()

print("Fixtures Downloaded.")


print("Inserting Fixtures...")

CURRENT_SEASON = "2025/26"

for fixture in fixtures:

    cursor.execute(
        """
        INSERT INTO processed.fixtures
        (
            fixture_id,
            code,
            pulse_id,
            season,
            gameweek,
            kickoff_time,
            home_team_id,
            away_team_id,
            team_h_score,
            team_a_score,
            started,
            finished,
            finished_provisional,
            minutes,
            team_h_difficulty,
            team_a_difficulty,
            stats_available
        )

        VALUES
        (
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
        )

        ON CONFLICT (fixture_id)
        DO NOTHING;
        """,

        (
            fixture["id"],
            fixture["code"],
            fixture["pulse_id"],
            CURRENT_SEASON,

            fixture["event"],

            fixture["kickoff_time"],

            fixture["team_h"],
            fixture["team_a"],

            fixture["team_h_score"],
            fixture["team_a_score"],

            fixture["started"],
            fixture["finished"],
            fixture["finished_provisional"],

            fixture["minutes"],

            fixture["team_h_difficulty"],
            fixture["team_a_difficulty"],

            bool(fixture["stats"])
        )
    )

connection.commit()

print("Fixtures inserted successfully.")