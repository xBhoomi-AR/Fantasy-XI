# Official FPL API Research

## Endpoint

`https://fantasy.premierleague.com/api/bootstrap-static/`

### What is bootstrap-static?

The `bootstrap-static` endpoint acts as the master endpoint of the Fantasy Premier League API. It provides all the static information required by the FPL website when it loads, including player details, teams, positions, gameweeks, chips, and various metadata.

Instead of making multiple API requests for basic information, the FPL website retrieves most of its core data from this endpoint.

---

## Elements (Players)

The `elements` section contains the complete list of all Fantasy Premier League players.

Each object inside `elements` represents **one player**.

Each player object contains:
- Identity information
- Availability information
- Fantasy market information
- Performance statistics
- Advanced statistics
- Expected statistics
- Per-90 statistics
- Ranking statistics

## 1. Identity Information

The Identity Information category contains the basic metadata required to uniquely identify each player and establish relationships with other datasets. These fields are primarily used for database indexing, data integration, and display purposes rather than as direct machine learning features.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| id | Unique FPL player identifier | Keep | Primary key used throughout the database. |
| code | Internal FPL player code | Keep | Useful for mapping FPL assets and cross-referencing data. |
| opta_code | Official Opta player identifier | Keep | Important for matching external datasets such as Understat and FBref. |
| first_name | Player's first name | Keep | Display and identification purposes. |
| second_name | Player's surname | Keep | Display and identification purposes. |
| web_name | Official FPL display name | Keep | Commonly used player name in Fantasy Premier League. |
| known_name | Alternate/common name | Optional | Useful only if available for player matching. |
| photo | Player image filename | Discard | No analytical or predictive value. |
| birth_date | Player's date of birth | Keep (Derived) | Used to derive player's age instead of using the raw date. |
| region | Player's nationality/region identifier | Optional | May be useful for future analysis but not a core feature. |
| team | Current team identifier | Keep | Required for fixture, opponent and team-based analysis. |
| team_code | Internal team code | Optional | Redundant if team ID is already stored. |
| element_type | Player position (GK, DEF, MID, FWD) | Keep | Essential feature for squad construction and model training. |
| team_join_date | Date player joined current club | Optional | Can be used for experience-related analysis. |
| squad_number | Jersey number | Discard | No predictive value for Fantasy Premier League. |
-------

## 2. Availability & Status Information

The Availability and Status category contains information about a player's eligibility and likelihood of participating in upcoming fixtures. These fields are crucial for Fantasy Premier League since a player who is unavailable due to injury, suspension, or rotation cannot contribute fantasy points. Such information can significantly improve prediction accuracy and transfer recommendations.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| status | Current availability status (Available, Injured, Suspended, etc.) | Keep | Essential for determining player availability. |
| can_select | Whether the player can currently be selected | Keep | Useful for validating squad selection. |
| can_transact | Whether transfers involving the player are allowed | Keep | Important during transfer recommendations. |
| chance_of_playing_this_round | Probability of playing in the current Gameweek | Keep | Strong predictive feature for expected minutes and points. |
| chance_of_playing_next_round | Probability of playing in the next Gameweek | Keep | Valuable for transfer planning and long-term prediction. |
| news | Injury or availability news provided by FPL | Keep | Can be processed to identify injuries or fitness concerns. |
| news_added | Timestamp of the latest news update | Optional | Useful for tracking freshness of injury information. |
| removed | Indicates whether the player has been removed from the game | Keep | Prevents selecting unavailable or retired players. |
| scout_risks | Additional risk indicators from FPL Scout | Optional | May provide supplementary qualitative information. |
| scout_news_link | Link to detailed Scout news | Discard | External reference only; no direct analytical value. |
----------------------------------

## 3. Market & Fantasy Information

The Market and Fantasy Information category captures the behaviour of Fantasy Premier League managers rather than the player alone. These fields reflect player popularity, price fluctuations, ownership trends, and transfer activity. They are valuable for modelling market sentiment, identifying differential picks, and understanding price dynamics throughout the season.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| now_cost | Current player price (×10) | Keep | Essential for budget optimization and value calculations. |
| price_change_percent | Percentage change in price | Keep | Useful for analysing market trends and price volatility. |
| cost_change_event | Price change during the current Gameweek | Keep | Indicates short-term market movement. |
| cost_change_event_fall | Price decrease during the current Gameweek | Keep | Helps identify falling assets. |
| cost_change_start | Total price increase since season start | Keep | Useful for analysing long-term value growth. |
| cost_change_start_fall | Total price decrease since season start | Keep | Captures long-term depreciation. |
| selected_by_percent | Percentage of managers owning the player | Keep | Critical for ownership modelling, differential picks, and game-theoretic analysis. |
| transfers_in | Total transfers into the player | Keep | Reflects player popularity and market sentiment. |
| transfers_out | Total transfers out of the player | Keep | Indicates declining confidence or injury concerns. |
| transfers_in_event | Transfers into the player during the current Gameweek | Keep | Useful for detecting sudden market trends. |
| transfers_out_event | Transfers out of the player during the current Gameweek | Keep | Helps identify panic selling or injury-driven transfers. |
| special | Indicates whether the player is marked as a special asset | Discard | Rarely used and provides little predictive value. |
-----

## 4. Basic Performance Statistics

The Basic Performance Statistics category contains the traditional football performance metrics accumulated by a player throughout the season. These statistics directly measure a player's contribution during matches and form the foundation for evaluating historical performance. Most of these fields are strong predictive features for forecasting future Fantasy Premier League points.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| minutes | Total minutes played | Keep | One of the strongest indicators of player reliability and expected future points. |
| starts | Number of matches started | Keep | Reflects manager trust and expected playing time. |
| goals_scored | Total goals scored | Keep | Primary attacking performance metric. |
| assists | Total assists | Keep | Measures creative contribution and fantasy scoring potential. |
| clean_sheets | Number of clean sheets | Keep | Crucial for goalkeepers and defenders. |
| goals_conceded | Goals conceded while on the pitch | Keep | Important for defensive players and clean sheet probability. |
| saves | Total goalkeeper saves | Keep | Major fantasy point source for goalkeepers. |
| own_goals | Number of own goals | Optional | Rare occurrence with limited predictive value. |
| penalties_saved | Penalties saved | Keep | Important goalkeeper statistic. |
| penalties_missed | Penalties missed | Keep | Reflects finishing reliability and future penalty performance. |
| yellow_cards | Yellow cards received | Keep | Indicates disciplinary risk and potential point deductions. |
| red_cards | Red cards received | Keep | Strong indicator of suspension risk and missed matches. |
| bonus | Total bonus points earned | Keep | Captures overall match influence beyond standard statistics. |
| bps | Bonus Point System score | Keep | Strong indicator of overall player contribution and bonus potential. |
---

## 5. Advanced Performance Statistics

The Advanced Performance Statistics category consists of analytical metrics designed to capture player performance beyond traditional football statistics. These metrics provide a deeper understanding of attacking, defensive, and creative contributions, making them highly valuable for machine learning models and modern sports analytics.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| expected_goals | Expected Goals (xG) | Keep | Better indicator of goal-scoring potential than actual goals alone. |
| expected_assists | Expected Assists (xA) | Keep | Measures chance creation independent of teammate finishing. |
| expected_goal_involvements | Expected Goal Involvement (xGI) | Keep | Combines attacking contributions into a single metric. |
| expected_goals_conceded | Expected Goals Conceded (xGC) | Keep | Useful for evaluating defensive strength. |
| influence | FPL Influence Index | Keep | Measures overall impact on matches. |
| creativity | FPL Creativity Index | Keep | Quantifies chance creation and attacking creativity. |
| threat | FPL Threat Index | Keep | Measures goal-scoring threat based on attacking actions. |
| ict_index | Influence-Creativity-Threat Index | Keep | Composite metric summarizing attacking performance. |
| clearances_blocks_interceptions | Defensive actions | Keep | Important for evaluating defenders. |
| recoveries | Ball recoveries | Keep | Indicates defensive work rate. |
| tackles | Successful tackles | Keep | Useful defensive performance indicator. |
| defensive_contribution | Overall defensive contribution | Keep | Captures defensive effectiveness beyond traditional statistics. |
-----

## 6. Per-90 Performance Statistics

Per-90 statistics normalize player performance by playing time, allowing fair comparisons between players with different minutes played. These metrics are widely used in football analytics because they measure efficiency rather than cumulative output, making them valuable predictive features.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| expected_goals_per_90 | xG per 90 minutes | Keep | Measures scoring efficiency independent of playing time. |
| expected_assists_per_90 | xA per 90 minutes | Keep | Measures creative efficiency. |
| expected_goal_involvements_per_90 | xGI per 90 minutes | Keep | Overall attacking efficiency metric. |
| expected_goals_conceded_per_90 | xGC per 90 minutes | Keep | Defensive efficiency indicator. |
| goals_conceded_per_90 | Goals conceded per 90 | Keep | Useful for defensive evaluation. |
| saves_per_90 | Saves per 90 minutes | Keep | Goalkeeper efficiency metric. |
| starts_per_90 | Starts normalized per 90 | Optional | Limited additional information beyond starts and minutes. |
| clean_sheets_per_90 | Clean sheets per 90 | Keep | Defensive efficiency metric. |
| defensive_contribution_per_90 | Defensive contribution per 90 | Keep | Normalized defensive effectiveness. |
------

## 7. Fantasy Performance Metrics

The Fantasy Performance Metrics category contains statistics specifically designed for Fantasy Premier League rather than real-world football. These metrics summarize a player's fantasy output, consistency, and expected returns, making them useful for evaluating overall fantasy performance and value.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| total_points | Total FPL points scored this season | Keep | Primary historical fantasy performance metric. |
| event_points | Points scored in the latest Gameweek | Keep | Captures recent fantasy performance. |
| points_per_game | Average fantasy points per game | Keep | Measures consistency across the season. |
| form | Recent form calculated by FPL | Keep | Strong short-term performance indicator. |
| value_form | Form relative to player price | Keep | Useful for identifying value picks. |
| value_season | Season value relative to price | Keep | Measures long-term cost effectiveness. |
| ep_this | Expected points this Gameweek (FPL estimate) | Optional | Useful as a baseline for comparison, but avoid using as a training feature if building an independent prediction model. |
| ep_next | Expected points next Gameweek (FPL estimate) | Optional | Same as above; useful for benchmarking rather than prediction. |
| dreamteam_count | Number of Dream Team appearances | Optional | Indicates exceptional performances but contributes little additional predictive value. |
| in_dreamteam | Whether player is in the current Dream Team | Discard | Snapshot information with limited predictive usefulness. |

## 8. Ranking Statistics

Ranking Statistics compare each player against all other players (or players in the same position) for different performance metrics. While useful for analysis and visualization, many of these rankings can be derived directly from the underlying statistics and therefore are not essential features.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| influence_rank | Rank based on Influence Index | Optional | Can be derived from influence values. |
| influence_rank_type | Positional influence rank | Optional | Useful for positional comparisons. |
| creativity_rank | Rank based on Creativity Index | Optional | Can be derived from creativity values. |
| creativity_rank_type | Positional creativity rank | Optional | Useful for positional analysis. |
| threat_rank | Rank based on Threat Index | Optional | Derived from threat values. |
| threat_rank_type | Positional threat rank | Optional | Useful for positional comparisons. |
| ict_index_rank | Rank based on ICT Index | Optional | Can be derived from ICT Index. |
| ict_index_rank_type | Positional ICT rank | Optional | Useful for positional comparisons. |
| now_cost_rank | Rank based on current price | Optional | Can be derived from player prices. |
| now_cost_rank_type | Positional price rank | Optional | Derived information. |
| form_rank | Rank based on recent form | Optional | Derived from form values. |
| form_rank_type | Positional form rank | Optional | Derived information. |
| points_per_game_rank | Rank based on average points | Optional | Derived from points_per_game. |
| points_per_game_rank_type | Positional average points rank | Optional | Derived information. |
| selected_rank | Rank based on ownership | Optional | Can be derived from ownership percentage. |
| selected_rank_type | Positional ownership rank | Optional | Derived information. |