import numpy as np
from synthetic_data import generate_season


class FPLEnvironment:
    def __init__(self, seed=42):
        self.seed = seed
        self.num_gameweeks = 10
        self.budget_limit = 48.0
        self.transfer_penalty = 1.0

        self.data = None
        self.players = None
        self.player_ids = None
        self.actions = []

        self.current_gameweek = 1
        self.squad = []
        self.done = False

        self._create_actions()
        self.reset(seed)


    def _create_actions(self):
        # Action 0 always means no transfer
        self.actions = [("HOLD", None, None)]

        player_ids = [f"P{i:02d}" for i in range(1, 13)]

        # Every possible sell-buy pair gets a fixed action number
        for sell_player in player_ids:
            for buy_player in player_ids:
                if sell_player != buy_player:
                    self.actions.append(
                        ("TRANSFER", sell_player, buy_player)
                    )


    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed

        self.data = generate_season(seed=self.seed)

        # Player information that stays fixed during the season
        self.players = (
            self.data
            .sort_values("gameweek")
            .groupby("player_id")
            .first()
            .reset_index()
        )

        self.player_ids = self.players["player_id"].tolist()

        self.current_gameweek = 1
        self.done = False

        self._create_initial_squad()

        return self.get_state()


    def _create_initial_squad(self):
        squad = []

        # Pick two affordable players from each position
        for position in ["DEF", "MID", "FWD"]:
            position_players = self.players[
                self.players["position"] == position
            ].sort_values("price")

            selected = position_players.head(2)["player_id"].tolist()
            squad.extend(selected)

        self.squad = squad

        if self.get_squad_cost() > self.budget_limit:
            raise ValueError("Initial squad exceeds the budget limit.")


    def get_squad_cost(self):
        squad_players = self.players[
            self.players["player_id"].isin(self.squad)
        ]

        return float(squad_players["price"].sum())


    def get_remaining_budget(self):
        return self.budget_limit - self.get_squad_cost()


    def _get_player_position(self, player_id):
        row = self.players[
            self.players["player_id"] == player_id
        ].iloc[0]

        return row["position"]


    def _get_player_price(self, player_id):
        row = self.players[
            self.players["player_id"] == player_id
        ].iloc[0]

        return float(row["price"])


    def is_valid_action(self, action_index):
        if action_index < 0 or action_index >= len(self.actions):
            return False

        action_type, sell_player, buy_player = self.actions[action_index]

        if action_type == "HOLD":
            return True

        # The sold player must currently be in the squad
        if sell_player not in self.squad:
            return False

        # We cannot buy a player already in the squad
        if buy_player in self.squad:
            return False

        # Transfers must keep the same position
        if self._get_player_position(sell_player) != self._get_player_position(buy_player):
            return False

        new_cost = (
            self.get_squad_cost()
            - self._get_player_price(sell_player)
            + self._get_player_price(buy_player)
        )

        if new_cost > self.budget_limit:
            return False

        return True


    def get_valid_actions(self):
        return [
            i
            for i in range(len(self.actions))
            if self.is_valid_action(i)
        ]


    def get_action_mask(self):
        mask = np.zeros(len(self.actions), dtype=np.float32)

        for action_index in self.get_valid_actions():
            mask[action_index] = 1.0

        return mask


    def get_state(self):
        gw_data = (
            self.data[
                self.data["gameweek"] == self.current_gameweek
            ]
            .set_index("player_id")
        )

        state = []

        position_encoding = {
            "DEF": 0.0,
            "MID": 0.5,
            "FWD": 1.0
        }

        for player_id in self.player_ids:
            player = gw_data.loc[player_id]

            owned = 1.0 if player_id in self.squad else 0.0

            # Values are scaled so the neural network gets smaller inputs
            state.extend([
                float(player["price"]) / 10.0,
                float(player["form"]),
                float(player["fixture_difficulty"]) / 5.0,
                float(player["predicted_points"]) / 10.0,
                owned,
                position_encoding[player["position"]]
            ])

        state.extend([
            self.get_remaining_budget() / self.budget_limit,
            self.current_gameweek / self.num_gameweeks
        ])

        return np.array(state, dtype=np.float32)


    def _calculate_gameweek_points(self):
        gw_data = self.data[
            self.data["gameweek"] == self.current_gameweek
        ]

        squad_data = gw_data[
            gw_data["player_id"].isin(self.squad)
        ]

        return float(squad_data["actual_points"].sum())


    def step(self, action_index):
        if self.done:
            raise ValueError("Episode has already finished.")

        if not self.is_valid_action(action_index):
            raise ValueError("Invalid action selected.")

        action_type, sell_player, buy_player = self.actions[action_index]

        transfer_made = False

        if action_type == "TRANSFER":
            self.squad.remove(sell_player)
            self.squad.append(buy_player)
            transfer_made = True

        squad_points = self._calculate_gameweek_points()

        # Reward is based on average points per player
        reward = squad_points / len(self.squad)

        if transfer_made:
            reward -= self.transfer_penalty

        # Small bonus for a good gameweek
        if reward >= 6:
            reward += 2

        # Small penalty for a poor gameweek
        elif reward <= 4:
            reward -= 2
        info = {
            "gameweek": self.current_gameweek,
            "action_type": action_type,
            "sell_player": sell_player,
            "buy_player": buy_player,
            "squad_points": squad_points,
            "transfer_penalty": self.transfer_penalty if transfer_made else 0.0,
            "reward": reward,
            "squad_cost": self.get_squad_cost(),
            "remaining_budget": self.get_remaining_budget(),
            "squad": self.squad.copy()
        }

        if self.current_gameweek >= self.num_gameweeks:
            self.done = True
            next_state = np.zeros_like(self.get_state())
        else:
            self.current_gameweek += 1
            next_state = self.get_state()

        return next_state, reward, self.done, info


    def describe_action(self, action_index):
        action_type, sell_player, buy_player = self.actions[action_index]

        if action_type == "HOLD":
            return "HOLD"

        sell_name = self.players[
            self.players["player_id"] == sell_player
        ]["name"].iloc[0]

        buy_name = self.players[
            self.players["player_id"] == buy_player
        ]["name"].iloc[0]

        return f"{sell_name} ({sell_player}) -> {buy_name} ({buy_player})"


    def print_squad(self):
        squad_info = self.players[
            self.players["player_id"].isin(self.squad)
        ][
            ["player_id", "name", "position", "price"]
        ]

        print(squad_info.to_string(index=False))
        print(f"\nSquad cost: {self.get_squad_cost():.1f}")
        print(f"Remaining budget: {self.get_remaining_budget():.1f}")


if __name__ == "__main__":
    env = FPLEnvironment(seed=42)

    print("\nInitial squad:")
    env.print_squad()

    state = env.get_state()

    print(f"\nState size: {len(state)}")
    print(f"Total encoded actions: {len(env.actions)}")

    valid_actions = env.get_valid_actions()

    print(f"Valid actions in GW1: {len(valid_actions)}")

    print("\nSome valid actions:")
    for action_index in valid_actions[:10]:
        print(
            action_index,
            "-",
            env.describe_action(action_index)
        )

    # Pick the first available transfer for a manual test
    transfer_actions = [
        action
        for action in valid_actions
        if env.actions[action][0] == "TRANSFER"
    ]

    if transfer_actions:
        test_action = transfer_actions[0]
    else:
        test_action = 0

    print("\nManual test action:")
    print(env.describe_action(test_action))

    next_state, reward, done, info = env.step(test_action)

    print("\nGW result:")
    print(f"Squad points: {info['squad_points']}")
    print(f"Transfer penalty: {info['transfer_penalty']}")
    print(f"Reward: {info['reward']}")
    print(f"Episode finished: {done}")

    print("\nSquad after action:")
    env.print_squad()

    print(f"\nMoved to GW{env.current_gameweek}")