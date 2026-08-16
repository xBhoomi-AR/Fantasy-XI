import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """Long-only portfolio environment rewarded for beating equal weight."""

    feature_cols = [
        "daily_return", "price_momentum", "price_sma20_ratio", "price_sma50_ratio",
        "sma_crossover", "price_ema12_ratio", "rsi14", "macd", "macd_signal",
        "volatility20", "high_low_range", "volume_ratio",
    ]

    def __init__(self, feature_data, tickers, window=20, initial_capital=100_000,
                 transaction_cost=0.001, episode_length=252):
        super().__init__()
        self.tickers, self.n_stocks = tickers, len(tickers)
        self.window, self.initial_capital = window, initial_capital
        self.transaction_cost, self.episode_length = transaction_cost, episode_length
        self._build_arrays(feature_data)
        obs_size = window * self.n_stocks * len(self.feature_cols) + self.n_stocks + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, shape=(self.n_stocks,), dtype=np.float32)

    def _build_arrays(self, feature_data):
        dates = None
        for ticker in self.tickers:
            index = pd.DatetimeIndex(pd.to_datetime(feature_data[ticker]["Date"]))
            dates = index if dates is None else dates.intersection(index)
        self.trading_dates = dates.sort_values()
        self.total_trading_days = len(self.trading_dates)
        self.feature_array = np.empty((self.total_trading_days, self.n_stocks, len(self.feature_cols)), dtype=np.float32)
        self.close_price_array = np.empty((self.total_trading_days, self.n_stocks), dtype=np.float32)
        for index, ticker in enumerate(self.tickers):
            frame = feature_data[ticker].copy()
            frame["Date"] = pd.to_datetime(frame["Date"])
            frame = frame.set_index("Date").loc[self.trading_dates]
            self.feature_array[:, index] = frame[self.feature_cols].fillna(0.0).to_numpy(np.float32)
            self.close_price_array[:, index] = frame["Close"].to_numpy(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        maximum_start = self.total_trading_days - self.episode_length - 1
        if maximum_start <= self.window:
            raise ValueError("Not enough data for the requested window and episode length.")
        self.start_day = self.np_random.integers(self.window, maximum_start)
        self.current_day = self.start_day
        self.portfolio_weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        self.portfolio_value = float(self.initial_capital)
        self.returns_history = []
        return self._observation(), self._info()

    def step(self, action):
        logits = np.clip(action.astype(np.float32), -20.0, 20.0)
        logits -= logits.max()
        weights = np.exp(logits)
        new_weights = weights / (weights.sum() + 1e-8)
        stock_returns = (self.close_price_array[self.current_day + 1] - self.close_price_array[self.current_day]) / (self.close_price_array[self.current_day] + 1e-8)
        portfolio_return = float(np.dot(new_weights, stock_returns))
        turnover = float(np.abs(new_weights - self.portfolio_weights).sum())
        net_return = portfolio_return - self.transaction_cost * turnover
        benchmark_return = float(stock_returns.mean())
        reward = float(np.log1p(max(net_return, -0.999)) - np.log1p(max(benchmark_return, -0.999)))
        self.portfolio_value *= 1.0 + net_return
        self.portfolio_weights = new_weights
        self.current_day += 1
        self.returns_history.append(net_return)
        terminated = self.current_day - self.start_day >= self.episode_length
        return self._observation(), reward, terminated, False, self._info()

    def _observation(self):
        features = self.feature_array[self.current_day - self.window:self.current_day].flatten()
        return np.concatenate((features, self.portfolio_weights, [self.portfolio_value / self.initial_capital])).astype(np.float32)

    def _info(self):
        return {"portfolio_value": self.portfolio_value, "date": str(self.trading_dates[self.current_day])}
