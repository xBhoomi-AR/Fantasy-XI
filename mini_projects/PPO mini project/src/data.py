"""Download prices, build technical features, and create chronological splits."""

import pandas as pd
import ta
import yfinance as yf

from config import END_DATE, PROCESSED_DATA_PATH, RAW_DATA_PATH, START_DATE, STOCKS


def download_prices() -> pd.DataFrame:
    frames = []
    for ticker in STOCKS:
        print(f"Downloading {ticker}...")
        frame = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        frame.columns = frame.columns.get_level_values(0)
        frame = frame.reset_index()
        frame["ticker"] = ticker
        frames.append(frame[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]])

    prices = pd.concat(frames, ignore_index=True).sort_values(["ticker", "Date"]).reset_index(drop=True)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(RAW_DATA_PATH, index=False)
    return prices


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy().sort_values(["ticker", "Date"]).reset_index(drop=True)
    grouped = data.groupby("ticker")
    data["daily_return"] = grouped["Close"].pct_change()
    data["volatility20"] = grouped["daily_return"].transform(lambda x: x.rolling(20).std())
    data["price_momentum"] = grouped["Close"].transform(lambda x: x / x.shift(20))
    data["sma20"] = grouped["Close"].transform(lambda x: x.rolling(20).mean())
    data["sma50"] = grouped["Close"].transform(lambda x: x.rolling(50).mean())
    data["price_sma20_ratio"] = data["Close"] / data["sma20"]
    data["price_sma50_ratio"] = data["Close"] / data["sma50"]
    data["sma10"] = grouped["Close"].transform(lambda x: x.rolling(10).mean())
    data["sma30"] = grouped["Close"].transform(lambda x: x.rolling(30).mean())
    data["sma_crossover"] = data["sma10"] / data["sma30"]
    data["ema12"] = grouped["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    data["price_ema12_ratio"] = data["Close"] / data["ema12"]
    data["rsi14"] = grouped["Close"].transform(lambda x: ta.momentum.RSIIndicator(x, window=14).rsi())
    data["macd"] = grouped["Close"].transform(lambda x: ta.trend.MACD(x).macd())
    data["macd_signal"] = grouped["Close"].transform(lambda x: ta.trend.MACD(x).macd_signal())
    data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["volume_sma20"] = grouped["Volume"].transform(lambda x: x.rolling(20).mean())
    data["volume_ratio"] = data["Volume"] / data["volume_sma20"]
    data = data.dropna().reset_index(drop=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH, index=False)
    return data


def load_feature_data() -> dict[str, pd.DataFrame]:
    if PROCESSED_DATA_PATH.exists():
        data = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])
    else:
        data = build_features(download_prices())
    return {ticker: frame.reset_index(drop=True) for ticker, frame in data.groupby("ticker")}


def split_data(feature_data: dict[str, pd.DataFrame]):
    train, validation, test = {}, {}, {}
    for ticker, frame in feature_data.items():
        frame = frame.sort_values("Date").reset_index(drop=True)
        train[ticker] = frame[frame["Date"] < "2022-01-01"].reset_index(drop=True)
        validation[ticker] = frame[(frame["Date"] >= "2022-01-01") & (frame["Date"] < "2023-07-01")].reset_index(drop=True)
        test[ticker] = frame[frame["Date"] >= "2023-07-01"].reset_index(drop=True)
    return train, validation, test
