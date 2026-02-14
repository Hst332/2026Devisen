# forex_forecast_backtest.py
# ------------------------------------------------------------
# Daily Forex Forecast (BUY/HOLD/SELL) + Backtest Performance
# Robust against yfinance MultiIndex issues
# ------------------------------------------------------------

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt


# ============================================================
# PAIR SELECTION (5 best + 1 secret)
# ============================================================

PAIRS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "EURJPY": "EURJPY=X",   # Geheimtipp
}


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    ret10 = close.pct_change(10)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    vol20 = ret1.rolling(20).std()

    rsi14 = rsi(close, 14)
    atr14 = atr(high, low, close, 14)
    macd_line, macd_sig = macd(close)

    ma_slope = ma50.pct_change(5)

    feats = pd.DataFrame({
        "ret1": ret1,
        "ret5": ret5,
        "ret10": ret10,
        "vol20": vol20,
        "rsi14": rsi14,
        "atr14": atr14 / close,
        "macd": macd_line,
        "macd_sig": macd_sig,
        "macd_hist": macd_line - macd_sig,
        "dist_ma20": (close / ma20) - 1.0,
        "dist_ma50": (close / ma50) - 1.0,
        "dist_ma200": (close / ma200) - 1.0,
        "ma50_slope": ma_slope,
    }, index=df.index)

    return feats


def make_labels(df: pd.DataFrame, hold_threshold: float = 0.0005) -> pd.Series:
    next_ret = df["Close"].pct_change().shift(-1)
    y = np.where(next_ret > hold_threshold, 1,
                 np.where(next_ret < -hold_threshold, -1, 0))
    return pd.Series(y, index=df.index)


def regime_from_price(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    ma200 = close.rolling(200).mean()
    dist = (close / ma200) - 1.0

    regime = pd.Series("neutral", index=df.index)
    regime[dist > 0.01] = "bull"
    regime[dist < -0.01] = "bear"
    return regime


# ============================================================
# ROBUST DATA LOADER
# ============================================================

def fetch_ohlc(ticker: str, years: int) -> pd.DataFrame:
    period = f"{years}y"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Normalize names
    df.columns = [str(c).strip().title() for c in df.columns]

    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"{ticker} missing column {col}. Got {df.columns}")

    return df[required].dropna().copy()


# ============================================================
# BACKTEST ENGINE
# ============================================================

@dataclass
class BacktestResult:
    equity: pd.Series
    metrics: Dict[str, float]


def compute_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def backtest(df: pd.DataFrame, signal: pd.Series, cost_bps: float = 0.5) -> BacktestResult:
    close = df["Close"]
    ret = close.pct_change().fillna(0)

    pos = signal.shift(1).fillna(0)
    gross = pos * ret

    turnover = (pos - pos.shift(1).fillna(0)).abs()
    cost = turnover * (cost_bps / 10000.0)

    net = gross - cost
    equity = (1 + net).cumprod()

    ann = 252
    cagr = equity.iloc[-1] ** (ann / len(equity)) - 1
    sharpe = (net.mean() * ann) / (net.std() * np.sqrt(ann)) if net.std() > 0 else 0
    maxdd = compute_max_drawdown(equity)

    return BacktestResult(
        equity=equity,
        metrics={
            "CAGR": cagr,
            "Sharpe": sharpe,
            "MaxDD": maxdd,
            "FinalEquity": equity.iloc[-1]
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--cost_bps", type=float, default=0.5)
    args = parser.parse_args()

    ts_utc = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M")

    print(f"\nForex Forecast – {ts_utc} UTC")
    print("=" * 100)

    for pair, ticker in PAIRS.items():
        df = fetch_ohlc(ticker, args.years)

        feats = build_features(df).dropna()
        labels = make_labels(df).loc[feats.index]

        X = feats
        y = labels

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial"))
        ])

        model.fit(X, y)

        probs = model.predict_proba(X.iloc[[-1]])[0]
        classes = model.named_steps["clf"].classes_

        prob_dict = dict(zip(classes, probs))
        best_class = classes[np.argmax(probs)]

        signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        signal = signal_map[int(best_class)]
        prob_up = prob_dict.get(1, 0.5)
        confidence = max(probs)

        regime = regime_from_price(df).iloc[-1]

        print(f"{pair:8} | Signal: {signal:5} | Conf: {confidence:.2f} | ProbUp: {prob_up:.2f} | Regime: {regime}")

        # Backtest
        preds = pd.Series(model.predict(X), index=X.index)
        bt = backtest(df.loc[preds.index], preds, args.cost_bps)

        print(f"   Backtest → CAGR: {bt.metrics['CAGR']*100:.2f}% | Sharpe: {bt.metrics['Sharpe']:.2f} | MaxDD: {bt.metrics['MaxDD']*100:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
