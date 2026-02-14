# forex_forecast_backtest.py
# ------------------------------------------------------------
# Daily Forex Forecast (BUY/HOLD/SELL) + Backtest Performance
# Fully compatible with new sklearn versions
# ------------------------------------------------------------

import argparse
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# PAIR SELECTION (5 best + 1 secret)
# ============================================================

PAIRS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "EURJPY": "EURJPY=X",
}


# ============================================================
# INDICATORS
# ============================================================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


# ============================================================
# DATA LOADER (ROBUST)
# ============================================================

def fetch_ohlc(ticker, years):
    period = f"{years}y"
    df = yf.download(ticker, period=period, interval="1d", progress=False)

    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.columns = [str(c).strip().title() for c in df.columns]

    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"{ticker} missing column {col}")

    return df[required].dropna().copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    rsi14 = rsi(close)
    atr14 = atr(high, low, close) / close
    macd_line, macd_sig = macd(close)

    feats = pd.DataFrame({
        "ret1": close.pct_change(),
        "ret5": close.pct_change(5),
        "ret10": close.pct_change(10),
        "rsi14": rsi14,
        "atr14": atr14,
        "macd": macd_line,
        "macd_hist": macd_line - macd_sig,
        "dist_ma50": (close / ma50) - 1,
        "dist_ma200": (close / ma200) - 1,
    }, index=df.index)

    return feats.dropna()


def make_labels(df, threshold=0.0005):
    next_ret = df["Close"].pct_change().shift(-1)
    y = np.where(next_ret > threshold, 1,
                 np.where(next_ret < -threshold, -1, 0))
    return pd.Series(y, index=df.index)


# ============================================================
# BACKTEST
# ============================================================

@dataclass
class BacktestResult:
    cagr: float
    sharpe: float
    maxdd: float


def compute_max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1
    return dd.min()


def backtest(df, signal, cost_bps=0.5):
    ret = df["Close"].pct_change().fillna(0)
    pos = signal.shift(1).fillna(0)

    gross = pos * ret
    turnover = (pos - pos.shift(1).fillna(0)).abs()
    cost = turnover * (cost_bps / 10000)

    net = gross - cost
    equity = (1 + net).cumprod()

    ann = 252
    cagr = equity.iloc[-1] ** (ann / len(equity)) - 1
    sharpe = (net.mean() * ann) / (net.std() * np.sqrt(ann)) if net.std() > 0 else 0
    maxdd = compute_max_drawdown(equity)

    return BacktestResult(cagr, sharpe, maxdd)


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
    print("=" * 90)

    for pair, ticker in PAIRS.items():
        df = fetch_ohlc(ticker, args.years)

        feats = build_features(df)
        labels = make_labels(df).loc[feats.index]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])

        model.fit(feats, labels)

        probs = model.predict_proba(feats.iloc[[-1]])[0]
        classes = model.named_steps["clf"].classes_

        prob_map = dict(zip(classes, probs))
        best = classes[np.argmax(probs)]

        signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        signal = signal_map[int(best)]

        print(f"{pair:8} | Signal: {signal:5} | Confidence: {max(probs):.2f}")

        preds = pd.Series(model.predict(feats), index=feats.index)
        bt = backtest(df.loc[preds.index], preds, args.cost_bps)

        print(f"   Backtest → CAGR: {bt.cagr*100:.2f}% | Sharpe: {bt.sharpe:.2f} | MaxDD: {bt.maxdd*100:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
