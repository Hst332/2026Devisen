# forex_forecast_backtest.py
# ------------------------------------------------------------
# Daily Forex Forecast (BUY/HOLD/SELL) + Backtest Performance
# Data: Yahoo Finance via yfinance (e.g., "EURUSD=X")
#
# Install:
#   pip install pandas numpy scikit-learn yfinance matplotlib
#
# Run:
#   python forex_forecast_backtest.py
#
# Optional args:
#   python forex_forecast_backtest.py --years 12 --cost_bps 0.5 --hold_threshold 0.0005
# ------------------------------------------------------------

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt


# -----------------------------
# Pair selection
# -----------------------------
PAIRS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",        # Yahoo uses JPY=X as USDJPY
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",        # Yahoo uses CAD=X as USDCAD
    "EURJPY": "EURJPY=X",     # Geheimtipp
}


# -----------------------------
# Indicators (no external TA libs)
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


# -----------------------------
# Feature engineering
# -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df expected columns: Open, High, Low, Close, Volume (Volume may be missing for FX)
    returns features dataframe aligned with df index
    """
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

    # trend / regime proxy features
    ma_slope = ma50.pct_change(5)  # 1-week slope proxy
    dist_ma200 = (close / ma200) - 1.0

    feats = pd.DataFrame({
        "ret1": ret1,
        "ret5": ret5,
        "ret10": ret10,
        "vol20": vol20,
        "rsi14": rsi14,
        "atr14": atr14 / close,      # normalized ATR
        "macd": macd_line,
        "macd_sig": macd_sig,
        "macd_hist": (macd_line - macd_sig),
        "dist_ma20": (close / ma20) - 1.0,
        "dist_ma50": (close / ma50) - 1.0,
        "dist_ma200": dist_ma200,
        "ma50_slope": ma_slope,
    }, index=df.index)

    return feats


def make_labels(df: pd.DataFrame, hold_threshold: float = 0.0005) -> pd.Series:
    """
    Next-day direction labels:
      +1 = BUY  (next-day return > +threshold)
       0 = HOLD (abs(next-day return) <= threshold)
      -1 = SELL (next-day return < -threshold)
    """
    next_ret = df["Close"].pct_change().shift(-1)
    y = pd.Series(np.where(next_ret > hold_threshold, 1,
                           np.where(next_ret < -hold_threshold, -1, 0)),
                  index=df.index)
    return y


def regime_from_price(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    ma200 = close.rolling(200).mean()
    dist = (close / ma200) - 1.0

    # simple regime buckets
    reg = pd.Series("neutral", index=df.index)
    reg[dist > 0.01] = "bull"
    reg[dist < -0.01] = "bear"
    return reg


# -----------------------------
# Data loader
# -----------------------------
def fetch_ohlc(ticker: str, years: int) -> pd.DataFrame:
    period = f"{years}y"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for ticker {ticker}")
    # normalize columns
    df = df.rename(columns={c: c.title() for c in df.columns})
    # Sometimes yfinance returns multi-index columns; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    needed = ["Open", "High", "Low", "Close"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column {c} for {ticker}")
    return df.dropna().copy()


# -----------------------------
# Backtest
# -----------------------------
@dataclass
class BacktestResult:
    equity: pd.Series
    daily_strategy_ret: pd.Series
    metrics: Dict[str, float]


def compute_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def backtest_signals(df: pd.DataFrame, signal: pd.Series, cost_bps: float = 0.5) -> BacktestResult:
    """
    signal: -1,0,+1 (position held for next day close-to-close)
    We apply simple transaction cost when position changes (in bps of notional).
    """
    close = df["Close"]
    ret = close.pct_change().fillna(0.0)

    pos = signal.shift(1).fillna(0.0)  # enter at today's close for tomorrow's return
    gross = pos * ret

    # transaction cost when position changes (turnover)
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    cost = turnover * (cost_bps / 10000.0)

    net = gross - cost
    equity = (1.0 + net).cumprod()

    # metrics
    ann_factor = 252.0
    cagr = float(equity.iloc[-1] ** (ann_factor / max(1, len(equity))) - 1.0)
    vol = float(net.std() * math.sqrt(ann_factor))
    sharpe = float((net.mean() * ann_factor) / (net.std() * math.sqrt(ann_factor))) if net.std() > 1e-12 else 0.0
    maxdd = compute_max_drawdown(equity)

    hit = net[net != 0]
    hit_rate = float((hit > 0).mean()) if len(hit) else 0.0

    metrics = {
        "CAGR": cagr,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": maxdd,
        "HitRate": hit_rate,
        "FinalEquity": float(equity.iloc[-1]),
        "TradesApprox": float((turnover > 0).sum()),
    }

    return BacktestResult(equity=equity, daily_strategy_ret=net, metrics=metrics)


# -----------------------------
# Modeling / Forecast
# -----------------------------
def train_and_forecast_one_pair(df: pd.DataFrame, hold_threshold: float) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    feats = build_features(df)
    y = make_labels(df, hold_threshold=hold_threshold)
    reg = regime_from_price(df)

    data = feats.join(df[["Close"]]).copy()
    data["y"] = y
    data["regime"] = reg

    # drop warmup NaNs and last row (label unknown)
    data = data.dropna()
    data = data.iloc[:-1].copy()

    X = data[feats.columns]
    ycls = data["y"].astype(int)

    # Model: multinomial logistic regression (stable baseline)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            multi_class="multinomial",
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            n_jobs=None
        ))
    ])

    # Fit on full history (for live daily forecast you can add walk-forward later)
    model.fit(X, ycls)

    # Predict probabilities for all dates (for backtest)
    proba = model.predict_proba(X)
    classes = model.named_steps["lr"].classes_
    proba_df = pd.DataFrame(proba, index=X.index, columns=[f"p_{c}" for c in classes])

    # Map to signal by max-prob class
    pred_class = proba_df.idxmax(axis=1).str.replace("p_", "").astype(int)
    signal = pred_class.rename("signal")

    # For "ProbUp": probability of +1 class; if class missing (rare), use 0.5 fallback
    p_up = proba_df.get("p_1", pd.Series(0.5, index=proba_df.index))

    # Prepare per-day output
    out = pd.DataFrame(index=X.index)
    out["Close"] = df.loc[out.index, "Close"]
    out["PrevClose"] = out["Close"].shift(1)
    out["DeltaPct"] = (out["Close"] / out["PrevClose"] - 1.0) * 100.0
    out["Regime"] = data["regime"]
    out["ProbUp"] = p_up
    out["Signal"] = signal.map({1: "BUY", 0: "HOLD", -1: "SELL"})
    out["Conf"] = proba_df.max(axis=1)

    # latest row for next-day forecast (use most recent available feature row)
    latest_feats = build_features(df).dropna().iloc[[-1]]
    latest_proba = model.predict_proba(latest_feats)[0]
    latest = dict(zip(classes, latest_proba))
    latest_signal_class = int(classes[np.argmax(latest_proba)])
    latest_signal = {1: "BUY", 0: "HOLD", -1: "SELL"}[latest_signal_class]
    latest_conf = float(np.max(latest_proba))
    latest_prob_up = float(latest.get(1, 0.5))

    latest_forecast = {
        "signal": latest_signal,
        "conf": latest_conf,
        "prob_up": latest_prob_up
    }

    return out, signal, latest_forecast


def pretty_print_forecast(ts_utc: str, rows: List[Dict]) -> None:
    print(f"\nForex Forecasts – {ts_utc} UTC")
    print("=" * 110)
    header = f"{'Pair':8} | {'Prev Close':10} | {'Current':10} | {'Δ %':6} | {'Signal':5} | {'Conf':4} | {'Regime':7} | {'ProbUp':6}"
    print(header)
    print("-" * 110)
    for r in rows:
        print(f"{r['pair']:8} | {r['prev_close']:10.5f} | {r['current']:10.5f} | {r['delta_pct']:6.2f} | {r['signal']:5} | {r['conf']:4.2f} | {r['regime']:7} | {r['prob_up']:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=12, help="History length in years")
    ap.add_argument("--cost_bps", type=float, default=0.5, help="Transaction cost in basis points per position change")
    ap.add_argument("--hold_threshold", type=float, default=0.0005, help="Hold band threshold for next-day returns (e.g. 0.0005 = 5 bps)")
    ap.add_argument("--plot", action="store_true", help="Plot equity curves")
    args = ap.parse_args()

    ts_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")

    forecast_rows = []
    results = {}

    for pair, ticker in PAIRS.items():
        df = fetch_ohlc(ticker, years=args.years)

        # Train + in-sample probs for signal backtest
        perday, signal, latest = train_and_forecast_one_pair(df, hold_threshold=args.hold_threshold)

        # Backtest
        bt = backtest_signals(df.loc[signal.index], signal, cost_bps=args.cost_bps)
        results[pair] = bt

        # Latest "report row"
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else float("nan")
        delta_pct = (last_close / prev_close - 1.0) * 100.0 if prev_close == prev_close else float("nan")
        regime = regime_from_price(df).iloc[-1]

        forecast_rows.append({
            "pair": pair,
            "prev_close": prev_close,
            "current": last_close,
            "delta_pct": delta_pct,
            "signal": latest["signal"],
            "conf": latest["conf"],
            "regime": regime,
            "prob_up": latest["prob_up"],
        })

    # Print Forecast Table
    pretty_print_forecast(ts_utc, forecast_rows)

    # Print Performance Summary
    print("\nBacktest Summary (daily close-to-close, ML signals, costs applied on position change)")
    print("=" * 110)
    print(f"{'Pair':8} | {'CAGR':8} | {'Sharpe':7} | {'MaxDD':8} | {'HitRate':8} | {'FinalEq':8} | {'Trades':6}")
    print("-" * 110)
    for pair, bt in results.items():
        m = bt.metrics
        print(f"{pair:8} | {m['CAGR']*100:7.2f}% | {m['Sharpe']:7.2f} | {m['MaxDrawdown']*100:7.2f}% | {m['HitRate']*100:7.2f}% | {m['FinalEquity']:8.2f} | {m['TradesApprox']:6.0f}")

    # Plot
    if args.plot:
        plt.figure()
        for pair, bt in results.items():
            plt.plot(bt.equity.index, bt.equity.values, label=pair)
        plt.title("Equity Curves (normalized)")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
