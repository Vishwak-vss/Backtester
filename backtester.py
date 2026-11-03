from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Strategy = Callable[[pd.Series], pd.Series]


def sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    if (isinstance(fast, bool) or isinstance(slow, bool)
            or not isinstance(fast, (int, np.integer))
            or not isinstance(slow, (int, np.integer))
            or fast <= 0 or slow <= 0 or fast >= slow):
        raise ValueError("fast and slow must be positive integers (fast < slow)")
    fast_avg = prices.rolling(fast, min_periods=fast).mean()
    slow_avg = prices.rolling(slow, min_periods=slow).mean()
    return (fast_avg > slow_avg).astype(float)


def backtest(prices: pd.Series, strategy: Strategy = sma_crossover,
             cost_bps: float = 5.0, **strategy_params) -> pd.DataFrame:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if prices.empty or not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("prices must be non-empty, date-ordered, and have a unique index")
    try:
        prices = pd.to_numeric(prices, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prices must be numeric") from exc
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
        raise ValueError("prices must contain only finite, positive values")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps <= 10_000:
        raise ValueError("cost_bps must be finite and between 0 and 10,000")

    asset_ret = prices.pct_change().fillna(0.0)
    signal = strategy(prices, **strategy_params)
    if not isinstance(signal, pd.Series):
        raise TypeError("strategy must return a pandas Series")
    if signal.index.has_duplicates:
        raise ValueError("strategy signal index must be unique")
    signal = signal.reindex(prices.index)
    if (signal.isna().any() or not np.isfinite(signal.to_numpy(dtype=float)).all()
            or not signal.isin([0, 1]).all()):
        raise ValueError("strategy must return a date-aligned 0/1 position series")
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * cost_bps / 10_000
    strat_ret = (1 + position * asset_ret) * (1 - costs) - 1
    return pd.DataFrame({
        "price": prices,
        "signal": signal,
        "position": position,
        "asset_ret": asset_ret,
        "turnover": turnover,
        "cost": costs,
        "strat_ret": strat_ret,
        "equity": (1 + strat_ret).cumprod(),
    }, index=prices.index)


def load_prices(ticker: str, start: str) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance to download market data: pip install yfinance") from exc
    data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker!r} from {start}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    close.name = ticker
    return close


def synthetic_prices(days: int = 2500, seed: int = 7) -> pd.Series:
    if days < 2:
        raise ValueError("days must be at least 2")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, days)
    return pd.Series(100 * np.cumprod(1 + returns),
                     index=pd.bdate_range("2010-01-01", periods=days), name="price")


def max_drawdown(returns: pd.Series) -> float:
    if (returns.empty or not np.isfinite(returns.to_numpy(dtype=float)).all()
            or (returns < -1).any()):
        raise ValueError("returns must be non-empty, finite, and no less than -100%")
    equity = (1 + returns).cumprod()
    peaks = equity.cummax().clip(lower=1.0)
    return float((equity / peaks - 1).min())


def summarize(results: pd.DataFrame, periods: int = 252) -> dict[str, float]:
    if results.empty or "strat_ret" not in results or "asset_ret" not in results:
        raise ValueError("results must contain strat_ret and asset_ret columns")
    if periods <= 0:
        raise ValueError("periods must be positive")
    returns = results["strat_ret"]
    if (not np.isfinite(returns.to_numpy(dtype=float)).all()
            or (returns < -1).any()):
        raise ValueError("strategy returns must be finite and no less than -100%")
    equity = (1 + returns).cumprod()
    volatility = returns.std(ddof=1)
    sharpe = np.nan if not np.isfinite(volatility) or volatility == 0 else (
        returns.mean() / volatility * np.sqrt(periods)
    )
    ending_equity = float(equity.iloc[-1])
    annualized_return = (ending_equity ** (periods / len(returns)) - 1
                         if ending_equity >= 0 else np.nan)
    downside = returns.clip(upper=0).pow(2).mean() ** 0.5
    return {
        "total_return": ending_equity - 1,
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(volatility * np.sqrt(periods)),
        "sharpe": float(sharpe),
        "sortino": float(np.nan if downside == 0 else returns.mean() / downside * np.sqrt(periods)),
        "max_drawdown": max_drawdown(returns),
        "market_exposure": float((results.get("position", returns * 0 + 1)).mean()),
        "buy_hold_return": float((1 + results["asset_ret"]).prod() - 1),
    }


def plot(results: pd.DataFrame, title: str = "Backtest") -> None:
    import matplotlib.pyplot as plt

    ax = results[["equity"]].plot(figsize=(11, 5), title=title)
    buy_hold = (1 + results["asset_ret"]).cumprod()
    buy_hold.plot(ax=ax, label="Buy and hold", alpha=0.75)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    prices = synthetic_prices() if args.synthetic else load_prices(args.ticker, args.start)
    results = backtest(prices, cost_bps=args.cost_bps)
    for key, value in summarize(results).items():
        print(f"{key:<24} {value:.4f}")
    if not args.no_plot:
        plot(results, f"{args.ticker} SMA crossover") 