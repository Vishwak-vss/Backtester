"""
Walk-forward strategy tester (uses backtester.py in the same folder).

For each window:
  1. Grid-search strategy parameters on the TRAIN slice (maximize Sharpe).
  2. Apply the best parameters to the next TEST slice, which the optimizer never saw.
Stitch all TEST slices together -> an out-of-sample equity curve.

If the out-of-sample result is much worse than the in-sample result,
the strategy is probably overfit.

Usage:
    python test_strategy.py --ticker AAPL --start 2010-01-01
    python test_strategy.py --synthetic --no-plot
"""

import argparse
import itertools
import numpy as np
import pandas as pd

from backtester import (backtest, sma_crossover, load_prices,
                        synthetic_prices, summarize, max_drawdown)

PARAM_GRID = {
    "fast": [5, 10, 20, 30, 50],
    "slow": [50, 100, 150, 200],
}


def sharpe(r, periods=252):
    sd = r.std()
    return np.nan if sd == 0 or np.isnan(sd) else r.mean() / sd * np.sqrt(periods)


def grid(param_grid):
    keys = list(param_grid)
    for combo in itertools.product(*param_grid.values()):
        p = dict(zip(keys, combo))
        if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
            continue
        yield p


def optimize(train, strategy, param_grid, cost_bps):
    best, best_score = None, -np.inf
    for p in grid(param_grid):
        res = backtest(train, strategy, cost_bps=cost_bps, **p)
        s = sharpe(res["strat_ret"])
        if not np.isnan(s) and s > best_score:
            best, best_score = p, s
    return best, best_score


def walk_forward(prices, strategy, param_grid, train_days=756, test_days=126,
                 cost_bps=5.0):
    """train_days ~3y, test_days ~6 months (trading days)."""
    oos_parts, log = [], []
    start = 0
    while start + train_days + test_days <= len(prices):
        train = prices.iloc[start: start + train_days]
        # include the train tail so indicators are warmed up at the test start
        warm_test = prices.iloc[start: start + train_days + test_days]

        params, in_sharpe = optimize(train, strategy, param_grid, cost_bps)
        res = backtest(warm_test, strategy, cost_bps=cost_bps, **params)
        test_res = res.iloc[train_days:]  # only the unseen part

        oos_parts.append(test_res)
        log.append({
            "test_start": test_res.index[0].date(),
            "test_end": test_res.index[-1].date(),
            **params,
            "train_sharpe": round(in_sharpe, 2),
            "test_sharpe": round(sharpe(test_res["strat_ret"]), 2),
        })
        start += test_days

    oos = pd.concat(oos_parts)
    oos["equity"] = (1 + oos["strat_ret"]).cumprod()
    oos["buy_hold"] = (1 + oos["asset_ret"]).cumprod()
    return oos, pd.DataFrame(log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--train-days", type=int, default=756)
    ap.add_argument("--test-days", type=int, default=126)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    prices = synthetic_prices(2500) if a.synthetic else load_prices(a.ticker, a.start)
    oos, log = walk_forward(prices, sma_crossover, PARAM_GRID,
                            a.train_days, a.test_days, a.cost_bps)

    print("\n=== Window log ===")
    print(log.to_string(index=False))
    print("\n=== Out-of-sample summary ===")
    for k, v in summarize(oos).items():
        print(f"{k:<20} {v}")
    print(f"\nAvg train Sharpe: {log['train_sharpe'].mean():.2f} | "
          f"Avg test Sharpe: {log['test_sharpe'].mean():.2f}")
    print("A big gap between those two numbers is the signature of overfitting.")

    if not a.no_plot:
        from backtester import plot
        plot(oos, f"{a.ticker} walk-forward (out-of-sample only)")