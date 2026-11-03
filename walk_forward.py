"""Walk-forward parameter selection with strictly out-of-sample evaluation."""

import argparse
import itertools

import numpy as np
import pandas as pd

from backtester import (backtest, load_prices, plot, sma_crossover, summarize,
                        synthetic_prices)

PARAM_GRID = {"fast": [5, 10, 20, 30, 50], "slow": [50, 100, 150, 200]}


def sharpe(returns: pd.Series, periods: int = 252) -> float:
    deviation = returns.std(ddof=1)
    return np.nan if not np.isfinite(deviation) or deviation == 0 else (
        returns.mean() / deviation * np.sqrt(periods)
    )


def grid(param_grid):
    keys = list(param_grid)
    for values in itertools.product(*(param_grid[key] for key in keys)):
        params = dict(zip(keys, values))
        if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
            continue
        yield params


def optimize(train, strategy, param_grid, cost_bps):
    candidates = list(grid(param_grid))
    if not candidates:
        raise ValueError("parameter grid contains no valid combinations")
    best, best_score = None, -np.inf
    for params in candidates:
        result = backtest(train, strategy, cost_bps=cost_bps, **params)
        score = sharpe(result["strat_ret"])
        if np.isfinite(score) and score > best_score:
            best, best_score = params, score
    if best is None:
        raise ValueError("could not select parameters: all training Sharpe scores are undefined")
    return best, best_score


def walk_forward(prices, strategy=sma_crossover, param_grid=PARAM_GRID,
                 train_days=756, test_days=126, cost_bps=5.0):
    """Fit on rolling chronological windows, then score each following test slice."""
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be positive")
    if len(prices) < train_days + test_days:
        raise ValueError("not enough prices for one complete train and test window")

    oos_parts, log = [], []
    previous_position = 0.0
    start = 0
    while start + train_days + test_days <= len(prices):
        train = prices.iloc[start:start + train_days]
        # Warm the indicators using past observations, then retain test rows only.
        history = prices.iloc[:start + train_days + test_days]
        params, train_sharpe = optimize(train, strategy, param_grid, cost_bps)
        result = backtest(history, strategy, cost_bps=cost_bps, **params)
        test_result = result.iloc[start + train_days:start + train_days + test_days].copy()
        # Carry the portfolio across parameter-retraining boundaries. The first
        # row's position was signaled at the prior close; charge only the actual
        # change from the position held at the end of the preceding test window.
        positions = test_result["position"]
        turnover = positions.diff().abs()
        turnover.iloc[0] = abs(float(positions.iloc[0]) - previous_position)
        test_result["turnover"] = turnover
        test_result["cost"] = turnover * cost_bps / 10_000
        test_result["strat_ret"] = (
            (1 + positions * test_result["asset_ret"]) * (1 - test_result["cost"]) - 1
        )
        previous_position = float(positions.iloc[-1])
        oos_parts.append(test_result)
        log.append({
            "test_start": test_result.index[0].date(),
            "test_end": test_result.index[-1].date(),
            **params,
            "train_sharpe": round(train_sharpe, 2),
            "test_sharpe": round(sharpe(test_result["strat_ret"]), 2),
        })
        start += test_days

    oos = pd.concat(oos_parts)
    oos["equity"] = (1 + oos["strat_ret"]).cumprod()
    oos["buy_hold"] = (1 + oos["asset_ret"]).cumprod()
    return oos, pd.DataFrame(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    prices = synthetic_prices() if args.synthetic else load_prices(args.ticker, args.start)
    oos, log = walk_forward(prices, sma_crossover, PARAM_GRID,
                            args.train_days, args.test_days, args.cost_bps)
    print("\n=== Window log ===")
    print(log.to_string(index=False))
    print("\n=== Out-of-sample summary ===")
    for key, value in summarize(oos).items():
        print(f"{key:<24} {value:.4f}")
    print(f"\nAvg train Sharpe: {log['train_sharpe'].mean():.2f} | "
          f"Avg test Sharpe: {log['test_sharpe'].mean():.2f}")
    if not args.no_plot:
        plot(oos, f"{args.ticker} walk-forward (out-of-sample)")
