# Backtesting engine

A small daily-price backtester with a long/flat SMA crossover strategy, transaction costs, reproducible synthetic data, and walk-forward parameter selection.

## Setup

The core engine requires Python, NumPy, and pandas. Add `yfinance` to download market data and `matplotlib` to display charts:

```powershell
pip install numpy pandas yfinance matplotlib
```

## Run a single backtest

```powershell
python backtester.py --synthetic --no-plot
python backtester.py --ticker AAPL --start 2015-01-01
```

## Run walk-forward evaluation

Each rolling training window selects SMA lengths by Sharpe ratio. The selected parameters are then applied to the following, unseen test window. Only test-window returns are combined in the reported out-of-sample results, and the portfolio position and turnover costs carry across test-window boundaries.

```powershell
python walk_forward.py --synthetic --no-plot
python walk_forward.py --ticker MSFT --start 2010-01-01 --train-days 756 --test-days 126
```

`--cost-bps` sets the estimated cost per unit of turnover in basis points. Costs are applied multiplicatively to returns and must be between 0 and 10,000 basis points. Defaults are 5 basis points, 756 training trading days, and 126 test trading days.

## Strategy interface

A strategy receives the close-price Series and optional keyword parameters, and returns a date-aligned Series of `0` (flat) or `1` (long). Signals are shifted by one bar before returns are applied, so a close-based signal cannot earn the return that occurred before it was observed. Transaction costs are charged on position changes.

Summary metrics include Sharpe, Sortino, maximum drawdown from the initial $1 balance, market exposure, and buy-and-hold return. Market-data downloads require an internet connection. `--synthetic` runs offline with a fixed random seed.
# Backtester
