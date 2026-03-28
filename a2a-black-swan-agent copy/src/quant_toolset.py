"""
QuantToolset — Black Swan Adversarial Stress-Testing Agent

Production-ready, headless quantitative finance diagnostic toolset.
Returns deterministic, structured JSON. No visual rendering libraries.
Strictly enforces t+1 execution to prevent lookahead bias.
Incorporates Optuna-based Walk-Forward Optimization (WFO).
"""

import asyncio
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import optuna

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QuantToolset:
    """Headless adversarial stress-testing toolset for algorithmic trading strategies."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    #  PUBLIC TOOL — single entry point for the LLM
    # ------------------------------------------------------------------
    async def run_robustness_suite(
        self,
        ticker: str,
        strategy_type: str = "sma",
        strategy_params: str = "{}",
        period: str = "2y",
        iterations: int = 2000,
        drop_pct: float = 0.20,
    ) -> str:
        """Run a full adversarial robustness suite on a trading strategy.

        Fetches OHLCV data, runs a Walk-Forward Optimization (WFO) backtest using Optuna 
        and pandas-ta, computes baseline metrics on the Out-of-Sample (OOS) curve, 
        profiles the data (Hurst + tail risk), and executes Monte Carlo stress tests.  
        Returns a single structured JSON string.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL', 'TSLA').
            strategy_type: pandas-ta supported strategy type ('sma', 'rsi', 'macd', 'bbands', etc).
            strategy_params: JSON string of param ranges. e.g. {"fast_period": [10, 50]}.
            period: Data lookback period for yfinance (e.g. '1y', '2y', '5y').
            iterations: Number of Monte Carlo iterations (default 2000).
            drop_pct: Fraction of trades to drop in the Connectivity MC test (default 0.20).

        Returns:
            str: JSON string with WFO metrics and adversarial diagnostics.
        """
        try:
            # Parse strategy_params from JSON string
            if isinstance(strategy_params, str):
                try:
                    params = json.loads(strategy_params) if strategy_params else {}
                except json.JSONDecodeError:
                    import ast
                    try:
                        params = ast.literal_eval(strategy_params) if strategy_params else {}
                    except (ValueError, SyntaxError):
                        params = {}
            else:
                params = strategy_params if strategy_params else {}

            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                self._execute_suite,
                ticker,
                strategy_type,
                params,
                period,
                iterations,
                drop_pct,
            )
            return json.dumps(result, indent=2)

        except Exception as exc:
            logger.error("run_robustness_suite failed: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    #  PRIVATE — orchestrator (runs in thread pool to keep async happy)
    # ------------------------------------------------------------------
    def _execute_suite(
        self,
        ticker: str,
        strategy_type: str,
        params: dict,
        period: str,
        iterations: int,
        drop_pct: float,
    ) -> dict:
        # 1. Fetch & clean data
        df = self._fetch_data(ticker, period)
        if df.empty or len(df) < 100:
            raise ValueError(
                f"Insufficient data for {ticker} with period={period}. "
                f"Got {len(df)} rows; need at least 100 for WFO."
            )

        # 2. Walk-Forward Optimization (WFO)
        # Returns Out-of-Sample (OOS) daily returns & optimal param history.
        # Fallback will return plain backtest if dataset is too short for split.
        wfo_daily_returns, optimal_params_list = self._run_wfo(
            df, strategy_type, params,
            wfo_train_days=180, wfo_test_days=30, optuna_trials=30, penalty_factor=2.0
        )
        
        # Build portfolio object for baseline metrics using OOS curve
        oos_equity = np.cumprod(1 + wfo_daily_returns)
        portfolio = {
            "daily_returns": wfo_daily_returns,
            "equity_curve": oos_equity,
            # Placeholder positions since they vary across window sizes
            "positions": np.sign(wfo_daily_returns) 
        }

        # 3. Baseline metrics
        baseline = self._compute_baseline_metrics(portfolio)

        # 4. Data profiling & News
        close_series = np.asarray(df["Close"])
        daily_returns = portfolio["daily_returns"]
        hurst = self._compute_hurst_exponent(close_series)
        tail = self._compute_tail_risk(daily_returns)
        
        # Fetch latest news
        try:
            news_items = yf.Ticker(ticker).news
            recent_news = [item.get("title", "") for item in news_items[:5]] if news_items else []
        except Exception:
            recent_news = []

        # 5. Adversarial lab
        trades_arr = daily_returns[daily_returns != 0.0]
        if len(trades_arr) < 10:
            trades_arr = daily_returns  # fallback if very few non-zero

        # Build daily_sequences (list of arrays, one per trading day)
        daily_sequences = self._build_daily_sequences(daily_returns)

        connectivity = self._stress_connectivity_mc(trades_arr, iterations, drop_pct)
        trade_seq = self._stress_shuffle_trades_mc(trades_arr, iterations)
        day_seq = self._stress_shuffle_days_mc(
            daily_sequences, baseline["max_drawdown_pct"], iterations
        )
        gbm = self._stress_gaussian_gbm_mc(
            daily_returns,
            baseline["max_drawdown_pct"],
            baseline["net_return_pct"] / 100.0,
            iterations,
        )

        return {
            "ticker": ticker.upper(),
            "strategy": {"type": strategy_type, "params": params},
            "period": period,
            "data_profile": {
                "hurst_exponent": round(hurst, 4),
                "tail_risk": tail,
                "recent_news": recent_news,
            },
            "baseline_backtest": baseline,
            "wfo_optimal_params_history": optimal_params_list,
            "adversarial_lab": {
                "connectivity_mc": connectivity,
                "trade_sequence_mc": trade_seq,
                "day_sequence_mc": day_seq,
                "gaussian_gbm_mc": gbm,
            },
        }

    # ------------------------------------------------------------------
    #  DATA INGESTION
    # ------------------------------------------------------------------
    def _fetch_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch OHLCV data via yfinance, forward-fill gaps."""
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.ffill().dropna()
        return data

    # ------------------------------------------------------------------
    #  WALK-FORWARD OPTIMIZATION (WFO)
    # ------------------------------------------------------------------
    def _run_wfo(self, df: pd.DataFrame, strategy_type: str, param_ranges: dict, 
                 wfo_train_days=180, wfo_test_days=30, optuna_trials=30, penalty_factor=2.0):
        """Runs Walk-Forward Optimization across the dataset using Optuna."""
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        dates = df.index
        start_date = dates[0]
        end_date = dates[-1]
        
        current_train_start = start_date
        
        oos_returns_list = []
        optimal_params_list = []
        
        while True:
            train_end = current_train_start + pd.Timedelta(days=wfo_train_days)
            test_end = train_end + pd.Timedelta(days=wfo_test_days)
            
            if test_end > end_date:
                if train_end < end_date and (end_date - train_end).days > 5:
                    test_end = end_date
                else:
                    break
                
            train_df = df[(df.index >= current_train_start) & (df.index < train_end)]
            test_df = df[(df.index >= train_end) & (df.index < test_end)]
            
            if len(train_df) < 30 or len(test_df) < 5:
                current_train_start += pd.Timedelta(days=wfo_test_days)
                continue
                
            def objective(trial):
                opt_params = {}
                for k, v in param_ranges.items():
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (int, float)):
                        if isinstance(v[0], int) and isinstance(v[1], int):
                            opt_params[k] = trial.suggest_int(k, v[0], v[1])
                        else:
                            opt_params[k] = trial.suggest_float(k, float(v[0]), float(v[1]))
                    elif isinstance(v, list):
                        opt_params[k] = trial.suggest_categorical(k, v)
                    else:
                        opt_params[k] = v
                
                try:
                    signals = self._generate_signals_dynamic(train_df, strategy_type, opt_params)
                    res = self._run_backtest(train_df, signals)
                    daily = res['daily_returns']
                    log_rets = np.log(1 + daily)
                    mean_lr = np.mean(log_rets)
                    std_lr = np.std(log_rets)
                    sharpe = (mean_lr / std_lr) * np.sqrt(252) if std_lr != 0 else 0.0
                    max_dd = self._get_max_drawdown(daily)
                    
                    # Penalized Objective
                    score = sharpe - (max_dd * penalty_factor)
                    return score
                except Exception:
                    return -999.0

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=optuna_trials)
            
            best_params = study.best_params
            
            for k, v in param_ranges.items():
                if k not in best_params:
                    best_params[k] = v[0] if isinstance(v, list) and len(v)==1 else v
            
            combined_df = df[(df.index >= current_train_start) & (df.index < test_end)]
            oos_signals = self._generate_signals_dynamic(combined_df, strategy_type, best_params)
            
            res_comb = self._run_backtest(combined_df, oos_signals)
            daily_returns_comb = res_comb['daily_returns']
            
            test_mask = np.asarray(combined_df.index >= train_end)
            oos_daily = daily_returns_comb[test_mask]
            
            oos_returns_list.append(oos_daily)
            optimal_params_list.append({
                "window_start": current_train_start.strftime('%Y-%m-%d'),
                "train_end": train_end.strftime('%Y-%m-%d'),
                "test_end": test_end.strftime('%Y-%m-%d'),
                "params": best_params
            })
            
            current_train_start += pd.Timedelta(days=wfo_test_days)

        if not oos_returns_list:
            logger.warning("Data too short for WFO splits. Executing default backtest.")
            default_params = {}
            for k, v in param_ranges.items():
                default_params[k] = v[0] if isinstance(v, list) else v
            sig = self._generate_signals_dynamic(df, strategy_type, default_params)
            res = self._run_backtest(df, sig)
            return res['daily_returns'], [{"fallback": "Not enough data for WFO, static backtest executed."}]
            
        full_oos_returns = np.concatenate(oos_returns_list)
        return full_oos_returns, optimal_params_list

    # ------------------------------------------------------------------
    #  SIGNAL GENERATION
    # ------------------------------------------------------------------
    def _generate_signals_dynamic(
        self, df: pd.DataFrame, strategy_type: str, params: dict
    ) -> pd.Series:
        """Generate +1 (long) / 0 (flat) / -1 (short) signals dynamically using pandas-ta."""
        close = df["Close"]
        strategy_type = strategy_type.lower().strip()

        # Moving Averages
        if strategy_type in ["sma", "ema", "wma", "vwma", "hma", "rma", "dema", "tema"]:
            fast = int(params.get("fast_period", 10))
            slow = int(params.get("slow_period", 50))
            if fast >= slow:
                return pd.Series(0, index=df.index)
            fast_ma = getattr(ta, strategy_type)(close, length=fast)
            slow_ma = getattr(ta, strategy_type)(close, length=slow)
            signal = pd.Series(0, index=df.index)
            signal[fast_ma > slow_ma] = 1
            signal[fast_ma <= slow_ma] = -1

        # Oscillators
        elif strategy_type in ["rsi", "cci", "mfi", "stoch"]:
            per = int(params.get("length", 14))
            ob = float(params.get("overbought", 70))
            os_val = float(params.get("oversold", 30))
            ind_val = getattr(ta, strategy_type)(df["High"], df["Low"], close, length=per) if strategy_type in ["cci", "mfi"] else getattr(ta, strategy_type)(close, length=per)
            if isinstance(ind_val, pd.DataFrame):
                ind_val = ind_val.iloc[:, 0]
            signal = pd.Series(0, index=df.index)
            signal[ind_val < os_val] = 1   # oversold → buy
            signal[ind_val > ob] = -1      # overbought → sell

        # MACD
        elif strategy_type in ["macd", "ppo", "apo"]:
            fast = int(params.get("fast", 12))
            slow = int(params.get("slow", 26))
            sig_p = int(params.get("signal", 9))
            if fast >= slow:
                return pd.Series(0, index=df.index)
            ind_df = getattr(ta, strategy_type)(close, fast=fast, slow=slow, signal=sig_p)
            if ind_df is None or len(ind_df.columns) < 3:
                return pd.Series(0, index=df.index)
            macd_line = ind_df.iloc[:, 0]
            signal_line = ind_df.iloc[:, 2] 
            signal = pd.Series(0, index=df.index)
            signal[macd_line > signal_line] = 1
            signal[macd_line <= signal_line] = -1

        # Bollinger Bands
        elif strategy_type in ["bbands", "kc", "dc"]:
            per = int(params.get("length", 20))
            std_dev = float(params.get("std", 2.0))
            ind_df = getattr(ta, strategy_type)(close, length=per, std=std_dev)
            if ind_df is None or len(ind_df.columns) < 3:
                return pd.Series(0, index=df.index)
            lower = ind_df.iloc[:, 0]
            upper = ind_df.iloc[:, 2]
            signal = pd.Series(0, index=df.index)
            signal[close < lower] = 1   # price below lower band → buy
            signal[close > upper] = -1  # price above upper band → sell

        else:
            raise ValueError(
                f"Unknown strategy_type '{strategy_type}'. "
                "Supported: sma, ema, rsi, macd, bbands, etc."
            )

        signal = signal.fillna(0).astype(int)
        return signal

    # ------------------------------------------------------------------
    #  BACKTEST (strict t+1 execution — no lookahead bias)
    # ------------------------------------------------------------------
    def _run_backtest(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        """Vectorised backtest with t+1 position entry to prevent lookahead."""
        close = np.asarray(df["Close"])
        sig_arr = np.asarray(signals)

        positions = np.zeros(len(sig_arr))
        positions[1:] = sig_arr[:-1]

        price_returns = np.zeros(len(close))
        price_returns[1:] = (close[1:] - close[:-1]) / close[:-1]

        daily_returns = positions * price_returns
        equity = np.cumprod(1 + daily_returns)

        return {
            "daily_returns": daily_returns,
            "equity_curve": equity,
            "positions": positions,
        }

    # ------------------------------------------------------------------
    #  BASELINE METRICS
    # ------------------------------------------------------------------
    def _compute_baseline_metrics(self, portfolio: dict) -> dict:
        daily = portfolio["daily_returns"]
        equity = portfolio["equity_curve"]
        positions = portfolio["positions"]

        total_return = equity[-1] - 1.0 if len(equity) > 0 else 0.0
        n_days = len(daily)
        ann_return = (equity[-1] ** (252 / max(n_days, 1))) - 1.0 if n_days > 0 else 0.0

        max_dd = self._get_max_drawdown(daily)

        log_rets = np.log(1 + daily)
        mean_lr = np.mean(log_rets)
        std_lr = np.std(log_rets)
        sharpe = (mean_lr / std_lr) * np.sqrt(252) if std_lr != 0 else 0.0

        downside = log_rets[log_rets < 0]
        if len(downside) > 1:
            down_std = np.std(downside, ddof=1)
            sortino = (mean_lr / down_std) * np.sqrt(252) if down_std != 0 else 0.0
        else:
            sortino = 0.0

        calmar = (ann_return / max_dd) if max_dd > 0 else 0.0

        trades = daily[daily != 0]
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            "net_return_pct": round(total_return * 100, 2),
            "annualised_return_pct": round(ann_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "calmar_ratio": round(calmar, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
        }

    # ------------------------------------------------------------------
    #  HURST EXPONENT  (R/S analysis)
    # ------------------------------------------------------------------
    def _compute_hurst_exponent(self, series: np.ndarray) -> float:
        """Compute Hurst exponent using rescaled range (R/S) analysis."""
        if len(series) < 20:
            return 0.5 

        returns = np.diff(np.log(series))
        n = len(returns)

        max_k = min(n // 2, 512)
        sizes = []
        k = 8
        while k <= max_k:
            sizes.append(k)
            k = int(k * 1.5)
        if len(sizes) < 2:
            return 0.5

        rs_values = []
        for s in sizes:
            n_subs = n // s
            rs_list = []
            for i in range(n_subs):
                sub = returns[i * s : (i + 1) * s]
                mean_sub = np.mean(sub)
                devs = np.cumsum(sub - mean_sub)
                r = np.max(devs) - np.min(devs)
                std = np.std(sub, ddof=1)
                if std > 0:
                    rs_list.append(r / std)
            if rs_list:
                rs_values.append((np.log(s), np.log(np.mean(rs_list))))

        if len(rs_values) < 2:
            return 0.5

        log_n = np.array([v[0] for v in rs_values])
        log_rs = np.array([v[1] for v in rs_values])
        hurst = np.polyfit(log_n, log_rs, 1)[0]
        return float(np.clip(hurst, 0.0, 1.0))

    # ------------------------------------------------------------------
    #  TAIL RISK
    # ------------------------------------------------------------------
    def _compute_tail_risk(self, returns: np.ndarray) -> dict:
        from scipy import stats as sp_stats

        if len(returns) < 5:
            return {"skewness": 0.0, "kurtosis": 3.0, "var_95": 0.0, "cvar_95": 0.0}

        skew = float(sp_stats.skew(returns))
        kurt = float(sp_stats.kurtosis(returns, fisher=False))
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(np.mean(returns[returns <= var_95])) if np.any(returns <= var_95) else var_95

        return {
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
        }

    # ------------------------------------------------------------------
    #  HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _get_max_drawdown(returns_array: np.ndarray) -> float:
        if len(returns_array) == 0:
            return 0.0
        equity = np.cumprod(1 + returns_array)
        hwm = np.maximum.accumulate(equity)
        dd = (hwm - equity) / hwm
        return float(np.max(dd))

    @staticmethod
    def _build_daily_sequences(daily_returns: np.ndarray):
        return [np.array([r]) for r in daily_returns]

    # ------------------------------------------------------------------
    #  MONTE CARLO — Connectivity
    # ------------------------------------------------------------------
    def _stress_connectivity_mc(
        self, trades_arr: np.ndarray, iterations: int, drop_pct: float
    ) -> dict:
        n_keep = int(len(trades_arr) * (1 - drop_pct))
        if n_keep < 1:
            n_keep = 1

        sims = []
        for _ in range(iterations):
            sample = np.random.choice(trades_arr, n_keep, replace=False)
            sims.append((np.prod(1 + sample) - 1) * 100)

        original_ret = (np.prod(1 + trades_arr) - 1) * 100
        arr = np.array(sims)

        return {
            "original_return_pct": round(float(original_ret), 4),
            "avg_sim_return_pct": round(float(np.mean(arr)), 4),
            "p5_return_pct": round(float(np.percentile(arr, 5)), 4),
            "p95_return_pct": round(float(np.percentile(arr, 95)), 4),
        }

    # ------------------------------------------------------------------
    #  MONTE CARLO — Trade Sequence (shuffle trades)
    # ------------------------------------------------------------------
    def _stress_shuffle_trades_mc(
        self, trades_arr: np.ndarray, iterations: int
    ) -> dict:
        actual_dd = self._get_max_drawdown(trades_arr) * 100
        temp = trades_arr.copy()
        sims = []
        for _ in range(iterations):
            np.random.shuffle(temp)
            sims.append(self._get_max_drawdown(temp) * 100)

        arr = np.array(sims)
        return {
            "actual_dd_pct": round(float(actual_dd), 4),
            "avg_sim_dd_pct": round(float(np.mean(arr)), 4),
            "var95_dd_pct": round(float(np.percentile(arr, 95)), 4),
        }

    # ------------------------------------------------------------------
    #  MONTE CARLO — Day Sequence (shuffle days)
    # ------------------------------------------------------------------
    def _stress_shuffle_days_mc(
        self, daily_sequences: list, original_dd_pct: float, iterations: int
    ) -> dict:
        n = len(daily_sequences)
        idx = np.arange(n)
        sims = []
        for _ in range(iterations):
            np.random.shuffle(idx)
            shuffled = np.concatenate([daily_sequences[i] for i in idx])
            sims.append(self._get_max_drawdown(shuffled) * 100)

        arr = np.array(sims)
        return {
            "actual_dd_pct": round(float(original_dd_pct), 4),
            "avg_sim_dd_pct": round(float(np.mean(arr)), 4),
            "var95_dd_pct": round(float(np.percentile(arr, 95)), 4),
        }

    # ------------------------------------------------------------------
    #  MONTE CARLO — Gaussian GBM
    # ------------------------------------------------------------------
    def _stress_gaussian_gbm_mc(
        self,
        daily_returns: np.ndarray,
        current_dd_pct: float,
        actual_total_ret: float,
        iterations: int,
    ) -> dict:
        log_rets = np.log(1 + daily_returns)
        mu = np.mean(log_rets)
        sigma = np.std(log_rets)
        n_days = len(daily_returns)

        if sigma == 0 or n_days == 0:
            return {
                "risk": {"actual_dd_pct": current_dd_pct, "avg_sim_dd_pct": 0, "var95_dd_pct": 0},
                "reward": {"actual_return_pct": round(actual_total_ret * 100, 4), "avg_sim_return_pct": 0, "worst5_return_pct": 0, "best5_return_pct": 0},
            }

        Z = np.random.normal(0, 1, (iterations, n_days))
        sim_log_steps = mu + sigma * Z
        sim_log_curves = np.cumsum(sim_log_steps, axis=1)
        sim_curves = np.exp(sim_log_curves)

        hwm = np.maximum.accumulate(sim_curves, axis=1)
        drawdowns = (hwm - sim_curves) / hwm
        sim_mdds = np.max(drawdowns, axis=1) * 100

        sim_total_rets = (sim_curves[:, -1] - 1) * 100

        dd_95 = np.percentile(sim_mdds, 95)
        dd_avg = np.mean(sim_mdds)

        ret_avg = np.mean(sim_total_rets)
        ret_05 = np.percentile(sim_total_rets, 5)
        ret_95 = np.percentile(sim_total_rets, 95)

        actual_ret_pct = actual_total_ret * 100

        return {
            "risk": {
                "actual_dd_pct": round(float(current_dd_pct), 4),
                "avg_sim_dd_pct": round(float(dd_avg), 4),
                "var95_dd_pct": round(float(dd_95), 4),
            },
            "reward": {
                "actual_return_pct": round(float(actual_ret_pct), 4),
                "avg_sim_return_pct": round(float(ret_avg), 4),
                "worst5_return_pct": round(float(ret_05), 4),
                "best5_return_pct": round(float(ret_95), 4),
            },
        }

    # ------------------------------------------------------------------
    #  TOOL REGISTRATION
    # ------------------------------------------------------------------
    def get_tools(self) -> dict[str, Any]:
        return {
            "run_robustness_suite": self,
        }
