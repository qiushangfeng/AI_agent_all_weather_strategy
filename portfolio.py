import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import traceback

# --- High-Level Orchestration ---
def calculate_all_weights(price_data, ml_signals_df, config, log_emitter, progress_emitter, ai_agent_func):
    """Calculates weights for all strategies and returns a dictionary."""
    log_emitter("\n" + "-"*10 + " Pre-calculating all strategy weights " + "-"*10)
    
    naive_rp_weights = calculate_naive_risk_parity(price_data, config)
    equal_budget_dict = {'股票': 1/3, '债券': 1/3, '黄金': 1/3}
    budgeted_rp_weights = calculate_budgeted_risk_daily(price_data, equal_budget_dict, config, log_emitter)
    
    total_signals = ml_signals_df.sum(axis=1)
    ml_only_weights = ml_signals_df.div(total_signals, axis=0).where(total_signals > 1e-6, 0).fillna(0)

    num_assets = len(price_data.columns)
    equal_weights_series = pd.Series(1 / num_assets, index=price_data.columns)
    equal_weights = pd.DataFrame(index=price_data.index, columns=price_data.columns)
    equal_weights = equal_weights.apply(lambda x: equal_weights_series, axis=1)

    fallback_options = {
        "当前择时模型": ml_only_weights,
        "朴素风险平价": naive_rp_weights,
        "风险预算(均衡)": budgeted_rp_weights,
        "等权重": equal_weights
    }
    
    ai_driven_weights = generate_ai_weights(price_data, ml_signals_df, fallback_options, config, log_emitter, progress_emitter, ai_agent_func)
    if ai_driven_weights.empty: raise ValueError("Failed to generate AI-driven weights.")

    all_weights = {
        "AI Agent驱动策略": ai_driven_weights,
        "机器学习(纯择时)": ml_only_weights,
        "风险预算(均衡)": budgeted_rp_weights,
        "朴素风险平价(波动率倒数)": naive_rp_weights,
        "等权重基准": equal_weights
    }
    for name, df in all_weights.items():
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            
    return all_weights

def run_all_backtests(price_data, all_weights, log_emitter, progress_emitter):
    """Runs backtests for all strategies and returns a dictionary of portfolio values."""
    log_emitter("\n" + "-"*10 + " Running Backtests " + "-"*10)
    all_portfolios = {}
    for name, weights_df in all_weights.items():
        if name == "等权重基准":
            all_portfolios[name] = backtest_equal_weight(price_data)
        else:
            all_portfolios[name] = backtest_strategy(price_data, weights_df, name, progress_emitter)
    return all_portfolios

# --- Specific Weight Calculation Functions ---
def calculate_naive_risk_parity(price_df, config):
    returns = price_df.pct_change()
    vols = returns.rolling(window=config['vol_window']).std() * np.sqrt(252)
    inv_vols = 1 / (vols + 1e-6)
    weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)
    return weights.dropna()

def calculate_budgeted_risk_daily(price_df, risk_budget_dict, config, log_emitter):
    log_emitter(f"Calculating daily weights for budget: '{list(risk_budget_dict.keys())}'...")
    all_weights_list = []
    rebalance_dates = price_df.resample('ME').first().index
    
    for date in tqdm(rebalance_dates, desc="Risk Budget Weighting"):
        current_price_data = price_df.loc[:date]
        if len(current_price_data) < 60: continue
        
        weights = calculate_budgeted_risk_weights(current_price_data, risk_budget_dict, config, log_emitter)
        weights.name = date
        all_weights_list.append(weights)
        
    if not all_weights_list: return pd.DataFrame()
    
    weights_df = pd.concat(all_weights_list, axis=1).T
    full_date_range = pd.date_range(start=weights_df.index.min(), end=price_df.index.max(), freq='D')
    return weights_df.reindex(full_date_range).fillna(method='ffill').dropna()

def calculate_budgeted_risk_weights(price_df, risk_budget_dict, config, log_emitter):
    asset_to_class = config['asset_class_map']
    class_counts = pd.Series(asset_to_class).value_counts()
    budget_vector = pd.Series(index=price_df.columns, dtype=float)
    for asset, asset_class in asset_to_class.items():
        if asset in budget_vector.index:
            budget_vector[asset] = risk_budget_dict.get(asset_class, 0) / class_counts.get(asset_class, 1)
    budget_vector.fillna(0, inplace=True)
    if budget_vector.sum() == 0:
        return pd.Series(1 / len(price_df.columns), index=price_df.columns)
    budget_vector = budget_vector / budget_vector.sum()
    
    returns = price_df.pct_change().dropna()
    if len(returns) < 60:
        log_emitter(f"Warning: History length ({len(returns)}) is insufficient, using equal weights temporarily.")
        return pd.Series(1 / len(price_df.columns), index=price_df.columns)
        
    cov_matrix = returns.cov()
    return solve_risk_budgeting_weights(cov_matrix, budget_vector)

def solve_risk_budgeting_weights(cov_matrix, budgets):
    n = len(budgets)
    w0 = np.ones(n) / n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bnds = tuple((0, 1) for _ in range(n))
    
    def objective(w, cov_matrix, budgets):
        portfolio_variance = w.T @ cov_matrix @ w
        if portfolio_variance < 1e-12: return 0.0
        mrc = (cov_matrix @ w) / np.sqrt(portfolio_variance)
        rc = w * mrc
        rc_percent = rc / np.sum(rc)
        return np.sum((rc_percent - budgets)**2)
        
    solution = minimize(objective, w0, args=(cov_matrix, budgets), method='SLSQP', bounds=bnds, constraints=cons, options={'ftol': 1e-10, 'maxiter': 1000})
    return pd.Series(solution.x if solution.success else w0, index=budgets.index)

def generate_ai_weights(price_data, ml_signals_df, fallback_options, config, log_emitter, progress_emitter, ai_agent_func):
    log_emitter("\n--- Generating AI-Driven Dynamic Weights ---")
    all_weights_list = []
    rebalance_dates = price_data.resample(config['rebalance_freq']).first().index
    fallback_strategy_name = config['ai_fallback_strategy']
    
    is_fallback_to_ml = (fallback_strategy_name == "当前择时模型") and (not config['monthly_prompts'])
    if is_fallback_to_ml:
        log_emitter("Info: No prompt provided, using ML-only weights directly.")
        return fallback_options.get(fallback_strategy_name, pd.DataFrame())

    for i, date in enumerate(rebalance_dates):
        progress_emitter(int((i / len(rebalance_dates)) * 100), f"AI Analysis: {date.strftime('%Y-%m-%d')}")
        current_price_data = price_data.loc[:date]
        if len(current_price_data) < 60: continue

        current_macro_desc = config['monthly_prompts'].get(date.strftime('%Y-%m'), "").strip()
        ai_risk_budget = ai_agent_func(current_macro_desc, ['股票', '债券', '黄金']) if current_macro_desc else None
        
        weights_for_this_date = None
        if ai_risk_budget:
            log_emitter(f"AI generated weights for {date.strftime('%Y-%m-%d')}")
            base_weights = calculate_budgeted_risk_weights(current_price_data, ai_risk_budget, config, log_emitter)
            
            ml_signal = ml_signals_df.asof(date)
            if ml_signal is not None and not ml_signal.isna().all():
                aligned_base, aligned_signal = base_weights.align(ml_signal, join='left', fill_value=0.5)
                adjustment_factor = 2 * (aligned_signal - 0.5)
                target_exposure = 1 + adjustment_factor
                final_weights = (aligned_base * target_exposure).clip(lower=0)
                weights_for_this_date = final_weights / final_weights.sum() if final_weights.sum() > 1e-6 else aligned_base
            else:
                weights_for_this_date = base_weights
        else:
            log_emitter(f"AI failed or no prompt, using fallback: {fallback_strategy_name}")
            fallback_df = fallback_options.get(fallback_strategy_name)
            if fallback_df is not None and not fallback_df.loc[:date].empty:
                weights_for_this_date = fallback_df.asof(date)
        
        if weights_for_this_date is not None:
            weights_for_this_date.name = date
            all_weights_list.append(weights_for_this_date)

    if not all_weights_list: return pd.DataFrame()
    ai_driven_weights = pd.concat(all_weights_list, axis=1).T
    full_date_range = pd.date_range(start=ai_driven_weights.index.min(), end=price_data.index.max(), freq='D')
    return ai_driven_weights.reindex(full_date_range).fillna(method='ffill').dropna()

# --- Backtesting and Performance Functions ---
def backtest_strategy(price_df, weights_df, strategy_name, progress_emitter):
    portfolio_value = pd.Series(index=price_df.index, dtype=float)
    common_start_date = max(price_df.index.min(), weights_df.index.min())
    price_df_aligned = price_df.loc[common_start_date:]
    weights_df_aligned = weights_df.loc[common_start_date:]
    if price_df_aligned.empty or weights_df_aligned.empty: return pd.Series(dtype=float)
    
    portfolio_value.iloc[0] = 1.0
    for i in range(1, len(price_df_aligned)):
        progress_emitter(int((i / len(price_df_aligned)) * 100), f"Backtesting: {strategy_name}")
        current_date, prev_date = price_df_aligned.index[i], price_df_aligned.index[i-1]
        current_weights = weights_df_aligned.asof(current_date)
        daily_returns = price_df_aligned.loc[current_date] / price_df_aligned.loc[prev_date] - 1
        portfolio_return = (daily_returns * current_weights).sum()
        portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_return)
        
    return portfolio_value.dropna()

def backtest_equal_weight(price_df):
    num_assets = len(price_df.columns)
    weights = [1/num_assets] * num_assets
    daily_returns = price_df.pct_change().dropna()
    if daily_returns.empty: return pd.Series(dtype=float)
    benchmark_returns = (daily_returns * weights).sum(axis=1)
    benchmark_value = (1 + benchmark_returns).cumprod()
    return benchmark_value / benchmark_value.iloc[0] if not benchmark_value.empty else benchmark_value

def analyze_and_plot_performance(performance_dict, canvas, log_emitter):
    """Calculates, prints, and plots performance metrics on the given canvas."""
    results = pd.DataFrame()
    for name, series in performance_dict.items():
        if series.empty: continue
        total_return = series.iloc[-1] / series.iloc[0] - 1
        days = (series.index[-1] - series.index[0]).days
        annual_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
        daily_returns = series.pct_change().dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 1e-6 else 0
        rolling_max = series.expanding(min_periods=1).max()
        drawdown = (series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < -1e-6 else 0
        results[name] = [f"{total_return:.2%}", f"{annual_return:.2%}", f"{annual_volatility:.2%}",
                         f"{sharpe_ratio:.2f}", f"{max_drawdown:.2%}", f"{calmar_ratio:.2f}"]
    results.index = ["Total Return", "Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"]
    
    log_emitter("\n" + "="*20 + " Strategy Performance Analysis " + "="*20)
    log_emitter(results.to_string())
    log_emitter("="*60)
    
    try:
        canvas.axes.cla()
        valid_series = {k: v for k, v in performance_dict.items() if not v.empty}
        if not valid_series: 
            log_emitter("No valid strategy data to plot."); return

        start_date = max(s.index.min() for s in valid_series.values())
        end_date = min(s.index.max() for s in valid_series.values())
        colors = plt.cm.get_cmap('viridis', len(valid_series))
        
        for i, (name, series) in enumerate(valid_series.items()):
            plot_series = series.loc[start_date:end_date]
            plot_series = plot_series / plot_series.iloc[0]
            plot_series.plot(ax=canvas.axes, label=name, color=colors(i), linewidth=2.2 if i == 0 else 1.6, linestyle='-' if i < 2 else '--')

        canvas.axes.set_title('Multi-Strategy Equity Curve Comparison', fontsize=16, weight='bold')
        canvas.axes.set_ylabel('Cumulative Value (Log Scale)', fontsize=12)
        canvas.axes.set_xlabel('Date', fontsize=12)
        canvas.axes.legend(fontsize=11)
        canvas.axes.grid(True, which='both', linestyle='--', linewidth=0.5)
        canvas.axes.set_yscale('log')
        canvas.axes.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
        canvas.axes.set_xlim(start_date, end_date)
        canvas.figure.autofmt_xdate(rotation=15, ha='right')
        canvas.figure.tight_layout()
        canvas.draw()
        log_emitter("Chart plotting complete.")
    except Exception as e:
        error_msg = f"Error during plotting: {e}\n{traceback.format_exc()}"
        log_emitter(error_msg)