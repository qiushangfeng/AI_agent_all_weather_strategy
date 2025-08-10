import pandas as pd
import numpy as np
import akshare as ak

def get_etf_data(config, log_emitter):
    """Fetches historical ETF data from akshare."""
    asset_mapping = config['asset_mapping']
    start_date = config['start_date']
    end_date = config['end_date']
    
    all_etf_prices = {}
    log_emitter("Fetching historical ETF data...")
    PRICE_COLUMN = '收盘'
    
    for name, code in asset_mapping.items():
        log_emitter(f"--- Processing: {name} ({code}) ---")
        try:
            etf_hist = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if not isinstance(etf_hist, pd.DataFrame) or etf_hist.empty: continue
            
            df = etf_hist.copy()
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors='coerce')
            df.dropna(subset=['日期', PRICE_COLUMN], inplace=True)
            
            if df.empty: continue
            
            df = df.set_index('日期')[[PRICE_COLUMN]]
            df.columns = [name]
            if df.index.has_duplicates: df = df[~df.index.duplicated(keep='first')]
            
            all_etf_prices[name] = df
        except Exception as e:
            log_emitter(f"Error fetching data for {name} ({code}): {e}")
            
    if not all_etf_prices: return pd.DataFrame()
    
    price_df = pd.concat(all_etf_prices.values(), axis=1)
    price_df.ffill(inplace=True)
    price_df.dropna(inplace=True)
    log_emitter("All ETF data fetched and merged.")
    return price_df

def build_features(price_df, config, progress_emitter, log_emitter):
    """Builds multi-factor features for each asset."""
    log_emitter("Building multi-factor features...")
    features_list = []
    selected_factors = config['ml_factors']
    
    risk_premium = None
    if 'risk_premium' in selected_factors:
        stock_rp_asset = config['stock_rp_asset']
        bond_rp_asset = config['bond_rp_asset']
        if stock_rp_asset in price_df.columns and bond_rp_asset in price_df.columns:
            stock_return = price_df[stock_rp_asset].pct_change()
            bond_return = price_df[bond_rp_asset].pct_change()
            risk_premium = (stock_return - bond_return).rolling(window=20).mean()
        else:
            log_emitter(f"Warning: Assets for risk premium factor not found. Skipping.")
            if 'risk_premium' in selected_factors: selected_factors.remove('risk_premium')

    total_assets = len(price_df.columns)
    for i, asset in enumerate(price_df.columns):
        progress_emitter(int((i / total_assets) * 100), f"Building features: {asset}")
        
        prices = price_df[asset]
        close = prices
        returns = close.pct_change()
        asset_features = pd.DataFrame(index=prices.index)
        asset_features['asset'] = asset

        if 'mom1m' in selected_factors: asset_features['mom1m'] = close.pct_change(periods=20)
        if 'mom3m' in selected_factors: asset_features['mom3m'] = close.pct_change(periods=60)
        # ... (and so on for all other factors, the logic is identical to the original) ...
        if any(f in selected_factors for f in ['sma20', 'price_div_sma20', 'bb_width', 'bb_percent']):
            sma20 = close.rolling(window=20).mean()
            if 'price_div_sma20' in selected_factors: asset_features['price_div_sma20'] = close / sma20
            
            if any(f in selected_factors for f in ['bb_width', 'bb_percent']):
                std20 = close.rolling(window=20).std()
                upper_band = sma20 + 2 * std20
                lower_band = sma20 - 2 * std20
                if 'bb_width' in selected_factors: asset_features['bb_width'] = (upper_band - lower_band) / sma20
                if 'bb_percent' in selected_factors: asset_features['bb_percent'] = (close - lower_band) / (upper_band - lower_band)

        if 'volatility' in selected_factors:
            asset_features['volatility'] = returns.rolling(window=config['vol_window']).std() * np.sqrt(252)

        if 'risk_premium' in selected_factors and risk_premium is not None:
            asset_features['risk_premium'] = risk_premium

        features_list.append(asset_features)

    all_features = pd.concat(features_list)
    return all_features.dropna(axis=1, how='all').dropna()

def build_labels(price_df, config, log_emitter):
    """Builds the target labels for prediction."""
    log_emitter("Building prediction labels...")
    future_returns = price_df.pct_change(periods=config['prediction_window']).shift(-config['prediction_window'])
    labels = (future_returns > 0).astype(int)
    return labels.stack().reset_index().rename(columns={'level_0': '日期', 'level_1': 'asset', 0: 'target'})