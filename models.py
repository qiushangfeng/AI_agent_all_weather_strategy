import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- GRU Model Definition ---
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def get_ml_signals(features, labels, config, log_emitter):
    """
    Generates ML signals using Walk-Forward Optimization.
    """
    model_config = config['ml_model_config']
    log_emitter(f"Starting Walk-Forward model training with config: {model_config}")

    # --- Data Preparation ---
    features_reset = features.reset_index().rename(columns={'index': '日期'})
    data = pd.merge(features_reset, labels, on=['日期', 'asset'], how='inner')
    data = data.sort_values(by='日期').reset_index(drop=True)
    
    # --- Walk-Forward Settings ---
    is_gru = model_config.get('type') == 'single' and model_config.get('model_name') == 'GRU'
    retrain_freq = '3MS' if is_gru else 'MS' # GRU retrains every 3 months, others every month
    initial_train_months = 12

    all_dates = data['日期'].unique()
    rebalance_dates = pd.Series(pd.to_datetime(all_dates)).resample(retrain_freq).first().dropna()
    
    if len(rebalance_dates) <= initial_train_months:
        raise ValueError(f"Data length is too short for walk-forward with {initial_train_months}-month initial training.")
    
    # Set initial training period.
    initial_train_end_date = rebalance_dates[initial_train_months]
    rebalance_dates = rebalance_dates[rebalance_dates >= initial_train_end_date].values
    
    all_period_signals = []
    
    # --- Walk-Forward Loop ---
    for i in tqdm(range(len(rebalance_dates)), desc=f"Walk-Forward Training ({model_config.get('model_name', 'Stacking')})"):
        train_end_date = rebalance_dates[i]
        pred_start_date = train_end_date
        pred_end_date = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else all_dates[-1]

        log_emitter(f"\n--- Training period ends: {pd.to_datetime(train_end_date).strftime('%Y-%m-%d')} ---")
        log_emitter(f"--- Predicting for period: {pd.to_datetime(pred_start_date).strftime('%Y-%m-%d')} to {pd.to_datetime(pred_end_date).strftime('%Y-%m-%d')} ---")

        train_data = data[data['日期'] < train_end_date]
        predict_data = data[(data['日期'] >= pred_start_date) & (data['日期'] < pred_end_date)]

        if train_data.empty or predict_data.empty:
            log_emitter("Skipping period due to insufficient data.")
            continue

        # CHOOSE THE CORRECT TRAINING/PREDICTION FUNCTION BASED ON MODEL TYPE
        if is_gru:
            period_signals = _train_and_predict_gru_fold(train_data, predict_data, config, log_emitter)
        else:
            period_signals = _train_and_predict_sklearn_fold(train_data, predict_data, config, log_emitter)
            
        all_period_signals.append(period_signals)

    # --- Concatenate all out-of-sample signals ---
    if not all_period_signals:
        raise ValueError("Walk-forward training did not produce any signals.")
        
    final_signals_long = pd.concat(all_period_signals)
    final_signals_wide = final_signals_long.pivot(index='日期', columns='asset', values='signal')
    
    return final_signals_wide

def _train_and_predict_sklearn_fold(train_data, predict_data, config, log_emitter):
    """
    Trains sklearn-style models (Stacking) on one fold and predicts the next.
    """
    model_config = config['ml_model_config']
    base_model_names = model_config['base_models']
    meta_model_name = model_config.get('meta_model')

    feature_cols = [c for c in train_data.columns if c not in ['日期', 'asset', 'target']]
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_predict = predict_data[feature_cols]

    base_model_predictions = pd.DataFrame(index=X_predict.index)
    for model_name in base_model_names:
        model, scaler = _train_single_model(model_name, X_train, y_train, config, log_emitter)
        X_pred_scaled = scaler.transform(X_predict) if scaler else X_predict
        base_model_predictions[f'pred_{model_name}'] = model.predict_proba(X_pred_scaled)[:, 1]
    
    if meta_model_name is None:
        final_probabilities = base_model_predictions.mean(axis=1)
    else:
        base_model_preds_on_train = pd.DataFrame(index=X_train.index)
        for model_name in base_model_names:
             model, scaler = _train_single_model(model_name, X_train, y_train, config, log_emitter)
             X_train_scaled = scaler.transform(X_train) if scaler else X_train
             base_model_preds_on_train[f'pred_{model_name}'] = model.predict_proba(X_train_scaled)[:, 1]
        
        meta_model, meta_scaler = _train_single_model(meta_model_name, base_model_preds_on_train, y_train, config, log_emitter)
        X_meta_pred = meta_scaler.transform(base_model_predictions) if meta_scaler else base_model_predictions
        final_probabilities = meta_model.predict_proba(X_meta_pred)[:, 1]

    period_signals_df = predict_data[['日期', 'asset']].copy()
    period_signals_df['signal'] = final_probabilities.values
    return period_signals_df

def _train_and_predict_gru_fold(train_data, predict_data, config, log_emitter):
    """
    Trains a GRU model on one fold and predicts the next.
    """
    feature_cols = [c for c in train_data.columns if c not in ['日期', 'asset', 'target']]
    gru_params = config.get('model_params', {}).get('gru', {})
    sequence_length = 20 # How many past days of data to use for one prediction

    # 1. Prepare data for GRU
    # We need to combine train and predict data for scaling, then split them back
    combined_data = pd.concat([train_data, predict_data])
    scaler = StandardScaler().fit(combined_data[feature_cols])
    
    train_scaled = train_data.copy(); train_scaled[feature_cols] = scaler.transform(train_data[feature_cols])
    predict_scaled = predict_data.copy(); predict_scaled[feature_cols] = scaler.transform(predict_data[feature_cols])

    # a) Create training sequences
    X_train_seq, y_train_seq = [], []
    for asset, group in train_scaled.groupby('asset'):
        features = group[feature_cols].values
        target = group['target'].values
        for i in range(len(group) - sequence_length):
            X_train_seq.append(features[i:i+sequence_length])
            y_train_seq.append(target[i+sequence_length-1])
            
    X_train_tensor = torch.tensor(np.array(X_train_seq), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train_seq), dtype=torch.float32).view(-1, 1)

    # b) Create prediction sequences (needs to look back into training data)
    X_predict_seq, predict_info = [], []
    full_scaled_data_for_pred = pd.concat([train_scaled, predict_scaled])
    for asset, group in predict_scaled.groupby('asset'):
        # Find the full history for this asset to construct sequences
        full_asset_history = full_scaled_data_for_pred[full_scaled_data_for_pred['asset'] == asset]
        for i, row in group.iterrows():
            # Find the position of the current prediction row in the full history
            loc = full_asset_history.index.get_loc(i)
            if loc >= sequence_length:
                start_idx = loc - sequence_length
                X_predict_seq.append(full_asset_history.iloc[start_idx:loc][feature_cols].values)
                predict_info.append(row[['日期', 'asset']])

    X_predict_tensor = torch.tensor(np.array(X_predict_seq), dtype=torch.float32)

    # 2. Train GRU model
    model = GRUModel(
        input_dim=len(feature_cols), 
        hidden_dim=gru_params.get('hidden_dim', 32), 
        num_layers=gru_params.get('num_layers', 2), 
        output_dim=1
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    num_epochs = gru_params.get('num_epochs', 10)
    for epoch in range(num_epochs):
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    # 3. Predict signals for the period
    model.eval()
    with torch.no_grad():
        predictions = model(X_predict_tensor).numpy().flatten()
        
    # 4. Assemble results
    period_signals_df = pd.DataFrame(predict_info)
    period_signals_df['signal'] = predictions
    
    return period_signals_df

# --- Helper Functions ---
def _train_single_model(model_name, X_train, y_train, config, log_emitter):
    """Helper to train a single sklearn-style model and return it with its scaler."""
    model = _get_model_instance(model_name, config, log_emitter)
    scaler = None
    if isinstance(model, LogisticRegression):
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    return model, scaler

def _get_model_instance(model_name, config, log_emitter):
    """Returns a model instance with parameters from the config."""
    params = config.get('model_params', {})
    if model_name == 'LightGBM':
        lgb_params = params.get('lgb', {})
        log_emitter(f"  LightGBM Params: {lgb_params}")
        return lgb.LGBMClassifier(random_state=42, verbosity=-1, **lgb_params)
    elif model_name == '随机森林':
        rf_params = params.get('rf', {})
        log_emitter(f"  Random Forest Params: {rf_params}")
        return RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params)
    elif model_name == '逻辑回归':
        lr_params = params.get('lr', {})
        log_emitter(f"  Logistic Regression Params: {lr_params}")
        return LogisticRegression(random_state=42, solver='liblinear', **lr_params)
    raise ValueError(f"Unknown model instance name: {model_name}")

def _train_gru(data, config, log_emitter):
    """Trains the GRU model and returns signals."""
    feature_cols = [c for c in data.columns if c not in ['日期', 'asset', 'target']]
    log_emitter("Preparing sequence data for GRU model...")
    gru_params = config.get('model_params', {}).get('gru', {})
    hidden_dim = gru_params.get('hidden_dim', 32)
    num_layers = gru_params.get('num_layers', 2)
    num_epochs = gru_params.get('num_epochs', 10)
    sequence_length = 20
    
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    X_seq, y_seq, dates_assets = [], [], []
    for asset, group in data_scaled.groupby('asset'):
        features = group[feature_cols].values
        target = group['target'].values
        date_index = group[['日期', 'asset']]
        for i in range(len(group) - sequence_length):
            X_seq.append(features[i:i+sequence_length])
            y_seq.append(target[i+sequence_length])
            dates_assets.append(date_index.iloc[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

    log_emitter(f"Training GRU model (Params: hidden_dim={hidden_dim}, layers={num_layers}, epochs={num_epochs})...")
    model = GRUModel(input_dim=len(feature_cols), hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(num_epochs):
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log_emitter(f"GRU Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    log_emitter("GRU training complete, generating signals...")
    model.eval()
    with torch.no_grad():
        all_predictions = model(X_tensor).numpy().flatten()
    
    signals_df = pd.DataFrame(dates_assets)
    signals_df['signal'] = all_predictions
    return signals_df.pivot(index='日期', columns='asset', values='signal')
