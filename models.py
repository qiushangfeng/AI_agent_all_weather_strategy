import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

# --- Main Public Function ---
def get_ml_signals(features, labels, config, log_emitter):
    """
    Main function to get ML signals based on the configuration.
    """
    model_config = config['ml_model_config']
    log_emitter(f"Starting model training with config: {model_config}")

    features_reset = features.reset_index().rename(columns={'index': '日期'})
    data = pd.merge(features_reset, labels, on=['日期', 'asset'], how='inner')
    
    if model_config['type'] == 'single' and model_config['model_name'] == 'GRU':
        return _train_gru(data, config, log_emitter)

    # --- Stacking/Ensemble Logic ---
    base_model_names = model_config['base_models']
    meta_model_name = model_config.get('meta_model')
    
    split_date = data['日期'].quantile(0.7, interpolation='lower')
    train_base_data = data[data['日期'] < split_date]
    train_meta_data = data[data['日期'] >= split_date]

    feature_cols = [c for c in data.columns if c not in ['日期', 'asset', 'target']]
    X_base_train = train_base_data[feature_cols]
    y_base_train = train_base_data['target']
    X_all_orig = features_reset[feature_cols]

    # Train base models and get predictions for all data
    all_meta_features = pd.DataFrame(index=features_reset.index)
    for model_name in base_model_names:
        log_emitter(f"  Training base model: {model_name}...")
        model, scaler = _train_single_model(model_name, X_base_train, y_base_train, config, log_emitter)
        
        X_pred = scaler.transform(X_all_orig) if scaler else X_all_orig
        all_meta_features[f'pred_{model_name}'] = model.predict_proba(X_pred)[:, 1]

    # Generate final signal
    if meta_model_name is None:
        log_emitter("No meta-model selected. Averaging base model predictions.")
        final_probabilities = all_meta_features.mean(axis=1)
    else:
        log_emitter(f"Training meta-model: {meta_model_name}...")
        meta_features_train = all_meta_features.loc[train_meta_data.index]
        y_meta_train = train_meta_data['target']
        
        meta_model, meta_scaler = _train_single_model(meta_model_name, meta_features_train, y_meta_train, config, log_emitter)
        
        X_meta_pred = meta_scaler.transform(all_meta_features) if meta_scaler else all_meta_features
        final_probabilities = meta_model.predict_proba(X_meta_pred)[:, 1]

    signals_df = features_reset[['日期', 'asset']].copy()
    signals_df['signal'] = final_probabilities
    return signals_df.pivot(index='日期', columns='asset', values='signal')

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