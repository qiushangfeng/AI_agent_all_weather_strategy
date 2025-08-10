def get_config_from_ui(main_window):
    """
    Collects all configurations from the GUI widgets.
    Takes the main_window instance as an argument to access its widgets.
    """
    asset_mapping = {}
    asset_class_map = {}

    def parse_asset_line(line_text, class_name):
        if not line_text.strip(): return
        items = line_text.replace('\n', ',').split(',')
        for item in items:
            item = item.strip()
            if ':' in item:
                name, code = item.split(':', 1)
                name, code = name.strip(), code.strip()
                if name and code:
                    asset_mapping[name] = code
                    asset_class_map[name] = class_name
            else:
                raise ValueError(f"Asset '{item}' format is incorrect. Should be 'Name:Code'")

    try:
        parse_asset_line(main_window.stock_input.text(), '股票')
        parse_asset_line(main_window.bond_input.text(), '债券')
        parse_asset_line(main_window.gold_input.text(), '黄金')
        if not asset_mapping: raise ValueError("Asset pool cannot be empty.")
    except Exception as e:
        main_window.update_log(f"Error parsing asset configuration: {e}")
        return None

    prompt_text = main_window.prompt_input.toPlainText()
    monthly_prompts = {date_str.strip(): prompt.strip() for line in prompt_text.split('\n') if ':' in line for date_str, prompt in [line.split(':', 1)]}

    api_provider = main_window.api_provider_combo.currentText()
    api_key = main_window.api_key_input.text()
    if not api_key or "..." in api_key:
         main_window.update_log(f"Error: Please enter a valid API Key."); return None

    api_configs = {
        "openai": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
        "deepseek": {"model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"}
    }
    
    if main_window.gru_checkbox.isChecked():
        ml_model_config = {'type': 'single', 'model_name': 'GRU'}
    else:
        selected_base_models = [key for key, cb in main_window.base_model_checkboxes.items() if cb.isChecked()]
        if not selected_base_models:
            main_window.update_log("Error: Please select at least one base model."); return None
        
        meta_model_text = main_window.meta_model_combo.currentText()
        meta_model_name = None if '无' in meta_model_text else meta_model_text
        
        ml_model_config = {'type': 'stacking', 'base_models': selected_base_models, 'meta_model': meta_model_name}

    model_params = {}
    try:
        model_params['lgb'] = {'n_estimators': int(main_window.lgb_n_estimators.text()), 'learning_rate': float(main_window.lgb_learning_rate.text())}
        model_params['rf'] = {'n_estimators': int(main_window.rf_n_estimators.text()), 'max_depth': int(main_window.rf_max_depth.text())}
        model_params['lr'] = {'C': float(main_window.lr_c.text())}
        model_params['gru'] = {'hidden_dim': int(main_window.gru_hidden_dim.text()), 'num_layers': int(main_window.gru_num_layers.text()), 'num_epochs': int(main_window.gru_num_epochs.text())}
    except ValueError as e:
        main_window.update_log(f"Model parameter error: Please enter valid numbers. Details: {e}"); return None

    selected_factors = [key for key, checkbox in main_window.factor_checkboxes.items() if checkbox.isChecked()]
    config = {
        'asset_mapping': asset_mapping, 'asset_class_map': asset_class_map,
        'start_date': main_window.start_date_input.date().toString("yyyyMMdd"),
        'end_date': main_window.end_date_input.date().toString("yyyyMMdd"),
        'rebalance_freq': 'ME', 'vol_window': 60, 'prediction_window': 20,
        'ml_factors': selected_factors,
        'ml_model_config': ml_model_config,
        'model_params': model_params,
        'ai_fallback_strategy': main_window.fallback_combo.currentText(),
        'monthly_prompts': monthly_prompts,
        'api_provider': api_provider, 'api_key': api_key,
        'model_name': api_configs[api_provider]['model'], 'base_url': api_configs[api_provider]['base_url'],
        'stock_rp_asset': list(asset_mapping.keys())[0] if any(v == '股票' for v in asset_class_map.values()) else None,
        'bond_rp_asset': next((k for k,v in asset_class_map.items() if v == '债券'), None)
    }
    
    if 'risk_premium' in config['ml_factors'] and (not config['stock_rp_asset'] or not config['bond_rp_asset']):
        main_window.update_log("Error: 'Stock-Bond Risk Premium' factor requires at least one stock and one bond ETF."); return None
            
    return config
