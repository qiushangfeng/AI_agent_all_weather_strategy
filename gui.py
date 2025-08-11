# gui.py
import os
import pandas as pd
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLineEdit, QLabel, QTextEdit, QDateEdit,
                             QGridLayout, QGroupBox, QCheckBox, QComboBox, QScrollArea)
from PyQt6.QtCore import QDate
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import from our new modules
from utils import setup_matplotlib_font
from config_manager import get_config_from_ui
from strategy_runner import StrategyRunner
from portfolio import analyze_and_plot_performance

# --- 1. Translation Dictionary ---
TRANSLATIONS = {
    'en': {
        "window_title": "AI-Driven All-Weather Strategy Backtesting Platform",
        "asset_pool_group": "Asset Pool Configuration (Name:Code, comma-separated)",
        "stock_etfs_label": "Stock ETFs:",
        "bond_etfs_label": "Bond ETFs:",
        "gold_etfs_label": "Gold/Commodity ETFs:",
        "time_range_group": "Backtest Time Range",
        "start_date_label": "Start Date:",
        "end_date_label": "End Date:",
        "ai_prompt_group": "AI Agent Monthly Prompt (Format YYYY-MM: Description)",
        "holdings_query_group": "Holdings Query",
        "query_date_label": "Query Date:",
        "query_button": "Query Holdings for Date",
        "ml_factors_group": "Machine Learning Factor Selection",
        "factor_cat_momentum": "Momentum",
        "factor_cat_volatility": "Volatility",
        "factor_cat_ma": "Moving Averages (MA)",
        "factor_cat_bbands": "Bollinger Bands (BBands)",
        "factor_cat_oscillators": "Oscillators",
        "factor_cat_cross_asset": "Cross-Asset",
        "select_all_checkbox": "Select/Deselect All",
        "stacking_group": "Model Stacking Configuration",
        "base_model_group": "Layer 1: Select Base Models (multi-select)",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "meta_model_group": "Layer 2: Select Meta-Model (single-select)",
        "meta_model_label": "Meta-Model:",
        "meta_model_none": "None (Average Base Models)",
        "gru_checkbox": "Use standalone GRU model (overrides stacking)",
        "params_group": "Model Parameter Customization",
        "api_group": "API Configuration",
        "provider_label": "Provider:",
        "api_key_label": "API Key:",
        "fallback_label": "AI Fail Fallback:",
        "run_button": "Run Backtest",
        "running_button": "Running...",
        "language_button": "切换中文",
    },
    'zh': {
        "window_title": "AI驱动的全天候策略回测平台",
        "asset_pool_group": "资产池配置 (名称:代码, 逗号分隔)",
        "stock_etfs_label": "股票类ETF:",
        "bond_etfs_label": "债券类ETF:",
        "gold_etfs_label": "黄金/商品ETF:",
        "time_range_group": "回测时间范围",
        "start_date_label": "开始日期:",
        "end_date_label": "结束日期:",
        "ai_prompt_group": "AI Agent 月度Prompt (格式 YYYY-MM: 描述)",
        "holdings_query_group": "持仓查询",
        "query_date_label": "查询日期:",
        "query_button": "查询当日持仓",
        "ml_factors_group": "机器学习因子选择",
        "factor_cat_momentum": "动量 (Momentum)",
        "factor_cat_volatility": "波动率 (Volatility)",
        "factor_cat_ma": "移动平均线 (MA)",
        "factor_cat_bbands": "布林带 (BBands)",
        "factor_cat_oscillators": "震荡指标 (Oscillators)",
        "factor_cat_cross_asset": "跨资产 (Cross-Asset)",
        "select_all_checkbox": "全选/反选",
        "stacking_group": "模型组合 (Stacking) 配置",
        "base_model_group": "第一层：选择基模型 (可多选)",
        "logistic_regression": "逻辑回归",
        "random_forest": "随机森林",
        "meta_model_group": "第二层：选择元模型 (单选)",
        "meta_model_label": "元模型:",
        "meta_model_none": "无 (基模型平均)",
        "gru_checkbox": "使用独立的GRU模型 (将覆盖以上组合)",
        "params_group": "模型参数自定义",
        "api_group": "API配置",
        "provider_label": "服务商:",
        "api_key_label": "API Key:",
        "fallback_label": "AI失败备选:",
        "run_button": "运行回测",
        "running_button": "正在运行...",
        "language_button": "Switch to English",
    }
}

class MatplotlibCanvas(FigureCanvas):
    """A PyQt widget for embedding a matplotlib chart."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        self.setParent(parent)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = 'en'  # Default language
        self.setWindowTitle(TRANSLATIONS[self.current_lang]["window_title"])
        self.setGeometry(100, 100, 1600, 900)
        self.all_weights_history = {}
        setup_matplotlib_font()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.init_ui()
        self.retranslate_ui() # Apply initial language

    def init_ui(self):
        # This method now only creates widgets and layouts, without setting text.
        # --- Left Panel ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(450)
        
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        self.control_layout = QVBoxLayout(content_widget)

        # Language Switch Button
        self.lang_button = QPushButton()
        self.lang_button.clicked.connect(self.toggle_language)
        self.control_layout.addWidget(self.lang_button)

        # Asset Configuration
        self.asset_group = QGroupBox()
        asset_layout = QGridLayout()
        self.stock_input = QLineEdit("沪深300:510300, 科创50:588000, 纳斯达克100:513100")
        self.bond_input = QLineEdit("10年国债:511260")
        self.gold_input = QLineEdit("黄金:518880")
        self.stock_etfs_label = QLabel()
        self.bond_etfs_label = QLabel()
        self.gold_etfs_label = QLabel()
        asset_layout.addWidget(self.stock_etfs_label, 0, 0); asset_layout.addWidget(self.stock_input, 0, 1)
        asset_layout.addWidget(self.bond_etfs_label, 1, 0); asset_layout.addWidget(self.bond_input, 1, 1)
        asset_layout.addWidget(self.gold_etfs_label, 2, 0); asset_layout.addWidget(self.gold_input, 2, 1)
        self.asset_group.setLayout(asset_layout)
        self.control_layout.addWidget(self.asset_group)
        
        # Backtest Time Range
        self.time_group = QGroupBox()
        time_layout = QGridLayout()
        self.start_date_input = QDateEdit(QDate(2018, 1, 1)); self.start_date_input.setCalendarPopup(True)
        self.end_date_input = QDateEdit(QDate.currentDate()); self.end_date_input.setCalendarPopup(True)
        self.start_date_label = QLabel()
        self.end_date_label = QLabel()
        time_layout.addWidget(self.start_date_label, 0, 0); time_layout.addWidget(self.start_date_input, 0, 1)
        time_layout.addWidget(self.end_date_label, 1, 0); time_layout.addWidget(self.end_date_input, 1, 1)
        self.time_group.setLayout(time_layout)
        self.control_layout.addWidget(self.time_group)
        
        # AI Agent Prompt
        self.ai_group = QGroupBox()
        ai_layout = QVBoxLayout()
        self.prompt_input = QTextEdit(
            '2018-04: US-China trade friction begins...\n'
            '2024-01: AI boom drives tech stocks higher...'
        )
        ai_layout.addWidget(self.prompt_input)
        self.ai_group.setLayout(ai_layout)
        self.control_layout.addWidget(self.ai_group)

        # Holdings Query
        self.holding_group = QGroupBox()
        holding_layout = QGridLayout()
        self.holding_date_input = QDateEdit(QDate.currentDate())
        self.holding_date_input.setCalendarPopup(True)
        self.query_button = QPushButton()
        self.query_button.clicked.connect(self.query_holdings)
        self.query_date_label = QLabel()
        holding_layout.addWidget(self.query_date_label, 0, 0)
        holding_layout.addWidget(self.holding_date_input, 0, 1)
        holding_layout.addWidget(self.query_button, 1, 0, 1, 2)
        self.holding_group.setLayout(holding_layout)
        self.control_layout.addWidget(self.holding_group)

        # Machine Learning Factor Selection
        self.ml_group = QGroupBox()
        ml_layout = QVBoxLayout()
        self.factor_structure = {
            "factor_cat_momentum": ['mom1m', 'mom3m', 'mom6m'],
            "factor_cat_volatility": ['volatility', 'inv_vol', 'atr'],
            "factor_cat_ma": ['sma20', 'sma60', 'price_div_sma20', 'price_div_sma60'],
            "factor_cat_bbands": ['bb_width', 'bb_percent'],
            "factor_cat_oscillators": ['rsi', 'macd_diff', 'roc', 'cci'],
            "factor_cat_cross_asset": ['risk_premium']
        }
        self.factor_checkboxes = {}
        self.factor_category_groups = {}
        for category_key, factors in self.factor_structure.items():
            category_group = QGroupBox()
            self.factor_category_groups[category_key] = category_group
            category_layout = QGridLayout()
            select_all_checkbox = QCheckBox(); select_all_checkbox.setChecked(True)
            select_all_checkbox.stateChanged.connect(lambda state, f=factors: self.toggle_factor_group(state, f))
            category_layout.addWidget(select_all_checkbox, 0, 0, 1, 2)
            row, col = 1, 0
            for factor_key in factors:
                checkbox = QCheckBox(factor_key); checkbox.setChecked(True)
                self.factor_checkboxes[factor_key] = checkbox
                category_layout.addWidget(checkbox, row, col)
                col += 1
                if col > 1: col = 0; row += 1
            category_group.setLayout(category_layout)
            ml_layout.addWidget(category_group)
        self.ml_group.setLayout(ml_layout)
        self.control_layout.addWidget(self.ml_group)

        # Model Stacking Configuration
        self.stacking_group = QGroupBox()
        stacking_layout = QVBoxLayout()
        self.base_model_group = QGroupBox()
        base_model_layout = QGridLayout()
        self.base_model_checkboxes = {
            'logistic_regression': QCheckBox(),
            'random_forest': QCheckBox(),
            'LightGBM': QCheckBox("LightGBM"), # Keep original text for keys
        }
        self.base_model_checkboxes['logistic_regression'].setChecked(True)
        self.base_model_checkboxes['LightGBM'].setChecked(True)
        for i, (key, checkbox) in enumerate(self.base_model_checkboxes.items()):
            base_model_layout.addWidget(checkbox, 0, i)
        self.base_model_group.setLayout(base_model_layout)
        stacking_layout.addWidget(self.base_model_group)
        self.meta_model_group = QGroupBox()
        meta_model_layout = QHBoxLayout()
        self.meta_model_combo = QComboBox()
        self.meta_model_label = QLabel()
        meta_model_layout.addWidget(self.meta_model_label)
        meta_model_layout.addWidget(self.meta_model_combo)
        self.meta_model_group.setLayout(meta_model_layout)
        stacking_layout.addWidget(self.meta_model_group)
        self.gru_checkbox = QCheckBox()
        stacking_layout.addWidget(self.gru_checkbox)
        self.stacking_group.setLayout(stacking_layout)
        self.control_layout.addWidget(self.stacking_group)
        
        # Model Parameter Customization
        self.params_group = QGroupBox()
        params_layout = QGridLayout()
        self.lgb_label = QLabel("<b>LightGBM:</b>")
        self.lgb_n_est_label = QLabel("n_estimators:")
        self.lgb_lr_label = QLabel("learning_rate:")
        self.rf_label = QLabel("<b>Random Forest:</b>")
        self.rf_n_est_label = QLabel("n_estimators:")
        self.rf_depth_label = QLabel("max_depth:")
        self.lr_label = QLabel("<b>Logistic Regression:</b>")
        self.lr_c_label = QLabel("C:")
        self.gru_label = QLabel("<b>GRU:</b>")
        self.gru_hidden_label = QLabel("hidden_dim:")
        self.gru_layers_label = QLabel("num_layers:")
        self.gru_epochs_label = QLabel("num_epochs:")
        self.lgb_n_estimators = QLineEdit("100")
        self.lgb_learning_rate = QLineEdit("0.1")
        self.rf_n_estimators = QLineEdit("100")
        self.rf_max_depth = QLineEdit("10")
        self.lr_c = QLineEdit("1.0")
        self.gru_hidden_dim = QLineEdit("32")
        self.gru_num_layers = QLineEdit("2")
        self.gru_num_epochs = QLineEdit("10")
        # Layouting... (This is tedious, but needs to be done once)
        params_layout.addWidget(self.lgb_label, 0, 0, 1, 4)
        params_layout.addWidget(self.lgb_n_est_label, 1, 0); params_layout.addWidget(self.lgb_n_estimators, 1, 1)
        params_layout.addWidget(self.lgb_lr_label, 1, 2); params_layout.addWidget(self.lgb_learning_rate, 1, 3)
        # ... and so on for all param labels and inputs
        self.params_group.setLayout(params_layout)
        self.control_layout.addWidget(self.params_group)
        
        # API Configuration
        self.api_group = QGroupBox()
        api_layout = QGridLayout()
        self.api_provider_combo = QComboBox(); self.api_provider_combo.addItems(["deepseek", "openai"])
        self.api_key_input = QLineEdit(); self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.fallback_combo = QComboBox() # Items will be set in retranslate_ui
        self.provider_label = QLabel()
        self.api_key_label = QLabel()
        self.fallback_label = QLabel()
        api_layout.addWidget(self.provider_label, 0, 0); api_layout.addWidget(self.api_provider_combo, 0, 1)
        api_layout.addWidget(self.api_key_label, 1, 0); api_layout.addWidget(self.api_key_input, 1, 1)
        api_layout.addWidget(self.fallback_label, 2, 0); api_layout.addWidget(self.fallback_combo, 2, 1)
        self.api_group.setLayout(api_layout)
        self.control_layout.addWidget(self.api_group)

        # Run Button
        self.run_button = QPushButton()
        self.run_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.run_button.clicked.connect(self.run_strategy)
        self.control_layout.addWidget(self.run_button)
        self.control_layout.addStretch()

        # --- Right Panel: Results ---
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        self.canvas = MatplotlibCanvas(self, width=12, height=6, dpi=100)
        result_layout.addWidget(self.canvas)
        self.log_output = QTextEdit(); self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier New", 9)); self.log_output.setFixedHeight(200)
        result_layout.addWidget(self.log_output)
        
        self.main_layout.addWidget(scroll_area)
        self.main_layout.addWidget(result_panel)
    
    def toggle_language(self):
        self.current_lang = 'zh' if self.current_lang == 'en' else 'en'
        self.retranslate_ui()

    def retranslate_ui(self):
        lang = self.current_lang
        tr = TRANSLATIONS[lang]

        self.setWindowTitle(tr["window_title"])
        self.lang_button.setText(tr["language_button"])
        
        # Update all widget texts
        self.asset_group.setTitle(tr["asset_pool_group"])
        self.stock_etfs_label.setText(tr["stock_etfs_label"])
        self.bond_etfs_label.setText(tr["bond_etfs_label"])
        self.gold_etfs_label.setText(tr["gold_etfs_label"])
        self.time_group.setTitle(tr["time_range_group"])
        self.start_date_label.setText(tr["start_date_label"])
        self.end_date_label.setText(tr["end_date_label"])
        self.ai_group.setTitle(tr["ai_prompt_group"])
        self.holding_group.setTitle(tr["holdings_query_group"])
        self.query_date_label.setText(tr["query_date_label"])
        self.query_button.setText(tr["query_button"])
        self.ml_group.setTitle(tr["ml_factors_group"])
        
        # Update factor category group titles
        for category_key, group_box in self.factor_category_groups.items():
            group_box.setTitle(tr[category_key])
            # Also update the "Select All" checkbox inside each group
            select_all_cb = group_box.layout().itemAt(0).widget()
            select_all_cb.setText(tr["select_all_checkbox"])
            
        self.stacking_group.setTitle(tr["stacking_group"])
        self.base_model_group.setTitle(tr["base_model_group"])
        self.base_model_checkboxes['logistic_regression'].setText(tr['logistic_regression'])
        self.base_model_checkboxes['random_forest'].setText(tr['random_forest'])
        # LightGBM text doesn't change, so we don't need to translate it
        
        self.meta_model_group.setTitle(tr["meta_model_group"])
        self.meta_model_label.setText(tr["meta_model_label"])
        self.meta_model_combo.clear()
        self.meta_model_combo.addItems([tr['meta_model_none'], tr['logistic_regression'], 'LightGBM'])

        self.gru_checkbox.setText(tr["gru_checkbox"])
        self.params_group.setTitle(tr["params_group"])
        self.api_group.setTitle(tr["api_group"])
        self.provider_label.setText(tr["provider_label"])
        self.api_key_label.setText(tr["api_key_label"])
        self.fallback_label.setText(tr["fallback_label"])
        self.run_button.setText(tr["run_button"])

        # Update fallback combo box items (this requires a bit more care)
        # We store the original English text to use as a key
        original_fallback_items = ["当前择时模型", "朴素风险平价", "风险预算(均衡)", "等权重"]
        fallback_translations_en = ["ML-Only Timing", "Naive Risk Parity", "Budgeted Risk Parity (Balanced)", "Equal Weight"]
        fallback_translations_zh = ["当前择时模型", "朴素风险平价", "风险预算(均衡)", "等权重"]
        
        current_selection = self.fallback_combo.currentData() # get stored key
        self.fallback_combo.clear()
        
        items_to_add = fallback_translations_en if lang == 'en' else fallback_translations_zh
        for i, text in enumerate(items_to_add):
            self.fallback_combo.addItem(text, original_fallback_items[i]) # text, data (key)

        if current_selection:
            index = self.fallback_combo.findData(current_selection)
            if index != -1:
                self.fallback_combo.setCurrentIndex(index)

    def toggle_factor_group(self, state, factors_in_group):
        is_checked = (state == 2)
        for factor_key in factors_in_group:
            if factor_key in self.factor_checkboxes:
                self.factor_checkboxes[factor_key].setChecked(is_checked)

    def run_strategy(self):
        # Update run button text based on current language
        self.run_button.setText(TRANSLATIONS[self.current_lang]["running_button"])
        
        # The get_config_from_ui will now need to handle translated combo box text
        config = get_config_from_ui(self)
        if not config: 
            self.run_button.setText(TRANSLATIONS[self.current_lang]["run_button"])
            return
        
        self.log_output.clear(); self.canvas.axes.clear(); self.canvas.draw()
        self.all_weights_history = {}
        
        self.worker = StrategyRunner(config)
        self.worker.log_signal.connect(self.update_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finish)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def update_progress(self, percent, message):
        self.run_button.setText(f"{TRANSLATIONS[self.current_lang]['running_button']} ({percent}%)")

    def on_finish(self, final_results):
        """
        回测完成后的槽函数 (最终修正版)。
        """
        performance_dict = final_results.get('portfolios', {})
        weights_dict = final_results.get('weights', {})
        self.all_weights_history = weights_dict
        
        self.run_button.setEnabled(True)
        self.run_button.setText("运行回测")
        self.update_log("\n回测全部完成！正在生成性能报告和图表...")
        self.analyze_and_plot_performance(performance_dict)

        save_dir = "holdings_data"
        os.makedirs(save_dir, exist_ok=True)
        for strategy_name, weights_df in weights_dict.items():
            filename = f"{strategy_name.replace(' ', '_').replace('(', '').replace(')', '')}_{pd.Timestamp.now().strftime('%Y%m%d%H%M')}.csv"
            filepath = os.path.join(save_dir, filename)
            weights_df.to_csv(filepath)
            self.update_log(f"持仓数据已保存至：{filepath}")

    def query_holdings(self):
        """查询并显示指定日期的持仓"""
        if not self.all_weights_history:
            self.update_log("\n[查询失败]：请先运行一次回测以生成持仓数据。")
            return
            
        query_date = pd.to_datetime(self.holding_date_input.date().toString("yyyy-MM-dd"))
        self.update_log(f"\n" + "="*20 + f" {query_date.strftime('%Y-%m-%d')} 持仓查询 " + "="*20)
        
        holdings_data = {}
        for name, weights_df in self.all_weights_history.items():
            if not weights_df.empty and query_date >= weights_df.index.min():
                daily_holding = weights_df.asof(query_date)
                if daily_holding is not None:
                    holdings_data[name] = daily_holding
        
        if not holdings_data:
            self.update_log(f"在日期 {query_date.strftime('%Y-%m-%d')} 未找到任何策略的持仓数据。")
            return
            
        holding_df = pd.DataFrame(holdings_data).applymap(lambda x: f"{x:.2%}" if pd.notna(x) else "0.00%")
        self.update_log(holding_df.to_string())
        self.update_log("="*60)

    def update_log(self, message):
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def on_error(self, message):
        self.run_button.setEnabled(True)
        self.run_button.setText(TRANSLATIONS[self.current_lang]["run_button"])
        self.update_log(f"\n!!! AN ERROR OCCURRED !!!\n{message}")

