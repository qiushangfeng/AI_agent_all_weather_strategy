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
        self.setWindowTitle("AI-Driven All-Weather Strategy Backtesting Platform")
        self.setGeometry(100, 100, 1600, 900)
        self.all_weights_history = {}
        setup_matplotlib_font()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.init_ui()

    def init_ui(self):
        # --- Left Panel ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(450)
        
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        control_layout = QVBoxLayout(content_widget)

        # Asset Configuration
        asset_group = QGroupBox("Asset Pool Configuration (Name:Code, comma-separated)")
        asset_layout = QGridLayout()
        self.stock_input = QLineEdit("沪深300:510300, 科创50:588000, 纳斯达克100:513100")
        self.bond_input = QLineEdit("10年国债:511260")
        self.gold_input = QLineEdit("黄金:518880")
        asset_layout.addWidget(QLabel("Stock ETFs:"), 0, 0); asset_layout.addWidget(self.stock_input, 0, 1)
        asset_layout.addWidget(QLabel("Bond ETFs:"), 1, 0); asset_layout.addWidget(self.bond_input, 1, 1)
        asset_layout.addWidget(QLabel("Gold/Commodity ETFs:"), 2, 0); asset_layout.addWidget(self.gold_input, 2, 1)
        asset_group.setLayout(asset_layout)
        control_layout.addWidget(asset_group)
        
        # Backtest Time Range
        time_group = QGroupBox("Backtest Time Range")
        time_layout = QGridLayout()
        self.start_date_input = QDateEdit(QDate(2018, 1, 1)); self.start_date_input.setCalendarPopup(True)
        self.end_date_input = QDateEdit(QDate.currentDate()); self.end_date_input.setCalendarPopup(True)
        time_layout.addWidget(QLabel("Start Date:"), 0, 0); time_layout.addWidget(self.start_date_input, 0, 1)
        time_layout.addWidget(QLabel("End Date:"), 1, 0); time_layout.addWidget(self.end_date_input, 1, 1)
        time_group.setLayout(time_layout)
        control_layout.addWidget(time_group)
        
        # AI Agent Prompt
        ai_group = QGroupBox("AI Agent Monthly Prompt (Format YYYY-MM: Description)")
        ai_layout = QVBoxLayout()
        self.prompt_input = QTextEdit(
            '2018-04: US-China trade friction begins...\n'
            '2024-01: AI boom drives tech stocks higher...'
        )
        ai_layout.addWidget(self.prompt_input)
        ai_group.setLayout(ai_layout)
        control_layout.addWidget(ai_group)

        # Holdings Query
        holding_group = QGroupBox("Holdings Query")
        holding_layout = QGridLayout()
        self.holding_date_input = QDateEdit(QDate.currentDate())
        self.holding_date_input.setCalendarPopup(True)
        self.query_button = QPushButton("Query Holdings for Date")
        self.query_button.clicked.connect(self.query_holdings)
        holding_layout.addWidget(QLabel("Query Date:"), 0, 0)
        holding_layout.addWidget(self.holding_date_input, 0, 1)
        holding_layout.addWidget(self.query_button, 1, 0, 1, 2)
        holding_group.setLayout(holding_layout)
        control_layout.addWidget(holding_group)

        # Machine Learning Factor Selection
        ml_group = QGroupBox("Machine Learning Factor Selection")
        ml_layout = QVBoxLayout()
        self.factor_structure = {
            "Momentum": ['mom1m', 'mom3m', 'mom6m'],
            "Volatility": ['volatility', 'inv_vol', 'atr'],
            "Moving Averages (MA)": ['sma20', 'sma60', 'price_div_sma20', 'price_div_sma60'],
            "Bollinger Bands (BBands)": ['bb_width', 'bb_percent'],
            "Oscillators": ['rsi', 'macd_diff', 'roc', 'cci'],
            "Cross-Asset": ['risk_premium']
        }
        self.factor_checkboxes = {}
        for category, factors in self.factor_structure.items():
            category_group = QGroupBox(category)
            category_layout = QGridLayout()
            select_all_checkbox = QCheckBox("Select/Deselect All"); select_all_checkbox.setChecked(True)
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
        ml_group.setLayout(ml_layout)
        control_layout.addWidget(ml_group)

        # Model Stacking Configuration
        stacking_group = QGroupBox("Model Stacking Configuration")
        stacking_layout = QVBoxLayout()
        
        base_model_group = QGroupBox("Layer 1: Select Base Models (multi-select)")
        base_model_layout = QGridLayout()
        self.base_model_checkboxes = {
            '逻辑回归': QCheckBox("Logistic Regression"),
            '随机森林': QCheckBox("Random Forest"),
            'LightGBM': QCheckBox("LightGBM"),
        }
        self.base_model_checkboxes['逻辑回归'].setChecked(True)
        self.base_model_checkboxes['LightGBM'].setChecked(True)
        for i, (key, checkbox) in enumerate(self.base_model_checkboxes.items()):
            base_model_layout.addWidget(checkbox, 0, i)
        base_model_group.setLayout(base_model_layout)
        stacking_layout.addWidget(base_model_group)
        
        meta_model_group = QGroupBox("Layer 2: Select Meta-Model (single-select)")
        meta_model_layout = QHBoxLayout()
        self.meta_model_combo = QComboBox()
        self.meta_model_combo.addItems(['无 (基模型平均)', '逻辑回归', 'LightGBM'])
        meta_model_layout.addWidget(QLabel("Meta-Model:"))
        meta_model_layout.addWidget(self.meta_model_combo)
        meta_model_group.setLayout(meta_model_layout)
        stacking_layout.addWidget(meta_model_group)
        
        self.gru_checkbox = QCheckBox("Use standalone GRU model (overrides stacking)")
        stacking_layout.addWidget(self.gru_checkbox)
        stacking_group.setLayout(stacking_layout)
        control_layout.addWidget(stacking_group)
        
        # Model Parameter Customization
        params_group = QGroupBox("Model Parameter Customization")
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("<b>LightGBM:</b>"), 0, 0, 1, 4)
        params_layout.addWidget(QLabel("n_estimators:"), 1, 0); self.lgb_n_estimators = QLineEdit("100"); params_layout.addWidget(self.lgb_n_estimators, 1, 1)
        params_layout.addWidget(QLabel("learning_rate:"), 1, 2); self.lgb_learning_rate = QLineEdit("0.1"); params_layout.addWidget(self.lgb_learning_rate, 1, 3)
        params_layout.addWidget(QLabel("<b>Random Forest:</b>"), 2, 0, 1, 4)
        params_layout.addWidget(QLabel("n_estimators:"), 3, 0); self.rf_n_estimators = QLineEdit("100"); params_layout.addWidget(self.rf_n_estimators, 3, 1)
        params_layout.addWidget(QLabel("max_depth:"), 3, 2); self.rf_max_depth = QLineEdit("10"); params_layout.addWidget(self.rf_max_depth, 3, 3)
        params_layout.addWidget(QLabel("<b>Logistic Regression:</b>"), 4, 0, 1, 4)
        params_layout.addWidget(QLabel("C:"), 5, 0); self.lr_c = QLineEdit("1.0"); params_layout.addWidget(self.lr_c, 5, 1)
        params_layout.addWidget(QLabel("<b>GRU:</b>"), 6, 0, 1, 4)
        params_layout.addWidget(QLabel("hidden_dim:"), 7, 0); self.gru_hidden_dim = QLineEdit("32"); params_layout.addWidget(self.gru_hidden_dim, 7, 1)
        params_layout.addWidget(QLabel("num_layers:"), 7, 2); self.gru_num_layers = QLineEdit("2"); params_layout.addWidget(self.gru_num_layers, 7, 3)
        params_layout.addWidget(QLabel("num_epochs:"), 8, 0); self.gru_num_epochs = QLineEdit("10"); params_layout.addWidget(self.gru_num_epochs, 8, 1)
        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)
        
        # API Configuration
        api_group = QGroupBox("API Configuration")
        api_layout = QGridLayout()
        self.api_provider_combo = QComboBox(); self.api_provider_combo.addItems(["deepseek", "openai"])
        self.api_key_input = QLineEdit(); self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.fallback_combo = QComboBox(); self.fallback_combo.addItems(["当前择时模型", "朴素风险平价", "风险预算(均衡)", "等权重"])
        api_layout.addWidget(QLabel("Provider:"), 0, 0); api_layout.addWidget(self.api_provider_combo, 0, 1)
        api_layout.addWidget(QLabel("API Key:"), 1, 0); api_layout.addWidget(self.api_key_input, 1, 1)
        api_layout.addWidget(QLabel("AI Fail Fallback:"), 2, 0); api_layout.addWidget(self.fallback_combo, 2, 1)
        api_group.setLayout(api_layout)
        control_layout.addWidget(api_group)

        # Run Button
        self.run_button = QPushButton("Run Backtest")
        self.run_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.run_button.clicked.connect(self.run_strategy)
        control_layout.addWidget(self.run_button)
        
        control_layout.addStretch()

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

    def toggle_factor_group(self, state, factors_in_group):
        is_checked = (state == 2)
        for factor_key in factors_in_group:
            if factor_key in self.factor_checkboxes:
                self.factor_checkboxes[factor_key].setChecked(is_checked)

    def run_strategy(self):
        config = get_config_from_ui(self)
        if not config: return
        
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")
        self.log_output.clear()
        self.canvas.axes.clear()
        self.canvas.draw()
        self.all_weights_history = {}
        
        self.worker = StrategyRunner(config)
        self.worker.log_signal.connect(self.update_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finish)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def update_progress(self, percent, message):
        self.run_button.setText(f"Running... ({percent}%)")

    def on_finish(self, final_results):
        performance_dict = final_results.get('portfolios', {})
        self.all_weights_history = final_results.get('weights', {})
        
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Backtest")
        self.update_log("\nBacktest complete! Generating performance report and chart...")
        
        analyze_and_plot_performance(performance_dict, self.canvas, self.update_log)

        save_dir = "holdings_data"
        os.makedirs(save_dir, exist_ok=True)
        for strategy_name, weights_df in self.all_weights_history.items():
            if not weights_df.empty:
                safe_name = strategy_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')
                filename = f"{safe_name}_{pd.Timestamp.now().strftime('%Y%m%d%H%M')}.csv"
                filepath = os.path.join(save_dir, filename)
                weights_df.to_csv(filepath)
                self.update_log(f"Holdings data saved to: {filepath}")

    def query_holdings(self):
        if not self.all_weights_history:
            self.update_log("\n[Query Failed]: Please run a backtest first to generate holdings data.")
            return
            
        query_date = pd.to_datetime(self.holding_date_input.date().toString("yyyy-MM-dd"))
        self.update_log(f"\n" + "="*20 + f" Holdings on {query_date.strftime('%Y-%m-%d')} " + "="*20)
        
        holdings_data = {}
        for name, weights_df in self.all_weights_history.items():
            if not weights_df.empty and query_date >= weights_df.index.min():
                daily_holding = weights_df.asof(query_date)
                if daily_holding is not None:
                    holdings_data[name] = daily_holding
        
        if not holdings_data:
            self.update_log(f"No holdings data found for any strategy on {query_date.strftime('%Y-%m-%d')}.")
            return
            
        holding_df = pd.DataFrame(holdings_data).applymap(lambda x: f"{x:.2%}" if pd.notna(x) else "0.00%")
        self.update_log(holding_df.to_string())
        self.update_log("="*60)

    def update_log(self, message):
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def on_error(self, message):
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Backtest")
        self.update_log(f"\n!!! AN ERROR OCCURRED !!!\n{message}")