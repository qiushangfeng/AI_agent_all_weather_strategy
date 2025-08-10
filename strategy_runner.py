# strategy_runner.py
import traceback
from PyQt6.QtCore import QThread, pyqtSignal

# Import our new modules
import data_handler
import models
import portfolio
import ai_agent

class StrategyRunner(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def log(self, message):
        self.log_signal.emit(message)

    def progress(self, percent, message):
        self.progress_signal.emit(percent, message)

    def run(self):
        """The main execution function for the strategy thread."""
        try:
            # Step 1: Data Fetching and Feature Engineering
            price_data = data_handler.get_etf_data(self.config, self.log)
            if price_data.empty: raise ValueError("Failed to get any valid ETF data.")
            
            features_df = data_handler.build_features(price_data, self.config, self.progress, self.log)
            labels_df = data_handler.build_labels(price_data, self.config, self.log)

            # Step 2: ML Model Training to get signals
            ml_signals_df = models.get_ml_signals(features_df, labels_df, self.config, self.log)

            # Step 3: Portfolio Construction (Weight Calculation)
            # We create a simple wrapper for the AI function to pass to the portfolio module.
            def ai_function_wrapper(macro_desc, asset_classes):
                return ai_agent.get_ai_macro_view(self.config, macro_desc, asset_classes, self.log)

            all_weights = portfolio.calculate_all_weights(
                price_data, 
                ml_signals_df, 
                self.config, 
                self.log, 
                self.progress, 
                ai_function_wrapper
            )

            # Step 4: Backtesting
            all_portfolios = portfolio.run_all_backtests(price_data, all_weights, self.log, self.progress)
            
            # Step 5: Consolidate and emit results
            final_results = {
                'portfolios': all_portfolios,
                'weights': all_weights
            }
            self.finished_signal.emit(final_results)

        except Exception as e:
            error_msg = f"Strategy execution failed: {e}\n{traceback.format_exc()}"
            self.log(error_msg)
            self.error_signal.emit(error_msg)