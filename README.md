# AI-Enhanced All-Weather Backtesting Platform  
This is a sophisticated backtesting platform for designing, testing, and analyzing AI-enhanced "All-Weather" investment strategies. The application combines traditional quantitative finance techniques like Risk Parity with modern machine learning models (including Stacking ensembles and GRU networks) and a powerful AI Agent for dynamic, macro-aware asset allocation.

这是一个复杂精密的回测平台，用于设计、测试和分析由AI增强的“全天候”投资策略。该应用将风险平价（Risk Parity）等传统量化金融技术与现代机器学习模型（包括Stacking集成和GRU神经网络）以及一个强大的、用于动态宏观资产配置的AI智能体相结合。  


## ✨ Features | 功能特性  

### English  
- **Dynamic Asset Pool**: Configure any combination of stock, bond, and commodity ETFs available through akshare.  
- **Flexible Machine Learning**:  
  - Choose from a rich library of technical factors (Momentum, Volatility, Oscillators, etc.).  
  - Build powerful Stacking Ensembles by selecting multiple base models (e.g., Logistic Regression, Random Forest, LightGBM) and a meta-model.  
  - Optionally, use a standalone GRU (Gated Recurrent Unit) deep learning model for time-series forecasting.  
  - Customize key hyperparameters for all models directly from the UI.  
- **AI Agent Integration**:  
  - Provide high-level, human-language macroeconomic descriptions for specific months (e.g., "US-China trade friction intensifies").  
  - The AI Agent (powered by OpenAI or DeepSeek) interprets these prompts to generate a top-down risk budget.  
  - This AI-driven risk budget is then combined with quantitative risk parity optimization and ML-based timing signals.  
- **Comprehensive Strategy Suite**:  
  - **AI-Driven Strategy**: The flagship strategy combining AI macro views, risk parity, and ML timing.  
  - **ML-Only Timing**: A pure tactical allocation based on the output of the selected ML model(s).  
  - **Budgeted & Naive Risk Parity**: Classic and simple risk-based allocation.  
  - **Equal Weight**: A standard benchmark.  
- **In-depth Analysis**:  
  - Plots a comparative equity curve for all tested strategies.  
  - Calculates and displays key performance metrics (Annual Return, Volatility, Sharpe Ratio, Max Drawdown, etc.).  
  - Features a "Holdings Query" tool to inspect the exact asset allocation of any strategy on any given day.  
</details>  


### 中文  
- **动态资产池**：可配置akshare支持的任何股票、债券和商品ETF组合，使用爬虫稳定接口。  
- **灵活的机器学习**：  
  - 从丰富的技术因子库中选择（动量、波动率、震荡指标等）。  
  - 通过选择多个基模型（如逻辑回归、随机森林、LightGBM）和一个元模型，构建强大的Stacking集成模型。  
  - 可选择使用独立的**GRU（门控循环单元）**深度学习模型进行时间序列预测。  
  - 直接在UI中自定义所有模型的关键超参数。  
- **AI智能体集成**：  
  - 为特定月份提供高层次、自然语言的宏观经济描述（例如，“中美贸易摩擦加剧”）。  
  - AI智能体（由OpenAI或DeepSeek驱动）会解读这些提示，生成顶层的风险预算。  
  - 这个由AI驱动的风险预算，将与量化风险平价优化和基于机器学习的择时信号相结合。  
- **全面的策略组合**：  
  - **AI驱动策略**：结合了AI宏观观点、风险平价和机器学习择时的旗舰策略。  
  - **纯机器学习择时**：完全基于所选机器学习模型输出的战术配置。  
  - **预算风险平价 & 朴素风险平价**：经典和简化的基于风险的配置。  
  - **等权重策略**：标准的基准策略。  
- **深度分析**：  
  - 绘制所有测试策略的净值曲线对比图。  
  - 计算并显示关键性能指标（年化收益、波动率、夏普比率、最大回撤等）。  
  - 提供“持仓查询”工具，以查看任何策略在任何一天的确切资产配置。  
</details>  





## 🛠️ Installation & Setup | 安装与设置  
1. **Clone the repository**：  
   ```bash  
   git clone https://github.com/your-username/all_weather_project.git  
   cd all_weather_project

2. **Clone the repository**：
   ```bash
   # Create virtual environment  
   python -m venv venv  

   # Activate (Windows)  
   .\venv\Scripts\activate  

   # Activate (macOS/Linux)  
   source venv/bin/activate

3. **Install dependencies**：
   ```bash
   pip install -r requirements.txt

4. **Set up API Keys**：
   ```bash
   从 OpenAI 或 DeepSeek 获取 API 密钥。
   在应用 UI 的 “API Key” 字段中输入密钥（请勿硬编码到源代码中）。


## 🛠️ todo：  
支持模型滚动训练，减少过拟合

