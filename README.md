# The Impact of News on Stock Prices

**This is a university term-paper project** (a "курсовая работа" for a Financial Mathematics course), not a production system or a validated research result. It investigates whether news sentiment correlates with, and can help forecast, movements in Russian stock prices, using SBER, GAZP, LKOH, and MTSS as the tickers studied.

## Project Goals
- Analyze how specific types of news events correlate with changes in stock prices.
- Apply machine learning models to understand the potential for forecasting based on news data.
- Implement natural language processing techniques to classify and interpret news sentiment.

## Data sources and scope

- **News**: scraped from [bcs-express.ru](https://bcs-express.ru) (`get_news/news_parser.py`), filtered to articles mentioning ticker-specific keyword lists (real Russian company names/aliases, e.g. "Сбербанк", "Газпром" -- see `get_news/collect_news_data.py`). News sentiment is scored by translating each filtered sentence to English (`Helsinki-NLP/opus-mt-ru-en`) and classifying it with FinBERT (`get_news/sentiment_analyzer.py`); `model/sentiment_analysis_models.ipynb` also compares that approach against classifying the original Russian text directly with RuBERT-based models.
- **Stock prices**: pulled from Finam's CSV export endpoint (`get_stocks/stocks_parser.py`).
- **Date range**: varies by notebook and was not held constant across the project. `correlation_analysis/corr_analysis.ipynb` documents using six months of SBER data (2023-06-01 to 2023-12-31); `get_news/category_analysis.ipynb`'s captured output shows collected news spanning 2015-2024. The exact date range and row count behind `model/main.ipynb`'s trained forecasting model can't be reconstructed from this repository -- `data/data.csv` is gitignored and not committed.

## Methodology and temporal split

`model/main.ipynb` trains two LSTM models (with and without a news-sentiment feature) on the same chronologically sorted price series, using a **90/10 chronological split** -- `train_data = data.iloc[:train_size]`, `test_data = data.iloc[train_size:]` -- not a random shuffle-split, which avoids training on future data relative to the test period. Both models use `look_back=30` (a 30-step input window) and are trained for a single run each (no cross-validation, no repeated runs with different seeds).

**A known data-fidelity issue in this repo:** the notebook's headline comparison (`model/main.ipynb`, the Mann-Whitney U test cell) runs on a **hardcoded example RMSE list**, not on the actual single-run results computed earlier in the same notebook. The real, single-run RMSE values are **3.93 (with sentiment) vs. 8.04 (without sentiment)** -- a real difference in the expected direction, but from one run each, not a validated, statistically tested result. The significance test's "p < 0.05, sentiment improves forecast quality" conclusion is computed against synthetic placeholder numbers and should not be read as a validated finding. This is flagged directly in the notebook's own code as well.

**What this means:** treat the single-run RMSE comparison (3.93 vs. 8.04) as suggestive, not proven. No systematic leakage audit, cross-validation, or repeated-run significance test was performed; a rigorous version of this project would repeat training across multiple seeds/windows and run the significance test on those real repeated results.

## Features
- **Data Collection**: Scrapes news articles and stock prices from bcs-express.ru and Finam.
- **Sentiment Analysis**: Analyzes sentiment within news articles (via translation + FinBERT, or RuBERT directly) to evaluate market impact.
- **Predictive Modeling**: Trains LSTM models with and without a sentiment feature and compares their RMSE -- see the Methodology section above for what is and isn't validated here.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/denis-samatov/news_impact_on_stock_prices.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the scripts in `get_news/` and `get_stocks/` to collect data, then the notebooks in `correlation_analysis/` and `model/` for analysis and model training, in that order. There's no single entry point or CLI; each script/notebook was run manually as the project progressed.

## Full report
The original term-paper writeup, in Russian, is in [`report/`](report/).

## Contributing
Contributions are welcome! Please open an issue to discuss improvements, features, or bugs.
