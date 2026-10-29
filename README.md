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

**Historical data-fidelity issue:** the earlier notebook used hardcoded example RMSE lists for a Mann–Whitney test. That cell is now explanatory Markdown and makes no executable significance claim. Historical single-run outputs (including 3.93 vs 8.04 RMSE in one comparison) remain saved for provenance and were not regenerated after the software corrections. They do not establish a real repeated-run distribution or a validated sentiment forecasting advantage.

A rigorous real-data experiment still requires audited availability timestamps, an untouched chronological test, multiple seeds/windows, simple baselines and paired temporal uncertainty. The new offline smoke is explicitly synthetic software acceptance.

## Features
- **Data Collection**: Scrapes news articles and stock prices from bcs-express.ru and Finam.
- **Sentiment Analysis**: Analyzes sentiment within news articles (via translation + FinBERT, or RuBERT directly) to evaluate market impact.
- **Predictive Modeling**: Trains LSTM models with and without a sentiment feature and compares their RMSE -- see the Methodology section above for what is and isn't validated here.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/denis-samatov/news-impact-on-stock-prices.git
    ```
2. Install the required packages:
    ```bash
    pip install --require-hashes -r requirements-execution.txt
    ```

## Usage
Run the scripts in `get_news/` and `get_stocks/` to collect data, then the notebooks in `correlation_analysis/` and `model/` for analysis and model training, in that order. There's no single entry point or CLI; each script/notebook was run manually as the project progressed.

## Full report
The original term-paper writeup, in Russian, is in [`report/`](report/).

## Contributing
Contributions are welcome! Please open an issue to discuss improvements, features, or bugs.


## Corrected offline acceptance path

```bash
python -m pytest tests -q
OMP_NUM_THREADS=1 python -m model.offline_smoke > /tmp/news-synthetic-smoke.json
```

The second command uses a generated numeric series, three rolling-origin folds and two seeds,
with a persistence baseline and saved predictions. It is software acceptance, not an experiment
on real quotes/news. No collection, translation, model download, API call or real performance
result is implied. `scikit-learn` is required; install NLTK resources explicitly only when preparing
the collector (`python -m nltk.downloader punkt punkt_tab`). Imports no longer download NLTK data.

`PyTorchRegressor` now predicts one target per supplied row from the preceding `look_back`
observed feature rows. Its target scaler is separate from its feature scaler; both fit only
training data. Validation rows extend the observed history, so this is rolling one-step,
not open-loop forecasting. Use chronological `TimeSeriesSplit`, an untouched outer test and
multiple fixed seeds. `StockPredictor.calculate_rmse()` also returns the numeric RMSE.
The old synthetic significance cell is now non-executable explanatory Markdown. Saved earlier
notebook outputs are historical and were not regenerated after this repair.

`model.combine_stocks_news.join_available_news` joins one ticker at a time. Supply timezone-aware
`forecast_at` decisions from the actual trading calendar, and `available_at` timestamps stating
when article information became usable. Date-only/publication timestamps are not silently promoted
to that stronger claim. Only strictly earlier information enters a decision; equal-time articles
are mean-aggregated, latest available score is selected, and absence stays NaN. Audit original
source revisions, collection time, duplicates, after-close availability and training-only imputation
before real research. A genuine market experiment and its data rights remain prerequisites.

## Separate execution profile (2026-10-29)

`requirements-execution.in` is the maintained input; `requirements-execution.txt` is generated
with `uv pip compile requirements-execution.in --python-version 3.13 --generate-hashes --output-file requirements-execution.txt`.
It is a new Python 3.13 software-validation environment, not the historical scientific environment.
Create a local venv, then install with `python -m pip install --require-hashes -r requirements-execution.txt`.
Do not edit the generated lock by hand. OSV exact-version scanning covers the resolved public
packages; it does not establish exploit reachability or guarantee that undisclosed issues are absent.
Historical locks, predictions and numerical results remain unchanged. A new experimental result
requires its own inputs, source revision, environment and numerical qualification.

Collector imports perform no model downloads or browser creation. Browser page loads use a
30-second timeout, show-more attempts default to 100, and stock responses are capped at 16 MiB.
The collector test uses a fake browser and in-memory response; the real websites, browser driver,
translation checkpoints and market timestamps were not contacted or requalified.

The original `requirements.txt` is retained byte-for-byte as a historical dependency snapshot;
it is not the maintained installation command. Use the separate execution profile above.

This profile was resolved on macOS arm64 for Python 3.13. It is not a universal lock for
all operating systems, Python versions or accelerators; regenerate and qualify another target.
