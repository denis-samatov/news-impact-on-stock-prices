"""Offline synthetic sequence/availability contracts; no collectors or model downloads."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from model.select_parameters import PyTorchRegressor
from model import combine_stocks_news


def test_estimator_fit_predict_retains_samples_and_feature_contract():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2 * x[:, 0]
    estimator = PyTorchRegressor(look_back=3, hidden_dim=4, num_layers=1,
                                 num_epochs=1, batch_size=4, dropout_rate=0)
    fitted = clone(estimator).fit(x[:15], y[:15])
    predicted = fitted.predict(x[15:])
    assert predicted.shape == (5,)
    assert np.isfinite(predicted).all()
    assert np.isfinite(fitted.score(x[15:], y[15:]))
    # Future prices do not refit the training transform.
    assert fitted.scaler.data_max_.tolist() == [14.0]
    assert fitted.predict(x[15:16]).shape == (1,)


def test_after_close_news_only_becomes_available_next_decision():
    stocks = pd.DataFrame({"forecast_at": pd.to_datetime([
        "2024-01-01T15:00Z", "2024-01-02T15:00Z"]), "close_price": [10, 11]})
    news = pd.DataFrame({"available_at": pd.to_datetime([
        "2024-01-01T14:00Z", "2024-01-01T16:00Z"]), "real_score": [.1, .9]})
    joined = combine_stocks_news.join_available_news(stocks, news)
    assert joined.real_score.tolist() == [.1, .9]
    assert (joined.available_at < joined.forecast_at).all()


def test_naive_availability_is_rejected_instead_of_guessing_timezone():
    stocks = pd.DataFrame({"forecast_at": ["2024-01-01 15:00"]})
    news = pd.DataFrame({"available_at": ["2024-01-01T14:00Z"], "real_score": [.1]})
    with pytest.raises(ValueError, match="timezone"):
        combine_stocks_news.join_available_news(stocks, news)


def test_stock_download_has_timeout_and_bounded_read(monkeypatch, capsys):
    from io import BytesIO
    from get_stocks import stocks_parser

    class Response(BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 16 * 1024 * 1024 + 1
            return b"x" * size

    def open_fixture(url, timeout):
        assert timeout == 30
        return Response()

    monkeypatch.setattr(stocks_parser, "urlopen", open_fixture)
    parser = stocks_parser.StockParser(["SBER"], "2024-01-01", "2024-01-02")
    parser.parse_stocks()
    assert "limit" in capsys.readouterr().out
    assert parser.get_stocks_dataframe() is None


def test_news_collector_bounds_empty_pagination_and_closes_browser(tmp_path, monkeypatch):
    from get_news import news_parser

    class Browser:
        page_source = "<html></html>"
        clicks = 0
        closes = 0

        def set_page_load_timeout(self, timeout):
            assert timeout == 30

        def get(self, url):
            assert url == "https://example.invalid/fixture"

        def execute_script(self, script, button):
            self.clicks += 1

        def quit(self):
            self.closes += 1

    class Button:
        def is_displayed(self):
            return True

    class Wait:
        def __init__(self, browser, timeout):
            assert timeout == 10

        def until(self, condition):
            return Button()

    monkeypatch.chdir(tmp_path)
    browser = Browser()
    monkeypatch.setattr(news_parser.webdriver, "Chrome", lambda: browser)
    monkeypatch.setattr(news_parser, "WebDriverWait", Wait)
    parser = news_parser.NewsParser("https://example.invalid/fixture", ["fixture"], "TEST")
    parser.parse_news(len_dataset=5, max_pages=2)
    parser.close_driver()
    assert browser.clicks == 2
    assert browser.closes == 1


def test_asof_preserves_caller_fields_order_and_strict_temporal_cutoff():
    stocks = pd.DataFrame({"forecast_at": ["2024-01-01T16:00Z", "2024-01-01T14:00Z"],
                           "_input_order": [10, 11]})
    news = pd.DataFrame({"available_at": ["2024-01-01T13:00Z", "2024-01-01T13:00Z",
                                         "2024-01-01T15:00Z", "2024-01-01T16:00Z"],
                         "real_score": [.1, .3, .9, 1.]})
    original_stocks, original_news = stocks.copy(deep=True), news.copy(deep=True)
    joined = combine_stocks_news.join_available_news(stocks, news)
    assert joined["_input_order"].tolist() == [10, 11]
    assert joined.real_score.tolist() == pytest.approx([.9, .2])
    pd.testing.assert_frame_equal(stocks, original_stocks)
    pd.testing.assert_frame_equal(news, original_news)
