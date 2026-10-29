"""Deterministic latest-available sentiment features at explicit decision timestamps."""
import pandas as pd


def _aware_utc(values):
    if any(pd.Timestamp(value).tzinfo is None for value in values):
        raise ValueError("timestamps must include an explicit timezone")
    result = pd.to_datetime(values, utc=True)
    if result.isna().any():
        raise ValueError("timestamps must not be missing")
    return result


def join_available_news(stocks, news):
    """As-of join for one ticker, using only news available strictly before decision.

    Required columns: stocks.forecast_at; news.available_at and news.real_score.
    available_at is the time the information became usable, not an inferred date.
    Equal-time news are aggregated by mean; a decision with no prior news stays NaN.
    The caller supplies the trading calendar/decision times and runs tickers separately.
    """
    left, right = stocks.copy(), news[["available_at", "real_score"]].copy()
    if "real_score" in left or "available_at" in left:
        raise ValueError("stocks must not already contain joined news columns")
    left["forecast_at"] = _aware_utc(left["forecast_at"])
    right["available_at"] = _aware_utc(right["available_at"])
    right["real_score"] = pd.to_numeric(right["real_score"], errors="raise")
    right = right.groupby("available_at", as_index=False)["real_score"].mean()
    left["_input_order"] = range(len(left))
    result = pd.merge_asof(left.sort_values("forecast_at", kind="stable"),
                           right.sort_values("available_at", kind="stable"),
                           left_on="forecast_at", right_on="available_at",
                           direction="backward", allow_exact_matches=False)
    return result.sort_values("_input_order").drop(columns="_input_order").reset_index(drop=True)
