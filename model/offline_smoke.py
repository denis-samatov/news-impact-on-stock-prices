"""Synthetic rolling-origin/multi-seed acceptance, never a financial experiment."""
import json

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from model.select_parameters import PyTorchRegressor


def run():
    time = np.arange(48, dtype=float)
    prices = 10 + .1 * time + np.sin(time / 4)
    features = prices.reshape(-1, 1)
    rows = []
    for fold, (train, test) in enumerate(TimeSeriesSplit(n_splits=3).split(features)):
        naive = np.concatenate([[prices[train[-1]]], prices[test[:-1]]])
        for seed in (0, 1):
            estimator = PyTorchRegressor(look_back=3, hidden_dim=4, num_layers=1,
                                         num_epochs=1, batch_size=4, dropout_rate=0, seed=seed)
            predicted = estimator.fit(features[train], prices[train]).predict(features[test])
            rows.append({"fold": fold, "seed": seed, "n_test": len(test),
                         "rmse": float(np.sqrt(mean_squared_error(prices[test], predicted))),
                         "persistence_rmse": float(np.sqrt(mean_squared_error(prices[test], naive))),
                         "predictions": [{"index": int(i), "gold": float(g), "prediction": float(p),
                                          "persistence": float(b)}
                                         for i, g, p, b in zip(test, prices[test], predicted, naive)]})
    return {"synthetic": True, "scope": "software acceptance; no market/news efficacy claim",
            "protocol": "3 rolling-origin folds x 2 seeds; one-step observed history", "runs": rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
