"""Chronological one-step LSTM estimator compatible with sklearn model selection.

Each y[t] is predicted from X[t-look_back:t]. During predict, observed preceding
validation rows extend the training context. This is rolling one-step evaluation,
not an open-loop multi-step forecast. Split/tune only with chronological folds.
"""
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from model.LSTM_model import DeepStockLSTM, set_seed


class PyTorchRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, look_back=30, hidden_dim=64, num_layers=3, learning_rate=0.001,
                 dropout_rate=0.2, num_epochs=50, batch_size=32, seed=42):
        self.look_back = look_back
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1, 1)
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X/y must be finite aligned feature rows and targets")
        if self.look_back < 1 or len(X) < self.look_back + 2:
            raise ValueError("Need at least look_back + 2 training rows")
        if self.batch_size < 1 or self.num_epochs < 1:
            raise ValueError("batch_size and num_epochs must be positive")
        set_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features_in_ = X.shape[1]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler_ = MinMaxScaler(feature_range=(-1, 1))
        features = self.scaler.fit_transform(X)
        targets = self.target_scaler_.fit_transform(y).ravel()
        windows = np.stack([features[i-self.look_back:i] for i in range(self.look_back, len(X))])
        x_train = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(targets[self.look_back:], dtype=torch.float32, device=self.device)
        self.train_tail_ = features[-self.look_back:].copy()
        self.model = DeepStockLSTM(self.n_features_in_, self.hidden_dim, self.num_layers, 1,
                                   dropout_rate=self.dropout_rate).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        self.history = []
        self.model.train()
        for _ in range(self.num_epochs):
            order = np.random.permutation(len(x_train))
            # At least two samples per batch: BatchNorm cannot train on a singleton.
            batches = np.array_split(order, max(1, len(order) // max(2, self.batch_size)))
            epoch_loss = 0.0
            for indices in batches:
                optimizer.zero_grad()
                outputs = self.model(x_train[indices]).reshape(-1)
                loss = criterion(outputs, y_train[indices])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.history.append(epoch_loss)
        return self

    def predict(self, X):
        if not hasattr(self, "train_tail_"):
            raise ValueError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("predict requires finite rows with the fitted feature count")
        if not len(X):
            return np.empty(0)
        context = np.concatenate([self.train_tail_, self.scaler.transform(X)], axis=0)
        windows = np.stack([context[i:i+self.look_back] for i in range(len(X))])
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.as_tensor(windows, dtype=torch.float32, device=self.device))
        return self.target_scaler_.inverse_transform(output.cpu().numpy()).ravel()

    def score(self, X, y):
        return -float(np.sqrt(mean_squared_error(y, self.predict(X))))


param_grid = {"look_back": [15, 30, 60], "learning_rate": [0.001, 0.003, 0.01],
              "dropout_rate": [0.1, 0.2, 0.3], "num_epochs": [50, 100, 150],
              "batch_size": [8, 16, 32, 64]}
regressor = PyTorchRegressor()
grid_search = GridSearchCV(regressor, param_grid=param_grid,
                           scoring="neg_root_mean_squared_error",
                           cv=TimeSeriesSplit(n_splits=3), verbose=2)
# Explicitly call grid_search.fit(X, y) only after fixing dates, outer test and run budget.
