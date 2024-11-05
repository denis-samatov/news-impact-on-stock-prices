import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from LSTM_model import DeepStockLSTM




# Обертка для использования с GridSearchCV
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, look_back=30, hidden_dim=64, num_layers=3, learning_rate=0.001, dropout_rate=0.2, num_epochs=50, batch_size=32):
        self.look_back = look_back
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None
        self.history = []

    def fit(self, X, y):
        # Подготовка данных
        data = np.concatenate((X, y.reshape(-1,1)), axis=1)
        data = self.scaler.fit_transform(data)
        
        sequences = []
        targets = []
        for i in range(len(data) - self.look_back):
            seq = data[i:i+self.look_back]
            sequences.append(seq[:-1])
            targets.append(seq[-1, -1])  # Последнее значение - цель
        
        X_train = np.array(sequences)
        y_train = np.array(targets)
        
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)
        
        # Инициализация модели
        input_dim = X_train.shape[2]
        output_dim = 1
        self.model = DeepStockLSTM(input_dim, self.hidden_dim, self.num_layers, output_dim, self.dropout_rate)
        self.model.to(self.device)
        
        # Определение функции потерь и оптимизатора
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Обучение модели
        self.model.train()
        for epoch in range(self.num_epochs):
            permutation = torch.randperm(X_train.size(0))
            epoch_loss = 0.0
            for i in range(0, X_train.size(0), self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs.view(-1), batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            self.history.append(epoch_loss)
        return self

    def predict(self, X):
        # Подготовка данных
        data = self.scaler.transform(X)
        sequences = []
        for i in range(len(data) - self.look_back):
            seq = data[i:i+self.look_back]
            sequences.append(seq[:-1])
        X_test = np.array(sequences)
        X_test = torch.from_numpy(X_test).float().to(self.device)
        
        # Прогнозирование
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
        predictions = outputs.cpu().numpy()
        # Обратное преобразование масштаба
        predictions = self.scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], X.shape[1]-1)), predictions)))
        return predictions[:, -1]

    def score(self, X, y):
        y_pred = self.predict(X)
        rmse = math.sqrt(mean_squared_error(y[self.look_back:], y_pred))
        # Возвращаем отрицательный RMSE для совместимости с GridSearchCV
        return -rmse

    def get_params(self, deep=True):
        return {'look_back': self.look_back,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'learning_rate': self.learning_rate,
                'dropout_rate': self.dropout_rate,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Пример использования:

# Загрузка данных (замените на свои данные)
# Предположим, что prices - это одномерный массив с ценами акций
# prices = np.array([...])

# Разделение данных на признаки и целевую переменную
# В данном случае мы используем цены как X и y
# X = prices.reshape(-1, 1)
# y = prices

# Определение сетки гиперпараметров
param_grid = {
    'look_back': [15, 30, 60],
    'learning_rate': [0.001, 0.003, 0.01],
    'dropout_rate': [0.1, 0.2, 0.3],
    'num_epochs': [50, 100, 150],
    'batch_size': [8, 16, 32, 64]
}

# Инициализация регрессора
regressor = PyTorchRegressor()

# Определение метрики RMSE для GridSearchCV
def rmse_scorer(y_true, y_pred):
    return -math.sqrt(mean_squared_error(y_true[regressor.look_back:], y_pred))

scorer = make_scorer(rmse_scorer)

# Инициализация GridSearchCV
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring=scorer, cv=3, verbose=2)

# Запуск поиска по сетке гиперпараметров
# grid_search.fit(X, y)

# Получение наилучших параметров
# print("Наилучшие параметры: ", grid_search.best_params_)
# print("Лучшее значение RMSE: ", -grid_search.best_score_)

# Примечание:
# Обучение моделей глубокого обучения в цикле GridSearchCV может занять значительное время.
# Рекомендуется использовать меньший набор параметров или RandomizedSearchCV для ускорения процесса.
# Убедитесь, что данные подготовлены правильно и соответствуют требованиям модели.
