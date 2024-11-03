import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class StockLSTM(nn.Module):
    """
    Рекуррентная нейронная сеть с LSTM слоями для прогнозирования цен на акции.
    Аргументы:
        input_dim (int): Количество входных признаков.
        hidden_dim (int): Количество признаков в скрытом состоянии.
        num_layers (int): Количество рекуррентных слоев.
        output_dim (int): Количество выходных признаков.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Определяем LSTM слой
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        
        # Полносвязный слой для преобразования выходов LSTM в прогнозы
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Инициализация начальных состояний скрытого и ячейки (h0 и c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # Проходим данные через LSTM слои
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Полносвязный слой для получения окончательного прогноза
        out = self.fc(out[:, -1, :]) 
        return out


class StockPredictor:
    """
    Класс для обучения LSTM модели прогнозирования цен на акции и оценки её производительности.
    """
    def __init__(self):
        # Инициализация параметров
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.look_back = 30
        self.input_dim = 1
        self.hidden_dim = 50  # Увеличение скрытого размера для повышения производительности
        self.num_layers = 3   # Увеличение количества LSTM слоев
        self.output_dim = 1

    def load_data(self, train_data):
        """
        Масштабирует и форматирует данные для обучения.
        Аргументы:
            train_data (np.array): Массив, содержащий данные цен на акции.
        """
        train_data = self.scaler.fit_transform(train_data.reshape(-1, 1))
        
        # Подготовка последовательностей на основе окна look_back
        data = []
        for index in range(len(train_data) - self.look_back):
            data.append(train_data[index: index + self.look_back])
        data = np.array(data)
        
        # Разделение последовательностей на признаки и целевую переменную
        x_train = data[:, :-1]
        y_train = data[:, -1]
        
        # Преобразование numpy массивов в тензоры
        self.x_train = torch.from_numpy(x_train).type(torch.Tensor)
        self.y_train = torch.from_numpy(y_train).type(torch.Tensor)

    def train_model(self, num_epochs=300):
        """
        Обучает модель LSTM.
        Аргументы:
            num_epochs (int): Количество эпох обучения.
        """
        self.model = StockLSTM(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)  # Пониженная скорость обучения для стабильности
        hist = np.zeros(num_epochs)
        
        for t in range(num_epochs):
            y_train_pred = self.model(self.x_train)
            loss = loss_fn(y_train_pred, self.y_train)
            hist[t] = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if t % 10 == 0:
                print("Эпоха ", t, "MSE: ", loss.item())
        
        self.trained_model = self.model
        self.history = hist

    def predict(self, test_data):
        """
        Прогнозирует цены на акции с использованием обученной модели на тестовых данных.
        Аргументы:
            test_data (np.array): Массив с тестовыми ценами на акции.
        """
        test_data = self.scaler.transform(test_data.reshape(-1, 1))
        
        # Подготовка последовательностей для тестирования на основе окна look_back
        data = []
        for index in range(len(test_data) - self.look_back):
            data.append(test_data[index: index + self.look_back])
        data = np.array(data)
        
        # Разделение последовательностей на признаки и целевую переменную
        x_test = data[:, :-1]
        y_test = data[:, -1]
        
        # Преобразование numpy массивов в тензоры
        self.x_test = torch.from_numpy(x_test).type(torch.Tensor)
        self.y_test = torch.from_numpy(y_test).type(torch.Tensor)
        
        # Прогнозирование и обратное масштабирование прогнозов на исходный масштаб
        y_test_pred = self.trained_model(self.x_test)
        self.predicted_values = self.scaler.inverse_transform(y_test_pred.detach().numpy())
        self.y_test_original = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

    def calculate_rmse(self):
        """
        Вычисляет RMSE (корень из среднеквадратичной ошибки) прогноза.
        """
        test_score = math.sqrt(mean_squared_error(self.y_test_original, self.predicted_values))
        print('Тестовый результат: %.2f RMSE' % test_score)

    def plot_results(self):
        """
        Строит график реальных и прогнозируемых цен на акции для визуализации.
        """
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(self.y_test_original)), self.y_test_original, color='red', label='Реальная цена акции')
        plt.plot(range(len(self.y_test_original)), self.predicted_values, color='blue', label='Прогнозируемая цена акции')
        plt.title('Прогноз цен на акции')
        plt.xlabel('День')
        plt.ylabel('Цена акции')
        plt.legend()
        plt.show()

    def get_predictions_and_originals(self):
        """
        Возвращает массивы прогнозируемых и реальных цен на акции для дальнейшего анализа.
        """
        return self.predicted_values, self.y_test_original
