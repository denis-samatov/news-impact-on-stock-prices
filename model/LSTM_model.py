import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error




def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Определение модели DeepStockLSTM
class DeepStockLSTM(nn.Module):
    """
    Улучшенная модель LSTM с несколькими слоями LSTM, Batch Normalization и Dropout для прогнозирования цен на акции.
    Аргументы:
        input_dim (int): Количество входных признаков.
        hidden_dim (int): Количество признаков в скрытом состоянии.
        num_layers (int): Количество рекуррентных слоев.
        output_dim (int): Количество выходных признаков.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(DeepStockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Несколько LSTM слоев
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Полносвязный слой для преобразования выходов LSTM в прогнозы
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Инициализация начальных состояний скрытого слоя и ячейки (h0 и c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Пропуск данных через несколько LSTM слоев
        out, _ = self.lstm(x, (h0, c0))
        
        # Применение Batch Normalization к последнему выходу LSTM
        out = out[:, -1, :]  # Берем только последний временной шаг
        out = self.batch_norm(out)
        
        # Полносвязный слой для получения окончательного прогноза
        out = self.fc(out)
        return out

# Класс для обучения модели и прогнозирования
class StockPredictor:
    """
    Класс для обучения улучшенной LSTM модели прогнозирования цен на акции и оценки её производительности.
    """
    def __init__(self, look_back=30, hidden_dim=64, num_layers=3, learning_rate=0.001, input_dim=2):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.look_back = look_back
        self.input_dim = input_dim  # Учитываем количество признаков
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.model = DeepStockLSTM(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_data(self, data):
        # data - это DataFrame с признаками
        # Нормализация данных
        data = self.scaler.fit_transform(data)
        sequences = []
        targets = []
        for i in range(len(data) - self.look_back):
            seq = data[i:i+self.look_back]
            sequences.append(seq[:-1])  # Последний шаг используем как цель
            targets.append(data[i+self.look_back-1, 0])  # Предсказываем close_price (первый столбец)
        sequences = np.array(sequences)
        targets = np.array(targets)
        self.x_train = torch.from_numpy(sequences).type(torch.Tensor)
        self.y_train = torch.from_numpy(targets).type(torch.Tensor)

    def train_model(self, num_epochs=50, batch_size=32):
        self.model.train()
        hist = []
        for epoch in range(num_epochs):
            permutation = torch.randperm(self.x_train.size(0))
            epoch_loss = 0.0
            for i in range(0, self.x_train.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = self.x_train[indices], self.y_train[indices]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.view(-1), batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            hist.append(epoch_loss)
            if epoch % 10 == 0:
                print(f"Эпоха {epoch}, Потери: {epoch_loss:.4f}")
        
        self.history = hist

    def predict(self, data):
        # data - это DataFrame с признаками
        data = self.scaler.transform(data)
        sequences = []
        targets = []
        for i in range(len(data) - self.look_back):
            seq = data[i:i+self.look_back]
            sequences.append(seq[:-1])  # Последний шаг используем как цель
            targets.append(data[i+self.look_back-1, 0])  # Реальное значение close_price
        sequences = np.array(sequences)
        targets = np.array(targets)
        x_test = torch.from_numpy(sequences).type(torch.Tensor)
        y_test = torch.from_numpy(targets).type(torch.Tensor)
        self.model.eval()
        with torch.no_grad():
            y_test_pred = self.model(x_test)
        # Обратное преобразование предсказанных значений
        y_test_pred_inv = []
        y_test_inv = []
        for i in range(len(y_test_pred)):
            pred = np.zeros(self.input_dim)
            actual = np.zeros(self.input_dim)
            pred[0] = y_test_pred[i].item()
            actual[0] = y_test[i].item()
            pred_inv = self.scaler.inverse_transform([pred])
            actual_inv = self.scaler.inverse_transform([actual])
            y_test_pred_inv.append(pred_inv[0][0])
            y_test_inv.append(actual_inv[0][0])
        self.predicted_values = np.array(y_test_pred_inv)
        self.y_test_original = np.array(y_test_inv)

    def calculate_rmse(self):
        rmse = math.sqrt(mean_squared_error(self.y_test_original, self.predicted_values))
        print('Тестовый результат: %.2f RMSE' % rmse)

    def plot_results(self):
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(self.y_test_original)), self.y_test_original, color='red', label='Реальная цена')
        plt.plot(range(len(self.y_test_original)), self.predicted_values, color='blue', label='Прогнозируемая цена')
        plt.title('Прогноз цен на акции')
        plt.xlabel('День')
        plt.ylabel('Цена')
        plt.grid()
        plt.legend()
        plt.show()
