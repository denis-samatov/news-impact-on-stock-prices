import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error





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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, dropout=0.3)
        
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


class StockPredictor:
    """
    Класс для обучения улучшенной LSTM модели прогнозирования цен на акции и оценки её производительности.
    """
    def __init__(self, look_back=30, hidden_dim=64, num_layers=3, learning_rate=0.001):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.look_back = look_back
        self.input_dim = 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.model = DeepStockLSTM(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_data(self, train_data):
        train_data = self.scaler.fit_transform(train_data.reshape(-1, 1))
        data = []
        for index in range(len(train_data) - self.look_back):
            data.append(train_data[index: index + self.look_back])
        data = np.array(data)
        x_train = data[:, :-1]
        y_train = data[:, -1]
        self.x_train = torch.from_numpy(x_train).type(torch.Tensor)
        self.y_train = torch.from_numpy(y_train).type(torch.Tensor)

    def train_model(self, num_epochs=300, batch_size=16):
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
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            hist.append(epoch_loss)
            if epoch % 10 == 0:
                print(f"Эпоха {epoch}, Потери: {epoch_loss:.4f}")
        
        self.history = hist

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data.reshape(-1, 1))
        data = []
        for index in range(len(test_data) - self.look_back):
            data.append(test_data[index: index + self.look_back])
        data = np.array(data)
        x_test = torch.from_numpy(data[:, :-1]).type(torch.Tensor)
        y_test = torch.from_numpy(data[:, -1]).type(torch.Tensor)
        self.model.eval()
        with torch.no_grad():
            y_test_pred = self.model(x_test)
        self.predicted_values = self.scaler.inverse_transform(y_test_pred.numpy())
        self.y_test_original = self.scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

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
        plt.legend()
        plt.show()
