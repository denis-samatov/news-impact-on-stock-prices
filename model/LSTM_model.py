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

# Definition of the DeepStockLSTM model
class DeepStockLSTM(nn.Module):
    """
    An improved LSTM model with multiple LSTM layers, Batch Normalization, and Dropout for stock price forecasting.
    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        output_dim (int): Number of output features.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(DeepStockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Multiple LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        # Fully connected layer to turn the LSTM outputs into predictions
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize the hidden and cell state (h0 and c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Pass the data through the stacked LSTM layers
        out, _ = self.lstm(x, (h0, c0))

        # Apply Batch Normalization to the LSTM's last output
        out = out[:, -1, :]  # Keep only the last time step
        out = self.batch_norm(out)

        # Fully connected layer to get the final prediction
        out = self.fc(out)
        return out

# Class for training the model and forecasting
class StockPredictor:
    """
    Class for training the improved LSTM stock-price forecasting model and evaluating its performance.
    """
    def __init__(self, look_back=30, hidden_dim=64, num_layers=3, learning_rate=0.001, input_dim=2):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.look_back = look_back
        self.input_dim = input_dim  # Account for the number of features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.model = DeepStockLSTM(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_data(self, data):
        # data is a DataFrame of features
        # Normalize the data
        data = self.scaler.fit_transform(data)
        sequences = []
        targets = []
        for i in range(len(data) - self.look_back):
            seq = data[i:i+self.look_back]
            sequences.append(seq[:-1])  # Use the last step as the target
            targets.append(data[i+self.look_back-1, 0])  # Predict close_price (first column)
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
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        self.history = hist

    def predict(self, data):
        # data is a DataFrame of features
        data = self.scaler.transform(data)
        sequences = []
        targets = []
        for i in range(len(data) - self.look_back):
            seq = data[i:i+self.look_back]
            sequences.append(seq[:-1])  # Use the last step as the target
            targets.append(data[i+self.look_back-1, 0])  # The actual close_price value
        sequences = np.array(sequences)
        targets = np.array(targets)
        x_test = torch.from_numpy(sequences).type(torch.Tensor)
        y_test = torch.from_numpy(targets).type(torch.Tensor)
        self.model.eval()
        with torch.no_grad():
            y_test_pred = self.model(x_test)
        # Invert the scaling on the predicted values
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
        print('Test result: %.2f RMSE' % rmse)

    def plot_results(self):
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(self.y_test_original)), self.y_test_original, color='red', label='Actual price')
        plt.plot(range(len(self.y_test_original)), self.predicted_values, color='blue', label='Predicted price')
        plt.title('Stock price forecast')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.grid()
        plt.legend()
        plt.show()
