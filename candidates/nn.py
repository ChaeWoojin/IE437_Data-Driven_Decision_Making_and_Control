import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define the Neural Network with Dropout and L2 Regularization
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return self.sigmoid(x)

class NNOnlineLearning:
    def __init__(self, input_dim, lr=0.001, weight_decay=1e-5, batch_size=32):
        self.model = SimpleNN(input_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        feature_columns = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 
                           'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
                           'Valence', 'Tempo', 'Duration_ms', 'Year', 'Popularity', 'Explicit']
        data = data.dropna(subset=['Like'])
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        features = data[feature_columns]
        labels = data['Like'].values

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return features_tensor, labels_tensor

    def initial_learning_phase(self, features_tensor, labels_tensor, initial_phase_size):
        self.model.train()
        for epoch in range(10):  # Training for a few epochs for initial phase
            for i in range(0, initial_phase_size, self.batch_size):
                batch_features = features_tensor[i:i + self.batch_size]
                batch_labels = labels_tensor[i:i + self.batch_size]
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, features_tensor, labels_tensor):
        cumulative_regret = []
        total_regret = 0
        correct_predictions = 0
        total_predictions = 0
        winning_rate = []

        for i in range(0, len(features_tensor), self.batch_size):
            batch_features = features_tensor[i:i + self.batch_size]
            batch_labels = labels_tensor[i:i + self.batch_size]
            
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            predictions = (outputs.detach().numpy() >= 0.5).astype(int)
            batch_regret = sum(1 - (predictions.flatten() == batch_labels.numpy().flatten()))
            total_regret += batch_regret
            cumulative_regret.append(total_regret)
            
            correct_predictions += sum(predictions.flatten() == batch_labels.numpy().flatten())
            total_predictions += len(batch_labels)
            current_winning_rate = correct_predictions / total_predictions
            winning_rate.append(current_winning_rate)
        
        return cumulative_regret, winning_rate

def plot_results(cumulative_regret, winning_rate):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(cumulative_regret) + 1), cumulative_regret)
    plt.xlabel('Batch Number')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret Over Time with Online Learning (NN)')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(winning_rate) + 1), winning_rate)
    plt.xlabel('Batch Number')
    plt.ylabel('Winning Rate')
    plt.title('Winning Rate Over Time with Online Learning (NN)')

    plt.tight_layout()
    plt.show()
