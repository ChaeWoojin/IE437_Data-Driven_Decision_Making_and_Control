import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna

# Define the Neural Network with Dropout and L2 Regularization
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout_rate):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)
        self.dropout = nn.Dropout(dropout_rate)
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

def load_data(file_path):
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

def objective(trial):
    file_path = '../data/spotify_personal_kaggle.csv'
    features_tensor, labels_tensor = load_data(file_path)

    input_dim = features_tensor.shape[1]
    hidden_dim1 = trial.suggest_int('hidden_dim1', 64, 256)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 32, 128)
    hidden_dim3 = trial.suggest_int('hidden_dim3', 16, 64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

    model = SimpleNN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = 32
    cumulative_regret = []
    total_regret = 0
    correct_predictions = 0
    total_predictions = 0
    winning_rate = []

    for i in range(0, len(features_tensor), batch_size):
        batch_features = features_tensor[i:i + batch_size]
        batch_labels = labels_tensor[i:i + batch_size]
        
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = (outputs.detach().numpy() >= 0.5).astype(int)
        batch_regret = sum(1 - (predictions.flatten() == batch_labels.numpy().flatten()))  # Regret is 1 if prediction is wrong
        total_regret += batch_regret
        cumulative_regret.append(total_regret)
        
        correct_predictions += sum(predictions.flatten() == batch_labels.numpy().flatten())
        total_predictions += len(batch_labels)
        current_winning_rate = correct_predictions / total_predictions
        winning_rate.append(current_winning_rate)
    
    return total_regret

def tune_hyperparameters():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)
    print("Best value (lowest cumulative regret): ", study.best_value)

    return study.best_params

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
    # plt.savefig('../results/nn_batch_performance.png')  # Save the plot as an image file
    plt.show()

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print(f"Best parameters found: {best_params}")

    # Load data and evaluate the model with the best hyperparameters
    file_path = '../data/spotify_personal_kaggle.csv'
    features_tensor, labels_tensor = load_data(file_path)

    model = SimpleNN(input_dim=features_tensor.shape[1], 
                     hidden_dim1=best_params['hidden_dim1'], 
                     hidden_dim2=best_params['hidden_dim2'], 
                     hidden_dim3=best_params['hidden_dim3'], 
                     dropout_rate=best_params['dropout_rate'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    cumulative_regret = []
    total_regret = 0
    correct_predictions = 0
    total_predictions = 0
    winning_rate = []

    for i in range(0, len(features_tensor), 32):
        batch_features = features_tensor[i:i + 32]
        batch_labels = labels_tensor[i:i + 32]
        
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = (outputs.detach().numpy() >= 0.5).astype(int)
        batch_regret = sum(1 - (predictions.flatten() == batch_labels.numpy().flatten()))  # Regret is 1 if prediction is wrong
        total_regret += batch_regret
        cumulative_regret.append(total_regret)
        
        correct_predictions += sum(predictions.flatten() == batch_labels.numpy().flatten())
        total_predictions += len(batch_labels)
        current_winning_rate = correct_predictions / total_predictions
        winning_rate.append(current_winning_rate)

    plot_results(cumulative_regret, winning_rate)

# {'hidden_dim1': 222, 'hidden_dim2': 127, 'hidden_dim3': 32, 'dropout_rate': 0.23258782979156775, 'lr': 0.006346732441805734, 'weight_decay': 3.857836400904553e-06}