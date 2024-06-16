import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import optuna

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

    return features_scaled, labels

def objective(trial):
    file_path = '../data/spotify_personal_kaggle.csv'
    features, labels = load_data(file_path)

    # Split data for training and evaluation
    initial_phase_size = 100
    contexts_train = features[:initial_phase_size]
    rewards_train = labels[:initial_phase_size]
    contexts_eval = features[initial_phase_size:]
    rewards_eval = labels[initial_phase_size:]

    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
    eta0 = trial.suggest_float('eta0', 1e-4, 1e-1, log=True)
    loss = trial.suggest_categorical('loss', ['log_loss', 'hinge'])

    model = SGDClassifier(loss=loss, learning_rate='constant', eta0=eta0, alpha=alpha, random_state=42)

    scaler = StandardScaler()
    contexts_train = scaler.fit_transform(contexts_train)
    model.partial_fit(contexts_train, rewards_train, classes=[0, 1])

    cumulative_regret = []
    total_regret = 0
    correct_predictions = 0
    total_predictions = 0
    winning_rate = []

    batch_size = 32

    contexts_eval = scaler.transform(contexts_eval)

    for i in range(0, len(contexts_eval), batch_size):
        context_batch = contexts_eval[i:i + batch_size]
        reward_batch = rewards_eval[i:i + batch_size]
        arms = model.predict(context_batch)
        rewards_observed = (arms == reward_batch).astype(int)
        model.partial_fit(context_batch, reward_batch)

        batch_regret = sum(1 - rewards_observed)
        total_regret += batch_regret
        cumulative_regret.append(total_regret)

        correct_predictions += sum(rewards_observed)
        total_predictions += len(reward_batch)
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
    plt.title('Cumulative Regret Over Time with Online Learning (SGD)')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(winning_rate) + 1), winning_rate)
    plt.xlabel('Batch Number')
    plt.ylabel('Winning Rate')
    plt.title('Winning Rate Over Time with Online Learning (SGD)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print(f"Best parameters found: {best_params}")

    # Load data and evaluate the model with the best hyperparameters
    file_path = '../data/spotify_personal_kaggle.csv'
    features, labels = load_data(file_path)

    model = SGDClassifier(loss=best_params['loss'], learning_rate='constant', eta0=best_params['eta0'], alpha=best_params['alpha'], random_state=42)

    scaler = StandardScaler()
    contexts_train = scaler.fit_transform(features[:100])
    rewards_train = labels[:100]
    model.partial_fit(contexts_train, rewards_train, classes=[0, 1])

    contexts_eval = scaler.transform(features[100:])
    rewards_eval = labels[100:]

    cumulative_regret = []
    total_regret = 0
    correct_predictions = 0
    total_predictions = 0
    winning_rate = []

    batch_size = 32

    for i in range(0, len(contexts_eval), batch_size):
        context_batch = contexts_eval[i:i + batch_size]
        reward_batch = rewards_eval[i:i + batch_size]
        arms = model.predict(context_batch)
        rewards_observed = (arms == reward_batch).astype(int)
        model.partial_fit(context_batch, reward_batch)

        batch_regret = sum(1 - rewards_observed)
        total_regret += batch_regret
        cumulative_regret.append(total_regret)

        correct_predictions += sum(rewards_observed)
        total_predictions += len(reward_batch)
        current_winning_rate = correct_predictions / total_predictions
        winning_rate.append(current_winning_rate)

    plot_results(cumulative_regret, winning_rate)

# Best parameters found: {'alpha': 4.171535856354112e-05, 'eta0': 0.013673752596489827, 'loss': 'log_loss'}