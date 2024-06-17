import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

class SGDOnlineLearning:
    def __init__(self, input_dim, alpha=3.2e-06, eta0=0.01, batch_size=32):
        self.input_dim = input_dim
        self.alpha = alpha
        self.eta0 = eta0
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=self.eta0, alpha=self.alpha, random_state=42)
        self.initialized = False

    def select_arm(self, context):
        if not self.initialized:
            self.model.partial_fit(context.reshape(1, -1), [0], classes=[0, 1])
            self.initialized = True
        prob = self.model.predict_proba(context.reshape(1, -1))[0][1]
        return 1 if prob >= 0.5 else 0

    def update(self, contexts, rewards):
        # contexts = self.scaler.transform(contexts)
        self.model.partial_fit(contexts, rewards)

    def initial_learning_phase(self, contexts, rewards, initial_phase_size):
        contexts = self.scaler.fit_transform(contexts[:initial_phase_size])
        self.model.partial_fit(contexts, rewards[:initial_phase_size], classes=[0, 1])
        self.initialized = True

    def evaluate(self, contexts_eval, rewards_eval):
        cumulative_regret = []
        total_regret = 0
        correct_predictions = 0
        total_predictions = 0
        winning_rate = []

        contexts_eval = self.scaler.transform(contexts_eval)

        for i in range(0, len(contexts_eval), self.batch_size):
            context_batch = contexts_eval[i:i+self.batch_size]
            context_batch = contexts_eval[i:i + self.batch_size]
            reward_batch = rewards_eval[i:i + self.batch_size]
            arms = [self.select_arm(context) for context in context_batch]
            rewards_observed = (arms == reward_batch).astype(int)
            self.model.partial_fit(context_batch, reward_batch)

            batch_regret = sum(1 - rewards_observed)
            total_regret += batch_regret
            cumulative_regret.append(total_regret)

            correct_predictions += sum(rewards_observed)
            total_predictions += len(reward_batch)
            current_winning_rate = correct_predictions / total_predictions
            winning_rate.append(current_winning_rate)

        return cumulative_regret, winning_rate

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        feature_columns = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
                           'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
                           'Valence', 'Tempo', 'Duration_ms', 'Year', 'Popularity', 'Explicit']
        data = data.dropna(subset=['Like'])
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        features = data[feature_columns]
        labels = data['Like'].values

        features_scaled = self.scaler.fit_transform(features)

        return features_scaled, labels
