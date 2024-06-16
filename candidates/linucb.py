import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LinUCBAlgorithmBatch:
    def __init__(self, n_arms, n_features, alpha=1.0, batch_size=32):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.batch_size = batch_size
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        self.batch_data = []

    def select_arm(self, context):
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = np.dot(A_inv, self.b[arm])
            p[arm] = np.dot(theta, context) + self.alpha * np.sqrt(np.dot(context.T, np.dot(A_inv, context)))
        return np.argmax(p)

    def update(self, contexts, arms, rewards):
        for context, arm, reward in zip(contexts, arms, rewards):
            self.batch_data.append((context, arm, reward))
            if len(self.batch_data) >= self.batch_size:
                for c, a, r in self.batch_data:
                    self.A[a] += np.outer(c, c)
                    self.b[a] += r * c
                self.batch_data = []

    def initial_learning_phase(self, contexts, rewards, initial_phase_size):
        for i in range(initial_phase_size):
            context = contexts[i]
            arm = self.select_arm(context)
            reward = rewards[i]
            self.update([context], [arm], [reward])

    def evaluate(self, contexts_eval, rewards_eval, alpha):
        cumulative_regret = []
        total_regret = 0
        correct_predictions = 0
        total_predictions = 0
        winning_rate = []
        optimal_reward = max(rewards_eval)
        
        for i in range(0, len(contexts_eval), self.batch_size):
            context_batch = contexts_eval[i:i+self.batch_size]
            reward_batch = rewards_eval[i:i+self.batch_size]
            arms = [self.select_arm(context) for context in context_batch]
            rewards_observed = [1 if arm == reward else 0 for arm, reward in zip(arms, reward_batch)]
            self.update(context_batch, arms, rewards_observed)
            
            batch_regret = sum(optimal_reward - reward for reward in rewards_observed)
            total_regret += batch_regret
            cumulative_regret.append(total_regret)
            
            correct_predictions += sum(1 if arm == reward else 0 for arm, reward in zip(arms, reward_batch))
            total_predictions += len(context_batch)
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

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        return features_scaled, labels
