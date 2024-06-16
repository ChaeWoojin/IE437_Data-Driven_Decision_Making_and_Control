import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from candidates.linucb import LinUCBAlgorithmBatch

# Load the dataset and shuffle
file_name = '../data/spotify_personal_kaggle.csv'
df = pd.read_csv(file_name)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define features and prepare data
features = ['Like', 'Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 
            'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
            'Valence', 'Tempo', 'Duration_ms', 'Year', 'Popularity', 'Explicit']
data = df[features]
contexts = data.drop(columns=['Like']).values
rewards = data['Like'].values
initial_phase_size = 100
batch_size = 32

contexts_train = contexts[:initial_phase_size]
rewards_train = rewards[:initial_phase_size]
contexts_eval = contexts[initial_phase_size:]
rewards_eval = rewards[initial_phase_size:]

def evaluate_linucb_for_alphas(alphas):
    results = {}
    for alpha in alphas:
        algo = LinUCBAlgorithmBatch(n_arms=2, n_features=contexts.shape[1], alpha=alpha, batch_size=batch_size)
        
        # Initial training phase with the first 100 observations
        algo.update(contexts_train, np.random.choice(2, size=initial_phase_size), rewards_train)
        
        cumulative_regret = []
        total_regret = 0
        optimal_reward = max(rewards_eval)
        
        # Online updating and evaluation phase
        for i in range(0, len(contexts_eval), batch_size):
            context_batch = contexts_eval[i:i+batch_size]
            reward_batch = rewards_eval[i:i+batch_size]
            arms = [algo.select_arm(context) for context in context_batch]
            rewards_observed = [1 if arm == reward else 0 for arm, reward in zip(arms, reward_batch)]
            algo.update(context_batch, arms, rewards_observed)
            
            batch_regret = sum(optimal_reward - reward for reward in rewards_observed)
            total_regret += batch_regret
            cumulative_regret.extend([total_regret] * len(context_batch))
        
        results[alpha] = cumulative_regret
    return results

alphas_to_evaluate = [0.1, 0.5, 1.0, 2.0, 5.0]
results = evaluate_linucb_for_alphas(alphas_to_evaluate)

# Plot cumulative regret comparison
plt.figure(figsize=(10, 6))

for alpha, cumulative_regret in results.items():
    plt.plot(cumulative_regret, label=f'LinUCB (α={alpha})')

plt.title('Cumulative Regret Over Time for Different Alphas')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.grid(True)
plt.show()
