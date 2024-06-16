import matplotlib.pyplot as plt
from candidates import NNOnlineLearning, LinUCBAlgorithmBatch

# Load and preprocess data
file_path = '../data/spotify_personal_kaggle.csv'

# Neural Network evaluation
nn_model = NNOnlineLearning(input_dim=15)  # Assuming there are 15 features
features_tensor, labels_tensor = nn_model.load_data(file_path)
nn_cumulative_regret, nn_winning_rate = nn_model.evaluate(features_tensor[100:], labels_tensor[100:])

# LinUCB evaluation
linucb_model = LinUCBAlgorithmBatch(n_arms=2, n_features=15, alpha=1.0)
contexts, rewards = linucb_model.load_data(file_path)
initial_phase_size = 100
contexts_train = contexts[:initial_phase_size]
rewards_train = rewards[:initial_phase_size]
contexts_eval = contexts[initial_phase_size:]
rewards_eval = rewards[initial_phase_size:]
linucb_cumulative_regret, linucb_winning_rate = linucb_model.evaluate(contexts_train, rewards_train, contexts_eval, rewards_eval, alpha=1.0)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(nn_cumulative_regret) + 1), nn_cumulative_regret, label='NN')
plt.plot(range(1, len(linucb_cumulative_regret) + 1), linucb_cumulative_regret, label='LinUCB')
plt.xlabel('Batch Number')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret Over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(nn_winning_rate) + 1), nn_winning_rate, label='NN')
plt.plot(range(1, len(linucb_winning_rate) + 1), linucb_winning_rate, label='LinUCB')
plt.xlabel('Batch Number')
plt.ylabel('Winning Rate')
plt.title('Winning Rate Over Time')
plt.legend()

plt.tight_layout()
plt.show()
