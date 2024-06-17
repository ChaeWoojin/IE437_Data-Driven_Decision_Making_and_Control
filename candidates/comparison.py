import matplotlib.pyplot as plt
from candidates import (
    NNOnlineLearning, 
    LinUCBAlgorithmBatch,
    SGDOnlineLearning
)

# Load and preprocess data
file_path = '../data/spotify_personal_kaggle.csv'

# Load data
sgd_model = SGDOnlineLearning(input_dim=15)
features, labels = sgd_model.load_data(file_path)

# Split data for initial learning phase and evaluation
initial_phase_size = 100
contexts_train = features[:initial_phase_size]
rewards_train = labels[:initial_phase_size]
contexts_eval = features[initial_phase_size:]
rewards_eval = labels[initial_phase_size:]

# SGD evaluation
sgd_model.initial_learning_phase(contexts_train, rewards_train, initial_phase_size=100)
sgd_cumulative_regret, sgd_winning_rate = sgd_model.evaluate(contexts_eval, rewards_eval)


# Neural Network evaluation
nn_model = NNOnlineLearning(input_dim=15)  # Assuming there are 15 features
features_tensor, labels_tensor = nn_model.load_data(file_path)
nn_model.initial_learning_phase(features_tensor[:100], labels_tensor[:100], initial_phase_size=100)
nn_cumulative_regret, nn_winning_rate = nn_model.evaluate(features_tensor[100:], labels_tensor[100:])

# LinUCB evaluation
linucb_model = LinUCBAlgorithmBatch(n_arms=2, n_features=15, alpha=1.0)
contexts, rewards = linucb_model.load_data(file_path)
linucb_model.initial_learning_phase(contexts_train, rewards_train, initial_phase_size=100)
linucb_cumulative_regret, linucb_winning_rate = linucb_model.evaluate(contexts_eval, rewards_eval, alpha=1.0)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(sgd_cumulative_regret) + 1), sgd_cumulative_regret, label='SGD')
plt.plot(range(1, len(nn_cumulative_regret) + 1), nn_cumulative_regret, label='NN')
plt.plot(range(1, len(linucb_cumulative_regret) + 1), linucb_cumulative_regret, label='LinUCB')
plt.xlabel('Batch Number')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret Over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(sgd_winning_rate) + 1), sgd_winning_rate, label='SGD')
plt.plot(range(1, len(nn_winning_rate) + 1), nn_winning_rate, label='NN')
plt.plot(range(1, len(linucb_winning_rate) + 1), linucb_winning_rate, label='LinUCB')
plt.xlabel('Batch Number')
plt.ylabel('Winning Rate')
plt.title('Winning Rate Over Time')
plt.legend()

plt.tight_layout()
plt.savefig('../results/online_learning_cumulative_regret_winning_rate.png')  # Save the plot as an image file
plt.show()

# Print final winning rates
print(f"Final Winning Rate for SGD: {sgd_winning_rate[-1]:.4f}")
print(f"Final Winning Rate for NN: {nn_winning_rate[-1]:.4f}")
print(f"Final Winning Rate for LinUCB: {linucb_winning_rate[-1]:.4f}")