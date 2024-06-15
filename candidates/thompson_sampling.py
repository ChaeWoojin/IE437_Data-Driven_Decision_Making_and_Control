import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Load the dataset
file_name = '../data/spotify_personal_kaggle.csv'
df = pd.read_csv(file_name)

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define features
features = ['Like', 'Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 
            'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
            'Valence', 'Tempo', 'Duration_ms', 'Year', 'Popularity', 'Explicit']

data = df[features]

# Prepare the context and rewards
contexts = data.drop(columns=['Like']).values
rewards = data['Like'].values

# Initialize Thompson Sampling parameters
n_arms = 2  # In this case, we have two possible rewards: Like (1) or Not Like (0)
alpha_beta = np.ones((n_arms, 2))  # Beta distribution parameters for each arm

# Function to select an arm using Thompson Sampling
def select_arm_thompson_sampling(alpha_beta):
    samples = [beta.rvs(a, b) for a, b in alpha_beta]
    return np.argmax(samples)

# Initial learning phase with the first 100 tracks
initial_phase_size = 100
for i in range(initial_phase_size):
    context = contexts[i].reshape(1, -1)
    chosen_arm = select_arm_thompson_sampling(alpha_beta)
    actual_reward = rewards[i]
    reward = 1 if chosen_arm == actual_reward else 0
    alpha_beta[chosen_arm, 0] += reward  # Update beta distribution parameters
    alpha_beta[chosen_arm, 1] += 1 - reward
    print(f"Initial Phase - Iteration {i+1}, Chosen arm: {chosen_arm}, Reward: {reward}")

# Sequentially update the model based on feedback for the remaining tracks
cumulative_regret = []
optimal_reward = max(rewards)
total_regret = 0
correct_predictions = 0

chosen_arms = []
obtained_rewards = []

for i in range(initial_phase_size, len(contexts)):
    context = contexts[i].reshape(1, -1)
    
    # Select an action using Thompson Sampling
    chosen_arm = select_arm_thompson_sampling(alpha_beta)
    
    # Get the actual reward
    actual_reward = rewards[i]
    reward = 1 if chosen_arm == actual_reward else 0
    
    # Track chosen arms and rewards
    chosen_arms.append(chosen_arm)
    obtained_rewards.append(reward)
    
    # Update beta distribution parameters
    alpha_beta[chosen_arm, 0] += reward
    alpha_beta[chosen_arm, 1] += 1 - reward
    
    # Calculate regret
    regret = optimal_reward - reward
    total_regret += regret
    cumulative_regret.append(total_regret)

    # Update correct predictions
    if reward == 1:
        correct_predictions += 1

    print(f"Evaluation Phase - Iteration {i+1 - initial_phase_size}, Chosen arm: {chosen_arm}, Reward: {reward}, Regret: {regret}")

# Calculate winning rate
winning_rate = correct_predictions / (len(contexts) - initial_phase_size) * 100

# Plot cumulative regret
plt.figure(figsize=(10, 6))
plt.plot(cumulative_regret, marker='o')
plt.title('Cumulative Regret Over Time')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Regret')
plt.grid(True)
plt.show()

# Additional analysis: plot chosen arms and obtained rewards
plt.figure(figsize=(10, 6))
plt.plot(chosen_arms, marker='o', label='Chosen Arms')
plt.plot(obtained_rewards, marker='x', label='Obtained Rewards')
plt.title('Chosen Arms and Obtained Rewards Over Time')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

print(f"Bandit optimization completed. Winning rate: {winning_rate:.2f}%")
