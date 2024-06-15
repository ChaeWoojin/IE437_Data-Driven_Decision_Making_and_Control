import pandas as pd
import numpy as np
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

# Initialize parameters for ε-Greedy
epsilon = 0.1
n_arms = 2
counts = np.zeros(n_arms)  # Number of times each arm has been selected
values = np.zeros(n_arms)  # Average reward for each arm

# Function to select an action using ε-Greedy
def select_action(epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_arms)  # Explore
    else:
        return np.argmax(values)  # Exploit

# Function to update the action values
def update_values(chosen_arm, reward):
    counts[chosen_arm] += 1
    n = counts[chosen_arm]
    value = values[chosen_arm]
    new_value = ((n - 1) / n) * value + (1 / n) * reward
    values[chosen_arm] = new_value

# Initial learning phase with the first 100 tracks
initial_phase_size = 100
for i in range(initial_phase_size):
    chosen_arm = select_action(epsilon)
    actual_reward = rewards[i]
    reward = 1 if chosen_arm == actual_reward else 0
    update_values(chosen_arm, reward)
    print(f"Initial Phase - Iteration {i+1}, Chosen arm: {chosen_arm}, Reward: {reward}")

# Sequentially update the model based on feedback for the remaining tracks
cumulative_regret = []
optimal_reward = max(rewards)
total_regret = 0
correct_predictions = 0

chosen_arms = []
obtained_rewards = []

for i in range(initial_phase_size, len(contexts)):
    chosen_arm = select_action(epsilon)
    actual_reward = rewards[i]
    reward = 1 if chosen_arm == actual_reward else 0
    
    # Track chosen arms and rewards
    chosen_arms.append(chosen_arm)
    obtained_rewards.append(reward)
    
    # Update values
    update_values(chosen_arm, reward)
    
    # Calculate regret
    regret = optimal_reward - reward
    total_regret += regret
    cumulative_regret.append(total_regret)

    # Update correct predictions
    if reward == 1:
        correct_predictions += 1

    print(f"Evaluation Phase - Iteration {i+1}, Chosen arm: {chosen_arm}, Reward: {reward}, Regret: {regret}")

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
