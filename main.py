import pandas as pd
import numpy as np
from contextualbandits.online import LinUCB
import matplotlib.pyplot as plt

# Load the dataset
file_name = './data/spotify_personal_kaggle.csv'
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

# Initialize the LinUCB bandit
n_arms = 2  # In this case, we have two possible rewards: Like (1) or Not Like (0)
alpha = 1.0  # Exploration parameter
linucb = LinUCB(nchoices=n_arms, alpha=alpha)

# Initial learning phase with the first 100 tracks
initial_phase_size = 100
for i in range(initial_phase_size):
    context = contexts[i].reshape(1, -1)
    chosen_arm = linucb.predict(context)
    actual_reward = rewards[i]
    reward = 1 if chosen_arm == actual_reward else 0
    linucb.partial_fit(context, np.array([chosen_arm]), np.array([reward]))
    print(f"Initial Phase - Iteration {i+1}, Chosen arm: {chosen_arm}, Reward: {reward}")

# Function to calculate cumulative regret
def calculate_cumulative_regret(contexts, rewards, model):
    cumulative_regret = []
    total_regret = 0
    optimal_reward = max(rewards)
    
    for i in range(len(contexts)):
        context = contexts[i].reshape(1, -1)
        chosen_arm = model.predict(context)
        actual_reward = rewards[i]
        reward = 1 if chosen_arm == actual_reward else 0
        regret = optimal_reward - reward
        total_regret += regret
        cumulative_regret.append(total_regret)
    
    return cumulative_regret

# Sequentially update the model based on feedback for the remaining tracks
cumulative_regret = []
optimal_reward = max(rewards)
total_regret = 0
correct_predictions = 0

chosen_arms = []
obtained_rewards = []

for i in range(initial_phase_size, len(contexts)):
    context = contexts[i].reshape(1, -1)
    
    # Select an action using LinUCB
    chosen_arm = linucb.predict(context)
    
    # Get the actual reward
    actual_reward = rewards[i]
    reward = 1 if chosen_arm == actual_reward else 0
    
    # Track chosen arms and rewards
    chosen_arms.append(chosen_arm)
    obtained_rewards.append(reward)
    
    # Update the model with the observed reward
    linucb.partial_fit(context, np.array([chosen_arm]), np.array([reward]))
    
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
