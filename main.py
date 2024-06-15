import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from contextualbandits.online import LinUCB
import matplotlib.pyplot as plt

# Load the dataset
file_name = './data/spotify_personal_kaggle.csv'
df = pd.read_csv(file_name)

# Define features
features = ['Like', 'Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 
            'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
            'Valence', 'Tempo', 'Duration_ms', 'Year', 'Popularity', 'Explicit']

data = df[features]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data, data['Like'], test_size=0.4, random_state=5)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=5)

# Prepare the context and rewards
contexts_train = X_train.drop(columns=['Like']).values
rewards_train = X_train['Like'].values
contexts_valid = X_valid.drop(columns=['Like']).values
rewards_valid = X_valid['Like'].values
contexts_test = X_test.drop(columns=['Like']).values
rewards_test = X_test['Like'].values

# Initialize the LinUCB bandit
n_arms = 2  # In this case, we have two possible rewards: Like (1) or Not Like (0)
linucb = LinUCB(nchoices=n_arms, alpha=1.0)

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

# Train the model and calculate cumulative regret on training data
cumulative_regret_train = []
optimal_reward = max(rewards_train)
total_regret = 0

for i in range(len(contexts_train)):
    context = contexts_train[i].reshape(1, -1)
    
    # Select an action using LinUCB
    chosen_arm = linucb.predict(context)
    
    # Get the actual reward
    actual_reward = rewards_train[i]
    reward = 1 if chosen_arm == actual_reward else 0
    
    # Update the model with the observed reward
    linucb.partial_fit(context, np.array([chosen_arm]), np.array([reward]))
    
    # Calculate regret
    regret = optimal_reward - reward
    total_regret += regret
    cumulative_regret_train.append(total_regret)

    print(f"Iteration {i+1}, Chosen arm: {chosen_arm}, Reward: {reward}, Regret: {regret}")

# Calculate cumulative regret on validation data
cumulative_regret_valid = calculate_cumulative_regret(contexts_valid, rewards_valid, linucb)

# Calculate cumulative regret on test data
cumulative_regret_test = calculate_cumulative_regret(contexts_test, rewards_test, linucb)

# Plot cumulative regret
plt.figure(figsize=(10, 6))
plt.plot(cumulative_regret_train, marker='o', label='Train')
plt.plot(cumulative_regret_valid, marker='o', label='Validation')
plt.plot(cumulative_regret_test, marker='o', label='Test')
plt.title('Cumulative Regret Over Time')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.grid(True)
plt.show()

print("Bandit optimization completed.")
