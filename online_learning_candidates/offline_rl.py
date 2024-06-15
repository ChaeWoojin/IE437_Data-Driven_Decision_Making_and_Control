from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../data/spotify_personal_kaggle.csv'
music_data = pd.read_csv(file_path)

# Features and target
features = music_data[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']]
target = music_data['Like']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define state and action spaces
n_songs = features_scaled.shape[0]
n_features = features_scaled.shape[1]

# Initialize Q-table
Q_table = np.zeros((n_songs, n_songs))  # Q-table for state-action pairs

# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Function to recommend a slate of songs
def recommend_slate(Q_table, state, epsilon=0.1, slate_size=5):
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random slate
        return random.sample(range(n_songs), slate_size)
    else:
        # Exploit: choose the best slate based on Q-values
        return np.argsort(Q_table[state, :])[-slate_size:].tolist()

# Function to simulate user feedback
def simulate_feedback(slate, true_likes):
    feedback = []
    for song in slate:
        feedback.append(true_likes[song])
    return feedback

# Function to update Q-values
def update_q_table(Q_table, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(Q_table[next_state, :])
    td_target = reward + gamma * Q_table[next_state, best_next_action]
    td_error = td_target - Q_table[state, action]
    Q_table[state, action] += alpha * td_error

# Simulate online learning
n_iterations = 1000
slate_size = 5
accuracies = []
history = []

for iteration in range(n_iterations):
    # Recommend a slate of songs
    state = random.randint(0, n_songs - 1)  # Random initial state
    slate = recommend_slate(Q_table, state, epsilon, slate_size)
    
    # Simulate user feedback
    feedback = simulate_feedback(slate, target.values)
    
    # Update Q-values
    for song, reward in zip(slate, feedback):
        next_state = song  # Next state is the song itself
        update_q_table(Q_table, state, song, reward, next_state, alpha, gamma)
        state = next_state  # Move to the next state
    
    # Track accuracy (for simplicity, we use the mean feedback as a proxy for accuracy)
    accuracy = np.mean(feedback)
    accuracies.append(accuracy)
    history.append(slate)

# Plot the accuracies over iterations
plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Average Feedback (Accuracy)')
plt.title('Model Accuracy Over Online Learning Iterations')
plt.show()

# Display final accuracy
final_accuracy = accuracies[-1]
print("Final Accuracy after Online Learning:", final_accuracy)