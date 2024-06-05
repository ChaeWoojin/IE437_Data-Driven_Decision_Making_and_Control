from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Load the dataset
file_path = './music_pool.csv'
music_data = pd.read_csv(file_path)

# Features and target
features = music_data[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']]
target = music_data['like']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define state and action spaces
n_songs = features_scaled.shape[0]

# Parameters for epsilon-greedy strategy
epsilon = 0.1

# Initialize history
history = defaultdict(list)

# Function to recommend a slate of songs
def recommend_slate(history, epsilon=0.1, slate_size=5):
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random slate
        return random.sample(range(n_songs), slate_size)
    else:
        # Exploit: choose the best slate based on history
        slate_scores = np.zeros(n_songs)
        for song in range(n_songs):
            if song in history and len(history[song]) > 0:
                slate_scores[song] = np.mean(history[song])
        best_slate = np.argsort(slate_scores)[-slate_size:].tolist()
        return best_slate

# Function to simulate user feedback
def simulate_feedback(slate, true_likes):
    feedback = []
    for song in slate:
        feedback.append(true_likes[song])
    return feedback

# Function to update history with feedback
def update_history(history, slate, feedback):
    for song, reward in zip(slate, feedback):
        history[song].append(reward)

# Simulate online learning
n_iterations = 1000
slate_size = 5
accuracies = []

for iteration in range(n_iterations):
    # Recommend a slate of songs
    slate = recommend_slate(history, epsilon, slate_size)
    
    # Simulate user feedback
    feedback = simulate_feedback(slate, target.values)
    
    # Update history with feedback
    update_history(history, slate, feedback)
    
    # Track accuracy (for simplicity, we use the mean feedback as a proxy for accuracy)
    accuracy = np.mean(feedback)
    accuracies.append(accuracy)

# Plot the accuracies over iterations
plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Average Feedback (Accuracy)')
plt.title('Model Accuracy Over Online Learning Iterations')
plt.savefig('model_accuracy_model_free.png')  # Save the plot as an image file

# Display the saved plot
from IPython.display import Image
Image(filename='model_accuracy_model_free.png')

# Display final accuracy
final_accuracy = accuracies[-1]
print("Final Accuracy after Online Learning:", final_accuracy)
