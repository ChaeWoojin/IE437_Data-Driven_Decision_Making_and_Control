import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta

# Load the dataset
file_path = '../data/spotify_personal_kaggle.csv'
music_data = pd.read_csv(file_path)

# Features and target
target = music_data['like']

# Define the number of songs
n_songs = len(target)

# Initialize Beta priors for each song (a=1, b=1 is a uniform prior)
priors = np.ones((n_songs, 2))

# Parameters for epsilon-greedy strategy and fairness constraint
epsilon = 0.1
alpha = 0.03  # Minimum probability to ensure fairness

# Initialize the list of liked songs
liked_songs = set()

# Function to recommend a slate of songs
def recommend_slate(priors, liked_songs, epsilon=0.1, alpha=0.05, slate_size=5):
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random slate
        available_songs = list(set(range(n_songs)) - liked_songs)
        if len(available_songs) < slate_size:
            return available_songs
        return random.sample(available_songs, slate_size)
    else:
        # Exploit: choose the slate with the highest expected probability of being liked
        mean_prob = priors[:, 0] / (priors[:, 0] + priors[:, 1])
        # Apply fairness constraint
        fair_prob = np.maximum(mean_prob, alpha)
        available_songs = np.array(list(set(range(n_songs)) - liked_songs))
        best_slate = available_songs[np.argsort(fair_prob[available_songs])[-slate_size:]].tolist()
        return best_slate

# Function to simulate user feedback
def simulate_feedback(slate, true_likes):
    feedback = []
    for song in slate:
        feedback.append(true_likes[song])
    return feedback

# Function to update Beta priors with feedback and liked songs
def update_priors(priors, slate, feedback, liked_songs):
    for song, reward in zip(slate, feedback):
        priors[song, 0] += reward
        priors[song, 1] += (1 - reward)
        if reward == 1:
            liked_songs.add(song)

# Simulate online learning
n_iterations = 100
slate_size = 5
accuracies = []

for iteration in range(n_iterations):
    # Recommend a slate of songs
    slate = recommend_slate(priors, liked_songs, epsilon, alpha, slate_size)
    
    # Simulate user feedback
    feedback = simulate_feedback(slate, target.values)
    
    # Update priors with feedback and liked songs
    update_priors(priors, slate, feedback, liked_songs)
    
    # Track accuracy (for simplicity, we use the mean feedback as a proxy for accuracy)
    accuracy = np.mean(feedback)
    accuracies.append(accuracy)

# Plot the accuracies over iterations
plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Average Feedback (Accuracy)')
plt.title('Model Accuracy Over Online Learning Iterations with Fairness Constraint')
plt.show()
# plt.savefig('./results/model_accuracy_bayesian_fairness_(size_100_epsilon_0.03).png')  # Save the plot as an image file

# Display the saved plot
# from IPython.display import Image
# Image(filename='./results/model_accuracy_bayesian_fairness_(size_100_epsilon_0.03).png')

# Display final accuracy
final_accuracy = accuracies[-1]
print("Final Accuracy after Online Learning:", final_accuracy)
