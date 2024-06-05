import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta

# Load the dataset
file_path = './music_pool.csv'
music_data = pd.read_csv(file_path)

# Features and target
target = music_data['like']

# Define the number of songs
n_songs = len(target)

# Initialize Beta priors for each song (a=1, b=1 is a uniform prior)
priors = np.ones((n_songs, 2))

# Parameters for epsilon-greedy strategy
epsilon = 0.1

# Function to recommend a slate of songs
def recommend_slate(priors, epsilon=0.1, slate_size=5):
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random slate
        return random.sample(range(n_songs), slate_size)
    else:
        # Exploit: choose the slate with the highest expected probability of being liked
        mean_prob = priors[:, 0] / (priors[:, 0] + priors[:, 1])
        return np.argsort(mean_prob)[-slate_size:].tolist()

# Function to simulate user feedback
def simulate_feedback(slate, true_likes):
    feedback = []
    for song in slate:
        feedback.append(true_likes[song])
    return feedback

# Function to update Beta priors with feedback
def update_priors(priors, slate, feedback):
    for song, reward in zip(slate, feedback):
        priors[song, 0] += reward
        priors[song, 1] += (1 - reward)

# Simulate online learning
n_iterations = 50
slate_size = 5
accuracies = []

for iteration in range(n_iterations):
    # Recommend a slate of songs
    slate = recommend_slate(priors, epsilon, slate_size)
    
    # Simulate user feedback
    feedback = simulate_feedback(slate, target.values)
    
    # Update priors with feedback
    update_priors(priors, slate, feedback)
    
    # Track accuracy (for simplicity, we use the mean feedback as a proxy for accuracy)
    accuracy = np.mean(feedback)
    accuracies.append(accuracy)

# Plot the accuracies over iterations
plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Average Feedback (Accuracy)')
plt.title('Model Accuracy Over Online Learning Iterations')
plt.savefig('model_accuracy_bayesian.png')  # Save the plot as an image file

# Display the saved plot
from IPython.display import Image
Image(filename='model_accuracy_bayesian.png')

# Display final accuracy
final_accuracy = accuracies[-1]
print("Final Accuracy after Online Learning:", final_accuracy)
