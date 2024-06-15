import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tqdm import tqdm

# Load the dataset
file_path = '../data/spotify_personal_kaggle.csv'
music_data = pd.read_csv(file_path)

# Features and target
features = music_data[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']]
target = music_data['Like']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split to simulate the online learning scenario
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Define the neural network model
def create_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize the model
input_dim = X_train.shape[1]
model = create_model(input_dim)

# Parameters for epsilon-greedy strategy
epsilon = 0.1

# Function to recommend a slate of songs
def recommend_slate(model, X, epsilon=0.1, slate_size=5):
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random slate
        return random.sample(range(X.shape[0]), slate_size)
    else:
        # Exploit: choose the best slate based on model predictions
        predictions = model.predict(X)
        return np.argsort(predictions.ravel())[-slate_size:].tolist()

# Function to simulate user feedback
def simulate_feedback(slate, true_likes):
    feedback = []
    for song in slate:
        feedback.append(true_likes[song])
    return feedback

# Function to update the model with new data
def update_model(model, X, y, epochs=1, batch_size=32):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Simulate online learning
n_iterations = 100
slate_size = 10
accuracies = []

for iteration in tqdm(range(n_iterations)):
    # Shuffle the training data and reset indices
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    y_train = y_train.reset_index(drop=True)
    
    # Recommend a slate of songs
    slate = recommend_slate(model, X_train, epsilon, slate_size)
    
    # Simulate user feedback
    feedback = simulate_feedback(slate, y_train)
    
    # Prepare the batch for model update
    X_batch = X_train[slate]
    y_batch = np.array(feedback)
    
    # Update the model with the new batch
    model = update_model(model, X_batch, y_batch, epochs=1, batch_size=len(slate))
    
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(loss, accuracy)
    accuracies.append(accuracy)

# Plot the accuracies over iterations
plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Online Learning Iterations')
plt.show()
# plt.savefig('model_accuracy_deep_learning.png')  # Save the plot as an image file

# Display the saved plot
# from IPython.display import Image
# Image(filename='model_accuracy_deep_learning.png')

# Display final accuracy
final_accuracy = accuracies[-1]
print("Final Accuracy after Online Learning:", final_accuracy)