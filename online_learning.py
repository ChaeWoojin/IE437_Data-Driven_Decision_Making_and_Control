import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

# Ensure a safe backend for matplotlib
import matplotlib
matplotlib.use('agg')  # Use a non-interactive backend

# Load the dataset
file_path = './music_pool.csv'
music_data = pd.read_csv(file_path)

# Features and target
features = music_data[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']]
target = music_data['like']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split to simulate the online learning scenario
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Prepare the data with polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize the Gradient Boosting model
gb_model_poly = GradientBoostingClassifier(random_state=42)

# Shuffle the training data
X_train_poly, y_train = shuffle(X_train_poly, y_train, random_state=42)

# Function to train the model with larger batch sizes
def train_model_with_larger_batches(model, X, y, batch_size=50):
    n_batches = len(X) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]
        model.fit(X_batch, y_batch)
    return model

# Function to evaluate the model
def evaluate_model(model, X, y):
    accuracy = model.score(X, y)
    return accuracy

# Simulate online learning with larger batches
batch_size = 50
accuracies = []

# Train and evaluate the model in an online learning fashion
for start in range(0, len(X_train_poly), batch_size):
    end = start + batch_size
    X_batch = X_train_poly[start:end]
    y_batch = y_train[start:end]
    gb_model_poly = train_model_with_larger_batches(gb_model_poly, X_batch, y_batch, batch_size=batch_size)
    accuracy = evaluate_model(gb_model_poly, X_test_poly, y_test)
    accuracies.append(accuracy)

# Plot the accuracies over the online learning iterations
plt.plot(accuracies)
plt.xlabel('Batch Iteration')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Online Learning Iterations')
plt.savefig('model_accuracy.png')  # Save the plot as an image file

# Display the saved plot
from IPython.display import Image
Image(filename='model_accuracy.png')

final_accuracy = accuracies[-1]  # Final accuracy after online learning
print("Final Accuracy after Online Learning:", final_accuracy)
