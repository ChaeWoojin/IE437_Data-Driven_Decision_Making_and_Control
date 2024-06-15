from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../data/spotify_personal_kaggle.csv'
music_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
music_data.head()

# Features and target
features = music_data[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']]
target = music_data['Like']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split to simulate the online learning scenario
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Convert to DataFrame for batch processing
train_data = pd.DataFrame(X_train, columns=features.columns)
train_data['like'] = y_train.values

test_data = pd.DataFrame(X_test, columns=features.columns)
test_data['like'] = y_test.values

train_data.head()

# Initialize the logistic regression model with stochastic gradient descent
model = SGDClassifier(loss='log_loss', random_state=42)

# Function to train the model incrementally
def train_model(model, data, batch_size=10):
    for start in range(0, len(data), batch_size):
        end = start + batch_size
        batch = data.iloc[start:end]
        X_batch = batch.drop(columns=['like'])
        y_batch = batch['like']
        model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
    return model

# Initial training with the first batch
initial_batch_size = 10
model = train_model(model, train_data, batch_size=initial_batch_size)

# Function to evaluate the model
def evaluate_model(model, data):
    X = data.drop(columns=['like'])
    y = data['like']
    accuracy = model.score(X, y)
    return accuracy

# Evaluate the model on the test set after initial training
initial_accuracy = evaluate_model(model, test_data)
initial_accuracy

# Continue training the model incrementally with more batches
def online_learning(model, train_data, test_data, batch_size=10):
    accuracies = []
    for i in range(0, len(train_data), batch_size):
        batch = train_data.iloc[i:i+batch_size]
        model = train_model(model, batch, batch_size=batch_size)
        accuracy = evaluate_model(model, test_data)
        accuracies.append(accuracy)
    return model, accuracies

# Perform online learning
model, accuracies = online_learning(model, train_data, test_data, batch_size=10)


# Plot the accuracies over the online learning iterations
plt.plot(accuracies)
plt.xlabel('Batch Iteration')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Online Learning Iterations')
plt.show()
# plt.savefig('../results/SGD_Classifier.png')  # Save the plot as an image file

# Display the saved plot
# from IPython.display import Image
# Image(filename='../results/SGD_Classifier.png')

# Display final accuracy
final_accuracy = accuracies[-1]
print("Final Accuracy after Online Learning:", final_accuracy)