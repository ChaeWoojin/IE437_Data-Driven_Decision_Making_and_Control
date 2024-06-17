# Bandit Algorithm Performance Comparison

## Introduction

This project implements and compares various online learning algorithms for bandit problems. Among the algorithms, we specifically focus on comparing the performance of the following:

1. LinUCB
2. SGD Online Learning
3. Neural Network Online Learning

## Algorithms Implemented

### LinUCB
LinUCB is a contextual bandit algorithm that uses a linear model to estimate the expected reward for each arm. It selects arms based on the upper confidence bound, balancing exploration and exploitation.

### SGD Online Learning
The SGD (Stochastic Gradient Descent) Online Learning algorithm uses an SGD classifier to make predictions in an online learning setup. It continuously updates the model with new data to improve performance. We tuned the hyperparameters using `optuna` to minimize cumulative regret.

### Neural Network Online Learning
The Neural Network (NN) Online Learning algorithm uses a simple feedforward neural network to make predictions. The model is updated in an online learning manner, with dropout and L2 regularization to prevent overfitting. We tuned the hyperparameters using `optuna` to minimize cumulative regret.

## Data
The data used in this project is from the Spotify dataset, which includes features such as `Danceability`, `Energy`, `Key`, `Loudness`, `Mode`, `Speechiness`, `Acousticness`, `Instrumentalness`, `Liveness`, `Valence`, `Tempo`, `Duration_ms`, `Year`, `Popularity`, and `Explicit`.

## Performance Comparison

### Cumulative Regret Over Time
The following plot shows the cumulative regret over time for each algorithm. Lower cumulative regret indicates better performance.

![Cumulative Regret Over Time (Among Conventional Bandit Algorithms)](results/bandit_models_cumulative_regret.png)

![Cumulative Regret Over Time (Best Bandit Algorithm (LinUCB) vs NN vs SGD Classifier)](results/online_learning_cumulative_regret_winning_rate.png)

### Winning Rate Over Time
The following plot shows the winning rate over time for each algorithm. A higher winning rate indicates better performance.

![Winning Rate Over Time](images/winning_rate.png)

## Results
In our experiments, we observed that the LinUCB algorithm consistently outperformed both the SGD Online Learning and Neural Network Online Learning algorithms. The LinUCB algorithm had lower cumulative regret and higher winning rates over time.

### Final Winning Rates
- **LinUCB**: *winning_rate*
- **SGD**: *winning_rate*
- **Neural Network**: *winning_rate*

## Conclusion
Among the bandit algorithms compared, LinUCB outperformed both the SGD Online Learning and Neural Network Online Learning algorithms. This suggests that LinUCB is a robust choice for contextual bandit problems, providing a good balance between exploration and exploitation.

## How to Run
1. **Hyperparameter Tuning for SGD Online Learning**:
    ```bash
    python sgd_batch_performance.py
    ```

2. **Comparison of All Algorithms**:
    ```bash
    python comparison.py
    ```

## Dependencies
- numpy
- pandas
- scikit-learn
- torch
- optuna
- matplotlib

## Acknowledgements
We would like to thank the providers of the Spotify dataset for making this comparison possible.
