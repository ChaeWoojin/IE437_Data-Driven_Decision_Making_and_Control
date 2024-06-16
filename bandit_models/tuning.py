import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bandit_models import (
    LinUCBAlgorithm,
    ThompsonSamplingAlgorithm,
    EpsilonGreedyAlgorithm,
    UCB1Algorithm,
    SoftmaxAlgorithm,
    BayesUCBAlgorithm
)

# Load the dataset
file_name = '../data/spotify_personal_kaggle.csv'
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

# Initial learning phase size
initial_phase_size = 100

# Evaluation phase
contexts_eval = contexts[initial_phase_size:]
rewards_eval = rewards[initial_phase_size:]

def tune_epsilon_greedy():
    best_epsilon = None
    best_cumulative_regret = float('inf')
    best_winning_rate = 0
    for epsilon in [0.01, 0.05, 0.1, 0.2, 0.5]:
        algo = EpsilonGreedyAlgorithm(n_arms=2, epsilon=epsilon)
        algo.initial_learning_phase(contexts, rewards, initial_phase_size)
        cumulative_regret, winning_rate = algo.evaluate(contexts_eval, rewards_eval)
        if cumulative_regret[-1] < best_cumulative_regret:
            best_cumulative_regret = cumulative_regret[-1]
            best_winning_rate = winning_rate
            best_epsilon = epsilon
    return best_epsilon, best_cumulative_regret, best_winning_rate

def tune_softmax():
    best_tau = None
    best_cumulative_regret = float('inf')
    best_winning_rate = 0
    for tau in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        algo = SoftmaxAlgorithm(n_arms=2, tau=tau)
        algo.initial_learning_phase(contexts, rewards, initial_phase_size)
        cumulative_regret, winning_rate = algo.evaluate(contexts_eval, rewards_eval)
        if cumulative_regret[-1] < best_cumulative_regret:
            best_cumulative_regret = cumulative_regret[-1]
            best_winning_rate = winning_rate
            best_tau = tau
    return best_tau, best_cumulative_regret, best_winning_rate

def tune_bayes_ucb():
    best_alpha = None
    best_cumulative_regret = float('inf')
    best_winning_rate = 0
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        algo = BayesUCBAlgorithm(n_arms=2, alpha=alpha)
        algo.initial_learning_phase(contexts, rewards, initial_phase_size)
        cumulative_regret, winning_rate = algo.evaluate(contexts_eval, rewards_eval)
        if cumulative_regret[-1] < best_cumulative_regret:
            best_cumulative_regret = cumulative_regret[-1]
            best_winning_rate = winning_rate
            best_alpha = alpha
    return best_alpha, best_cumulative_regret, best_winning_rate

def tune_thompson_sampling():
    best_cumulative_regret = float('inf')
    best_winning_rate = 0
    algo = ThompsonSamplingAlgorithm(n_arms=2)
    algo.initial_learning_phase(contexts, rewards, initial_phase_size)
    cumulative_regret, winning_rate = algo.evaluate(contexts_eval, rewards_eval)
    if cumulative_regret[-1] < best_cumulative_regret:
        best_cumulative_regret = cumulative_regret[-1]
        best_winning_rate = winning_rate
    return best_cumulative_regret, best_winning_rate

def tune_ucb1():
    # UCB1 doesn't have a tunable parameter, but we include it for completeness
    algo = UCB1Algorithm(n_arms=2)
    algo.initial_learning_phase(contexts, rewards, initial_phase_size)
    cumulative_regret, winning_rate = algo.evaluate(contexts_eval, rewards_eval)
    return None, cumulative_regret[-1], winning_rate

def tune_linucb():
    best_alpha = None
    best_cumulative_regret = float('inf')
    best_winning_rate = 0
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        algo = LinUCBAlgorithm(n_arms=2, alpha=alpha)
        algo.initial_learning_phase(contexts, rewards, initial_phase_size)
        cumulative_regret, winning_rate = algo.evaluate(contexts_eval, rewards_eval)
        if cumulative_regret[-1] < best_cumulative_regret:
            best_cumulative_regret = cumulative_regret[-1]
            best_winning_rate = winning_rate
            best_alpha = alpha
    return best_alpha, best_cumulative_regret, best_winning_rate


best_epsilon, reg_epsilon, win_rate_epsilon = tune_epsilon_greedy()
best_tau, reg_softmax, win_rate_softmax = tune_softmax()
best_alpha_bayes, reg_bayes, win_rate_bayes = tune_bayes_ucb()
reg_thompson, win_rate_thompson = tune_thompson_sampling()
_, reg_ucb1, win_rate_ucb1 = tune_ucb1()
best_alpha_linucb, reg_linucb, win_rate_linucb = tune_linucb()

print(f"Best Epsilon-Greedy: epsilon={best_epsilon}, Regret={reg_epsilon}, Winning Rate={win_rate_epsilon:.2f}%")
print(f"Best Softmax: tau={best_tau}, Regret={reg_softmax}, Winning Rate={win_rate_softmax:.2f}%")
print(f"Best BayesUCB: alpha={best_alpha_bayes}, Regret={reg_bayes}, Winning Rate={win_rate_bayes:.2f}%")
print(f"Thompson Sampling: Regret={reg_thompson}, Winning Rate={win_rate_thompson:.2f}%")
print(f"UCB1: Regret={reg_ucb1}, Winning Rate={win_rate_ucb1:.2f}%")
print(f"Best LinUCB: alpha={best_alpha_linucb}, Regret={reg_linucb}, Winning Rate={win_rate_linucb:.2f}%")

# Plot cumulative regret comparison
plt.figure(figsize=(10, 6))

# Re-run the best algorithms to get their cumulative regrets
epsilon_greedy_best = EpsilonGreedyAlgorithm(n_arms=2, epsilon=best_epsilon)
epsilon_greedy_best.initial_learning_phase(contexts, rewards, initial_phase_size)
cumulative_regret_epsilon_best, _ = epsilon_greedy_best.evaluate(contexts_eval, rewards_eval)

softmax_best = SoftmaxAlgorithm(n_arms=2, tau=best_tau)
softmax_best.initial_learning_phase(contexts, rewards, initial_phase_size)
cumulative_regret_softmax_best, _ = softmax_best.evaluate(contexts_eval, rewards_eval)

bayes_ucb_best = BayesUCBAlgorithm(n_arms=2, alpha=best_alpha_bayes)
bayes_ucb_best.initial_learning_phase(contexts, rewards, initial_phase_size)
cumulative_regret_bayes_best, _ = bayes_ucb_best.evaluate(contexts_eval, rewards_eval)

thompson_best = ThompsonSamplingAlgorithm(n_arms=2)
thompson_best.initial_learning_phase(contexts, rewards, initial_phase_size)
cumulative_regret_thompson_best, _ = thompson_best.evaluate(contexts_eval, rewards_eval)

ucb1_best = UCB1Algorithm(n_arms=2)
ucb1_best.initial_learning_phase(contexts, rewards, initial_phase_size)
cumulative_regret_ucb1_best, _ = ucb1_best.evaluate(contexts_eval, rewards_eval)

linucb_best = LinUCBAlgorithm(n_arms=2, alpha=best_alpha_linucb)
linucb_best.initial_learning_phase(contexts, rewards, initial_phase_size)
cumulative_regret_linucb_best, _ = linucb_best.evaluate(contexts_eval, rewards_eval)


plt.plot(cumulative_regret_epsilon_best, marker='*', label=f'Epsilon Greedy (ε={best_epsilon})')
plt.plot(cumulative_regret_softmax_best, marker='s', label=f'Softmax (τ={best_tau})')
plt.plot(cumulative_regret_bayes_best, marker='d', label=f'BayesUCB (α={best_alpha_bayes})')
plt.plot(cumulative_regret_thompson_best, marker='x', label='Thompson Sampling')
plt.plot(cumulative_regret_ucb1_best, marker='+', label='UCB1')
plt.plot(cumulative_regret_linucb_best, marker='o', label=f'LinUCB (α={best_alpha_linucb})')
plt.title('Cumulative Regret Over Time')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.grid(True)
plt.show()