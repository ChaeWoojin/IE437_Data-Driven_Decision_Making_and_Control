import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bandit_models.linucb import LinUCBAlgorithm
from bandit_models.thompson_sampling import ThompsonSamplingAlgorithm
from bandit_models.epsilon_greedy import EpsilonGreedyAlgorithm
from bandit_models.ucb1 import UCB1Algorithm
from bandit_models.softmax import SoftmaxAlgorithm
from bandit_models.bayesucb import BayesUCBAlgorithm

# Load the dataset
file_name = './data/spotify_personal_kaggle.csv'
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

# Initialize algorithms
linucb = LinUCBAlgorithm(n_arms=2, alpha=1.0)
thompson_sampling = ThompsonSamplingAlgorithm(n_arms=2)
epsilon_greedy = EpsilonGreedyAlgorithm(n_arms=2, epsilon=0.5)
ucb1 = UCB1Algorithm(n_arms=2)
softmax = SoftmaxAlgorithm(n_arms=2, tau=1.0)
bayes_ucb = BayesUCBAlgorithm(n_arms=2, alpha=1.0)

# Initial learning phase
linucb.initial_learning_phase(contexts, rewards, initial_phase_size)
thompson_sampling.initial_learning_phase(contexts, rewards, initial_phase_size)
epsilon_greedy.initial_learning_phase(contexts, rewards, initial_phase_size)
ucb1.initial_learning_phase(contexts, rewards, initial_phase_size)
softmax.initial_learning_phase(contexts, rewards, initial_phase_size)
bayes_ucb.initial_learning_phase(contexts, rewards, initial_phase_size)

# Evaluation phase
contexts_eval = contexts[initial_phase_size:]
rewards_eval = rewards[initial_phase_size:]

cumulative_regret_linucb, winning_rate_linucb = linucb.evaluate(contexts_eval, rewards_eval)
cumulative_regret_thompson, winning_rate_thompson = thompson_sampling.evaluate(contexts_eval, rewards_eval)
cumulative_regret_epsilon, winning_rate_epsilon = epsilon_greedy.evaluate(contexts_eval, rewards_eval)
cumulative_regret_ucb1, winning_rate_ucb1 = ucb1.evaluate(contexts_eval, rewards_eval)
cumulative_regret_softmax, winning_rate_softmax = softmax.evaluate(contexts_eval, rewards_eval)
cumulative_regret_bayesucb, winning_rate_bayesucb = bayes_ucb.evaluate(contexts_eval, rewards_eval)

# Plot cumulative regret comparison
plt.figure(figsize=(10, 6))
plt.plot(cumulative_regret_linucb, marker='o', label='LinUCB (α={1.0})')
plt.plot(cumulative_regret_thompson, marker='x', label='Thompson Sampling')
plt.plot(cumulative_regret_epsilon, marker='*', label='Epsilon Greedy (ε={0.2})')
plt.plot(cumulative_regret_ucb1, marker='+', label='UCB1')
plt.plot(cumulative_regret_softmax, marker='s', label='Softmax (τ={0.1})')
plt.plot(cumulative_regret_bayesucb, marker='d', label='BayesUCB (α={1.0})')
plt.title('Cumulative Regret Over Time')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.grid(True)
plt.show()

print(f"LinUCB Winning Rate: {winning_rate_linucb:.2f}%")
print(f"Thompson Sampling Winning Rate: {winning_rate_thompson:.2f}%")
print(f"Epsilon Greedy Winning Rate: {winning_rate_epsilon:.2f}%")
print(f"UCB1 Winning Rate: {winning_rate_ucb1:.2f}%")
print(f"Softmax Winning Rate: {winning_rate_softmax:.2f}%")
print(f"BayesUCB Winning Rate: {winning_rate_bayesucb:.2f}%")

# # Optional: Plot chosen arms and obtained rewards for additional analysis
# def plot_arms_rewards(chosen_arms, obtained_rewards, title):
#     plt.figure(figsize=(10, 6))
#     plt.plot(chosen_arms, marker='o', label='Chosen Arms')
#     plt.plot(obtained_rewards, marker='x', label='Obtained Rewards')
#     plt.title(title)
#     plt.xlabel('Iteration')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Extract chosen arms and rewards from evaluation phase (for additional analysis)
# def extract_arms_rewards(algorithm, contexts_eval, rewards_eval, is_thompson=False, is_linucb=False, is_bayesucb=False):
#     chosen_arms = []
#     obtained_rewards = []
#     if is_thompson:
#         alpha_beta_copy = np.ones((2, 2))
#         for i in range(len(contexts_eval)):
#             context = contexts_eval[i].reshape(1, -1)
#             chosen_arm = algorithm.select_arm()
#             actual_reward = rewards_eval[i]
#             reward = 1 if chosen_arm == actual_reward else 0
#             chosen_arms.append(chosen_arm)
#             obtained_rewards.append(reward)
#             alpha_beta_copy[chosen_arm, 0] += reward
#             alpha_beta_copy[chosen_arm, 1] += 1 - reward
#     elif is_linucb:
#         for i in range(len(contexts_eval)):
#             context = contexts_eval[i].reshape(1, -1)
#             chosen_arm = algorithm.linucb.predict(context)
#             actual_reward = rewards_eval[i]
#             reward = 1 if chosen_arm == actual_reward else 0
#             chosen_arms.append(chosen_arm)
#             obtained_rewards.append(reward)
#             algorithm.linucb.partial_fit(context, np.array([chosen_arm]), np.array([reward]))
#     elif is_bayesucb:
#         successes = np.ones(2)
#         failures = np.ones(2)
#         for i in range(len(contexts_eval)):
#             chosen_arm = algorithm.select_arm()
#             actual_reward = rewards_eval[i]
#             reward = 1 if chosen_arm == actual_reward else 0
#             chosen_arms.append(chosen_arm)
#             obtained_rewards.append(reward)
#             if reward == 1:
#                 successes[chosen_arm] += 1
#             else:
#                 failures[chosen_arm] += 1
#     else:
#         for i in range(len(contexts_eval)):
#             context = contexts_eval[i].reshape(1, -1)
#             chosen_arm = algorithm.select_arm()
#             actual_reward = rewards_eval[i]
#             reward = 1 if chosen_arm == actual_reward else 0
#             chosen_arms.append(chosen_arm)
#             obtained_rewards.append(reward)
#             algorithm.update(chosen_arm, reward)
#     return chosen_arms, obtained_rewards

# chosen_arms_linucb, obtained_rewards_linucb = extract_arms_rewards(linucb, contexts_eval, rewards_eval, is_linucb=True)
# chosen_arms_thompson, obtained_rewards_thompson = extract_arms_rewards(thompson_sampling, contexts_eval, rewards_eval, is_thompson=True)
# chosen_arms_epsilon, obtained_rewards_epsilon = extract_arms_rewards(epsilon_greedy, contexts_eval, rewards_eval)
# chosen_arms_ucb1, obtained_rewards_ucb1 = extract_arms_rewards(ucb1, contexts_eval, rewards_eval)
# chosen_arms_softmax, obtained_rewards_softmax = extract_arms_rewards(softmax, contexts_eval, rewards_eval)
# chosen_arms_bayesucb, obtained_rewards_bayesucb = extract_arms_rewards(bayes_ucb, contexts_eval, rewards_eval, is_bayesucb=True)

# plot_arms_rewards(chosen_arms_linucb, obtained_rewards_linucb, 'LinUCB - Chosen Arms and Obtained Rewards')
# plot_arms_rewards(chosen_arms_thompson, obtained_rewards_thompson, 'Thompson Sampling - Chosen Arms and Obtained Rewards')
# plot_arms_rewards(chosen_arms_epsilon, obtained_rewards_epsilon, 'Epsilon Greedy - Chosen Arms and Obtained Rewards')
# plot_arms_rewards(chosen_arms_ucb1, obtained_rewards_ucb1, 'UCB1 - Chosen Arms and Obtained Rewards')
# plot_arms_rewards(chosen_arms_softmax, obtained_rewards_softmax, 'Softmax - Chosen Arms and Obtained Rewards')
# plot_arms_rewards(chosen_arms_bayesucb, obtained_rewards_bayesucb, 'BayesUCB - Chosen Arms and Obtained Rewards')
