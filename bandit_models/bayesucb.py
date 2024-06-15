import numpy as np
from scipy.stats import beta

class BayesUCBAlgorithm:
    def __init__(self, n_arms, alpha=1):
        self.n_arms = n_arms
        self.alpha = alpha
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)
    
    def select_arm(self):
        # Calculate the upper confidence bound for each arm
        upper_bounds = beta.ppf(1 - 1 / (np.arange(1, self.n_arms + 1) * self.alpha), self.successes, self.failures)
        return np.argmax(upper_bounds)
    
    def initial_learning_phase(self, contexts, rewards, initial_phase_size):
        for i in range(initial_phase_size):
            chosen_arm = self.select_arm()
            actual_reward = rewards[i]
            reward = 1 if chosen_arm == actual_reward else 0
            self.update(chosen_arm, reward)
    
    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1
    
    def evaluate(self, contexts, rewards):
        cumulative_regret = []
        optimal_reward = max(rewards)
        total_regret = 0
        correct_predictions = 0
        
        for i in range(len(contexts)):
            chosen_arm = self.select_arm()
            actual_reward = rewards[i]
            reward = 1 if chosen_arm == actual_reward else 0
            self.update(chosen_arm, reward)
            regret = optimal_reward - reward
            total_regret += regret
            cumulative_regret.append(total_regret)
            if reward == 1:
                correct_predictions += 1
        
        winning_rate = correct_predictions / len(contexts) * 100
        return cumulative_regret, winning_rate
