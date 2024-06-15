import numpy as np
from scipy.stats import beta

class ThompsonSamplingAlgorithm:
    def __init__(self, n_arms):
        self.alpha_beta = np.ones((n_arms, 2))
    
    def select_arm(self):
        samples = [beta.rvs(a, b) for a, b in self.alpha_beta]
        return np.argmax(samples)
    
    def initial_learning_phase(self, contexts, rewards, initial_phase_size):
        for i in range(initial_phase_size):
            context = contexts[i].reshape(1, -1)
            chosen_arm = self.select_arm()
            actual_reward = rewards[i]
            reward = 1 if chosen_arm == actual_reward else 0
            self.alpha_beta[chosen_arm, 0] += reward
            self.alpha_beta[chosen_arm, 1] += 1 - reward
    
    def evaluate(self, contexts, rewards):
        cumulative_regret = []
        optimal_reward = max(rewards)
        total_regret = 0
        correct_predictions = 0
        
        for i in range(len(contexts)):
            context = contexts[i].reshape(1, -1)
            chosen_arm = self.select_arm()
            actual_reward = rewards[i]
            reward = 1 if chosen_arm == actual_reward else 0
            self.alpha_beta[chosen_arm, 0] += reward
            self.alpha_beta[chosen_arm, 1] += 1 - reward
            regret = optimal_reward - reward
            total_regret += regret
            cumulative_regret.append(total_regret)
            if reward == 1:
                correct_predictions += 1
        
        winning_rate = correct_predictions / len(contexts) * 100
        return cumulative_regret, winning_rate
