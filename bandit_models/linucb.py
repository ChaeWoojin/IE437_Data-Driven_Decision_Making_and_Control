import numpy as np
from contextualbandits.online import LinUCB

class LinUCBAlgorithm:
    def __init__(self, n_arms, alpha=1.0):
        self.linucb = LinUCB(nchoices=n_arms, alpha=alpha)
    
    def initial_learning_phase(self, contexts, rewards, initial_phase_size):
        for i in range(initial_phase_size):
            context = contexts[i].reshape(1, -1)
            chosen_arm = self.linucb.predict(context)
            actual_reward = rewards[i]
            reward = 1 if chosen_arm == actual_reward else 0
            self.linucb.partial_fit(context, np.array([chosen_arm]), np.array([reward]))
    
    def evaluate(self, contexts, rewards):
        cumulative_regret = []
        optimal_reward = max(rewards)
        total_regret = 0
        correct_predictions = 0
        
        for i in range(len(contexts)):
            context = contexts[i].reshape(1, -1)
            chosen_arm = self.linucb.predict(context)
            actual_reward = rewards[i]
            reward = 1 if chosen_arm == actual_reward else 0
            self.linucb.partial_fit(context, np.array([chosen_arm]), np.array([reward]))
            regret = optimal_reward - reward
            total_regret += regret
            cumulative_regret.append(total_regret)
            if reward == 1:
                correct_predictions += 1
        
        winning_rate = correct_predictions / len(contexts) * 100
        return cumulative_regret, winning_rate
