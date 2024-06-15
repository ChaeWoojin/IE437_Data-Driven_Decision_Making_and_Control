import numpy as np

class EpsilonGreedyAlgorithm:
    def __init__(self, n_arms, epsilon=0.1):
        self.epsilon = epsilon
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms)
    
    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.Q))
        else:
            return np.argmax(self.Q)
    
    def initial_learning_phase(self, contexts, rewards, initial_phase_size):
        for i in range(initial_phase_size):
            context = contexts[i].reshape(1, -1)
            chosen_arm = self.select_arm()
            actual_reward = rewards[i]
            reward = 1 if chosen_arm == actual_reward else 0
            self.update(chosen_arm, reward)
    
    def update(self, chosen_arm, reward):
        self.N[chosen_arm] += 1
        n = self.N[chosen_arm]
        value = self.Q[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.Q[chosen_arm] = new_value
    
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
            self.update(chosen_arm, reward)
            regret = optimal_reward - reward
            total_regret += regret
            cumulative_regret.append(total_regret)
            if reward == 1:
                correct_predictions += 1
        
        winning_rate = correct_predictions / len(contexts) * 100
        return cumulative_regret, winning_rate
