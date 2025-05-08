import numpy as np
import collections
import time
import gymnasium as gym

class ModelBased:
    def __init__(self, env, gamma=0.99, num_trajectories=10, t_max=100, threshold=0.01):
        self.env = env
        self.state = None
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.num_trajectories = num_trajectories
        self.t_max = t_max
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset the agent state and initialize the environment"""
        self.state, _ = self.env.reset()
    
    def play_n_random_steps(self, count):
        """Collect transition data by randomly exploring the environment"""
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            
            if is_done:
                self.reset()
            else:
                self.state = new_state
    
    def calc_action_value(self, state, action):
        """Calculate the expected value of taking an action in a state"""
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        
        if total == 0:
            return 0.0
        
        action_value = 0.0
        for next_state, count in target_counts.items():
            prob = count / total  # Estimated transition probability
            reward = self.rewards[(state, action, next_state)]
            action_value += prob * (reward + self.gamma * self.V[next_state])
            
        return action_value
    
    def select_action(self, state):
        """Select the best action for a state based on current estimates"""
        best_action, best_value = None, None
        
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
                
        return best_action
    
    def value_iteration(self):
        """Perform one step of value iteration using the estimated model"""
        # First collect more data about the environment
        self.play_n_random_steps(self.num_trajectories)
        
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            
            new_V = max(state_values) if state_values else 0
            
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
                
            self.V[state] = new_V
            
        return self.V, max_diff
    
    def policy(self):
        """Extract the current greedy policy from the value function"""
        policy = np.zeros(self.env.observation_space.n, dtype=int)
        
        for s in range(self.env.observation_space.n):
            Q_values = [self.calc_action_value(s, a) for a in range(self.env.action_space.n)]
            policy[s] = np.argmax(np.array(Q_values)) if Q_values else 0
            
        return policy
    
    def train(self, num_episodes=1000):
        """Train the agent using model-based value iteration"""
        rewards = []
        max_diffs = []
        best_reward = -np.inf
        iteration = 0
        
        while iteration < num_episodes:
            _, max_diff = self.value_iteration()
            max_diffs.append(max_diff)
            
            reward_test = self.check_improvements()
            rewards.append(reward_test)
            
            iteration += 1
            
            if iteration % 50 == 0:
                print(f"Iteration {iteration}, reward: {reward_test:.2f}, max diff: {max_diff:.6f}")
            
            if reward_test > best_reward:
                best_reward = reward_test
            
            if max_diff < self.threshold:
                print(f"Converged after {iteration} iterations with max diff {max_diff:.6f}")
                break
                
        return rewards
    
    def check_improvements(self):
        """Evaluate the current policy by running test episodes"""
        reward_test = 0.0
        num_test_episodes = min(10, self.t_max)  # Limit the number of test episodes
        
        for _ in range(num_test_episodes):
            total_reward = 0.0
            state, _ = self.env.reset()
            
            for _ in range(self.t_max):
                action = self.select_action(state)
                new_state, reward, is_done, truncated, _ = self.env.step(action)
                total_reward += reward
                
                if is_done:
                    break
                    
                state = new_state
                
            reward_test += total_reward
            
        return reward_test / num_test_episodes
    
    def test(self, num_episodes=20):
        rewards = []
        
        for _ in range(num_episodes):
            total_reward = 0.0
            state, _ = self.env.reset()
            
            for _ in range(self.t_max):
                action = self.select_action(state)
                new_state, reward, is_done, truncated, _ = self.env.step(action)
                total_reward += reward
                
                if is_done:
                    break
                    
                state = new_state
                
            rewards.append(total_reward)
            
        return rewards