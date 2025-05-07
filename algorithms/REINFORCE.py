import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
""" CliffWalking-v1"""

class ReinforceAgent:
    def __init__(self, env, gamma, learning_rate, lr_decay=1, seed=0, t_max =100):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.T_MAX = t_max  # Número máximo de pasos por episodio
        # Objeto que representa la política (J(theta)) como una matriz estados X acciones,
        # con una probabilidad inicial para cada par estado accion igual a: pi(a|s) = 1/|A|
        self.policy_table = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        np.random.seed(seed)

    def select_action(self, state, training=True):
        action_probabilities = self.policy_table[state]
        if training:
            # Escogemos la acción según el vector de policy_table correspondiente a la acción,
            # con una distribución de probabilidad igual a los valores actuales de este vector
            return np.random.choice(np.arange(self.env.action_space.n), p=action_probabilities)
        else:
            return np.argmax(action_probabilities)

    def update_policy(self, episode):
        states, actions, rewards = episode
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
            
        loss = -np.sum(np.log(self.policy_table[states, actions]) * discounted_rewards) / len(states)
        policy_logits = np.log(self.policy_table)
        for t in range(len(states)):
            G_t = discounted_rewards[t]
            action_probs = np.exp(policy_logits[states[t]])
            action_probs /= np.sum(action_probs)
            policy_gradient = G_t * (1 - action_probs[actions[t]])
            policy_logits[states[t], actions[t]] += self.learning_rate * policy_gradient
            # Alternativa:
            #policy_gradient = 1.0 / action_probs[actions[t]]
            #policy_logits[states[t], actions[t]] += self.learning_rate * G_t * policy_gradient
            
        exp_logits = np.exp(policy_logits)
        self.policy_table = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return loss

    def learn_from_episode(self):
        state, _ = self.env.reset()
        episode = []
        done = False
        step = 0
        total_reward = 0

        while not done and step < self.T_MAX:
            action = self.select_action(state)
            next_state, reward, done, terminated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward = total_reward + reward
            step += 1
            if done:
                break
        loss = self.update_policy(zip(*episode))
        self.learning_rate = self.learning_rate * self.lr_decay
        return total_reward, loss

    def policy(self):
        policy = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy
    
    def train(self, num_episodes:int=1000): 
        rewards = []
        for i in range(num_episodes):
            reward, loss = self.learn_from_episode()
            rewards.append(reward)
            print(f"Episode {i+1}/{num_episodes}, Reward: {reward}, Loss: {loss}, Learning Rate: {self.learning_rate}")
        return rewards
    
    def test(self, num_episodes:int=1000):
        rewards = []
        for i in range(num_episodes):
            total_reward = 0.0
            state, _ = self.env.reset()
            for i in range(self.T_MAX):
                action = self.select_action(state, training=False)
                new_state, new_reward, is_done, truncated, _ = self.env.step(action)
                total_reward += new_reward
                if is_done: 
                    break
                state = new_state
            rewards.append(total_reward)
        return rewards