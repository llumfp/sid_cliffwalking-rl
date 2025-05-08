import numpy as np
import gymnasium as gym

class TabularActorCritic:
    def __init__(self, env, alpha=0.01, beta=0.1, gamma=0.99, t_max:int=30):
        self.env = env
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.theta = np.zeros((self.n_states, self.n_actions))  # Política (preferencias)
        self.V = np.zeros(self.n_states)                   # Valor crítico
        self.alpha = alpha  # Tasa de aprendizaje para actor
        self.beta = beta    # Tasa de aprendizaje para crítico
        self.gamma = gamma  # Factor de descuento
        self.t_max = t_max  # Máximo número de pasos por episodio

    def get_action_probs(self, state):
        preferences = self.theta[state]
        exp_prefs = np.exp(preferences - np.max(preferences))
        return exp_prefs / np.sum(exp_prefs)

    def select_action(self, state):
        probs = self.get_action_probs(state)
        return np.random.choice(self.n_actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        # TD target y error
        target = reward + (0 if done else self.gamma * self.V[next_state])
        td_error = target - self.V[state]

        # Actualiza el crítico (valor del estado)
        self.V[state] += self.beta * td_error

        # Gradiente de log π(a|s)
        probs = self.get_action_probs(state)
        grad_log = -probs
        grad_log[action] += 1  # ∇ log π(a|s)

        # Actualiza el actor
        self.theta[state] += self.alpha * td_error * grad_log

    def train(self, num_episodes=1000):
        rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            t = 0
            
            while t < self.t_max:
                action = self.select_action(state)
                next_state, reward, done, *info = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            rewards.append(total_reward)
            if (episode + 1) % 100 == 0:
                avg = np.mean(rewards[-100:])
                print(f"Episodio {episode+1}, recompensa promedio (últimos 100): {avg:.2f}")
        return rewards
    
    def test(self, num_episodes=100):
        rewards = []
        for i in range(num_episodes):
            total_reward = 0.0
            state, _ = self.env.reset()
            for i in range(self.t_max):
                action = self.select_action(state)
                new_state, new_reward, is_done, truncated, _ = self.env.step(action)
                total_reward += new_reward
                if is_done: 
                    break
                state = new_state
            rewards.append(total_reward)
        return rewards
    
    
    def policy(self):
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            action_probs = self.get_action_probs(s)
            policy[s] = np.argmax(action_probs)
        return policy

