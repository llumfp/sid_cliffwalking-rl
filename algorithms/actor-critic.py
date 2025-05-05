import numpy as np
import gymnasium as gym

class TabularActorCritic:
    def __init__(self, n_states, n_actions, alpha=0.01, beta=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.theta = np.zeros((n_states, n_actions))  # Política (preferencias)
        self.V = np.zeros(n_states)                   # Valor crítico
        self.alpha = alpha  # Tasa de aprendizaje para actor
        self.beta = beta    # Tasa de aprendizaje para crítico
        self.gamma = gamma  # Factor de descuento

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

    def train(self, env, num_episodes=1000, max_steps=100):
        rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0

            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, *info = env.step(action)
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

env = gym.make("CliffWalking-v0", is_slippery=True)


agent = TabularActorCritic(n_states=env.observation_space.n,
                           n_actions=env.action_space.n,
                           alpha=0.01, beta=0.1, gamma=0.99)

import matplotlib.pyplot as plt

# Después de entrenar
rewards = agent.train(env, num_episodes=2000)

# Ventana móvil de 50 episodios
def moving_average(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(moving_average(rewards), label="Recompensa promedio (ventana=50)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("Aprendizaje del agente Actor-Critic en CliffWalking")
plt.grid()
plt.legend()
plt.show()


def render_policy(agent, env):
    actions = ['↑', '→', '↓', '←']
    policy_grid = []

    for s in range(env.observation_space.n):
        if s in range(37, 47):  # El acantilado (cliff)
            policy_grid.append('·')
        elif s == 47:
            policy_grid.append('🏁')  # Meta
        else:
            best_action = np.argmax(agent.get_action_probs(s))
            policy_grid.append(actions[best_action])

    # Mostrar como una cuadrícula 4x12
    for i in range(4):
        row = policy_grid[i * 12:(i + 1) * 12]
        print(' '.join(row))


render_policy(agent, env)
