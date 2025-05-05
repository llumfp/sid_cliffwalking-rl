import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
import gc
from utils import CustomWrapper
gc.enable()

# Configuración del entorno
env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
env = CustomWrapper(env)


class REINFORCEAgent:
    def __init__(self, env, gamma=0.75, learning_rate=0.1, seed=0, t_max=50, n_episodes=20000):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.number_of_episodes = n_episodes
        self.n_rows = 4  # Número de filas del entorno
        self.n_cols = 12  # Número de columnas del entorno
        self.T_MAX = t_max  # Número máximo de pasos por episodio
        # Objeto que representa la política (J(theta)) como una matriz estados X acciones,
        # con una probabilidad inicial para cada par estado accion igual a: pi(a|s) = 1/|A|
        self.theta = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        self.V = np.zeros(self.env.observation_space.n)  # Tabla de valores por estado (baseline)
        np.random.seed(seed)
        
        # Para méticas de evaluación
        self.counting = np.zeros((4, 12))  # Contador de visitas por estado-acción
        self.goal_state = (3, 11)  # Estado objetivo (fila, columna)


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_action_probs(self, state, theta):
        return self.softmax(theta[state])


    def sample_action(self, state, theta):
        probs = self.get_action_probs(state, theta)
        return np.random.choice(len(probs), p=probs)

    def generate_episode(self):
        episode = []
        state, _ = self.env.reset()
        done = False
        i = 0
        while not done and i < 50:
            action = self.sample_action(state, self.theta)
            next_state, reward, done, truncation, _ = self.env.step(action)
            if next_state == 36 and reward == -100 and state != 36:
                state += 12
                cords = self.state_to_coords(state, self.n_cols)
                self.counting[cords] += 1  # Contar la visita a (s,a)
                episode.append((state, action, reward))
                state -= 12
                state = next_state

                
            else:
                cords = self.state_to_coords(next_state, self.n_cols)
                self.counting[cords] += 1  # Contar la visita a (s,a)
                """dist = self.manhattan_distance(cords, self.goal_state)
                reward += -dist  # Recompensa negativa por distancia al objetivo"""
            

                episode.append((state, action, reward))
                state = next_state
            i += 1
            
        return episode

    def compute_returns(self, episode):
        G = 0
        returns = []
        for _, _, reward in reversed(episode):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def reinforce_update(self, episode, returns):
        for t, (state, action, _) in enumerate(episode):
            probs = self.get_action_probs(state, self.theta)
            grad_log = -probs
            grad_log[action] += 1  # ∇θ log π(a|s)
            self.theta[state] += self.learning_rate * returns[t] * grad_log
        return self.theta

    def reinforce_with_baseline_update(self, episode, returns):
        for t, (state, action, _) in enumerate(episode):
            Gt = returns[t]
            baseline = self.V[state]
            advantage = Gt - baseline

            # Gradiente de log π(a|s)
            probs = self.get_action_probs(state, self.theta)
            grad_log = -probs
            grad_log[action] += 1

            # Actualización de la política
            self.theta[state] += self.learning_rate * advantage * grad_log

            # Actualización de la baseline (V[s] hacia Gt)
            self.V[state] += self.learning_rate * (Gt - self.V[state])
            
        self.learning_rate *= 0.99  # Decaimiento de la tasa de aprendizaje
        self.learning_rate = max(self.learning_rate, 0.0001)
        
        return self.theta, self.V
    
    
    def train(self):
        rewards_per_episode = []
        length_per_episode = []

        for episode_num in range(self.number_of_episodes):
            """episode = self.generate_episode()
            length_per_episode.append(len(episode))

            returns = self.compute_returns(episode)
            total_reward = sum([r for (_, _, r) in episode])
            rewards_per_episode.append(total_reward)
            theta = self.reinforce_update(episode, returns)"""

            episode = self.generate_episode()
            length_per_episode.append(len(episode))
            returns = self.compute_returns(episode)
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

            total_reward = sum([r for (_, _, r) in episode])
            rewards_per_episode.append(total_reward)
            theta, V = self.reinforce_with_baseline_update(episode, returns)
            

            if (episode_num + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"Episode {episode_num + 1}: Avg reward (last 100) = {avg_reward:.2f}")

        self.print_policy_map(map_shape=(4, 12))


        # Plot the log-transformed heatmap
        sns.heatmap(self.counting, annot=True, cmap="Blues", fmt=".2f")
        plt.title("Log-transformed Visit Count Heatmap")

        # Ventana móvil de 50 episodios
        def moving_average(x, w=50):
            return np.convolve(x, np.ones(w)/w, mode='valid')

        plt.figure(figsize=(10, 5))
        plt.plot(moving_average(rewards_per_episode), label="Recompensa promedio (ventana=50)")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.title("Aprendizaje del agente Actor-Critic en CliffWalking")
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(length_per_episode), label="Longitud promedio (ventana=50)")
        plt.xlabel("Episodio")
        plt.ylabel("Longitud del episodio")
        plt.title("Longitud del episodio")
        plt.grid()

        plt.show()


    def plot_policy_heatmap(self):
        policy_probs = np.apply_along_axis(self.softmax, 1, self.theta)
        plt.figure(figsize=(8, 6))
        sns.heatmap(policy_probs, annot=True, cmap="Blues", fmt=".2f",
                    xticklabels=[f"A{i}" for i in range(self.n_actions)],
                    yticklabels=[f"S{i}" for i in range(self.n_states)])
        plt.title("Política: Probabilidades por estado-acción")
        plt.xlabel("Acción")
        plt.ylabel("Estado")
        plt.show()
        
    def print_policy_map(self, map_shape=(4, 4)):
        visual_help = {0: '^', 1: '>', 2: 'v', 3: '<'}
        policy = np.argmax(np.apply_along_axis(self.softmax, 1, self.theta), axis=1)
        print("Política (acción más probable por estado):")
        for i in range(map_shape[0]):
            row = ""
            for j in range(map_shape[1]):
                state = i * map_shape[1] + j
                action = policy[state]
                row += visual_help[action] + " "
            print(row)
        
        
    def state_to_coords(self, state, n_cols):
        return divmod(state, n_cols)  # (fila, columna)
    
    def manhattan_distance(self, state1, state2):
        return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

agent = REINFORCEAgent(env, gamma=0.5, learning_rate=0.01, seed=0, t_max=30, n_episodes=20000)
agent.train()