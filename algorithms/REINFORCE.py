import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" CliffWalking-v1"""


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        reward = reward if state != 47 else 0
        if reward == -100:
            done = True
        done = done or truncated
        
        return state, reward, done, done, info



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
        meta = 0
        while not done and step < self.T_MAX:
            action = self.select_action(state)
            next_state, reward, done, terminated, _ = self.env.step(action)
            done = done or terminated
            #reward = reward if reward != -100 else -10
            episode.append((state, action, reward))
            state = next_state
            total_reward = total_reward + reward
            step = step + 1
            meta = meta + 1 if next_state == 47 else meta
        loss = self.update_policy(zip(*episode))
        self.learning_rate = self.learning_rate * self.lr_decay #if self.learning_rate > 0.001 else self.learning_rate
        return total_reward, loss, meta

    def policy(self):
        policy = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy, self.policy_table
    
def draw_history(history, title):
    average_rewards = [np.mean(history[i:i + 100]) for i in range(0, len(history), 100)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(average_rewards) + 1), average_rewards, marker='o', linestyle='-')
    plt.title('Average ' + title + ' Every 50 Episodes')
    plt.xlabel('Episode Group (100 episodes each)')
    plt.ylabel('Average ' + title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def print_policy(policy):
    visual_help = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    actual_policy = np.zeros((4, 12)).tolist()
    for i in range(len(policy)):
        row, col = revert_state_to_row_col(i)
        actual_policy[row][col] = visual_help[policy[i]]
    
    for row in actual_policy:
        print(" | ".join(row))


def revert_state_to_row_col(state):
    row = state // 12
    col = state % 12
    return row,col   


# Declaración de constantes
SLIPPERY = True
TRAINING_EPISODES = 15000
GAMMA = 0.75
T_MAX = 400
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.995

env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=False)
env = RewardWrapper(env)

agent = ReinforceAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE,
                       lr_decay=LEARNING_RATE_DECAY, seed=0, t_max=T_MAX)
rewards = []
losses = []
meta2 = 0
for i in range(TRAINING_EPISODES):
    reward, loss, meta = agent.learn_from_episode()
    meta2 += meta
    policy, policy_table = agent.policy()
    print(f"Last reward: {reward}, last loss: {loss}, new lr: {agent.learning_rate}. End of iteration [{i + 1}/{TRAINING_EPISODES}] - Meta: {1 if meta > 0 else 0}")
    rewards.append(reward)
    losses.append(loss)
    

print_policy(policy)


draw_history(rewards, "Reward")
draw_history(losses, "Loss")


"""best_actions = [0,1,1,1,1,1,1,1,2,1,1,1,2,2, 2]
env = gym.make("CliffWalking-v0", render_mode="human", is_slippery=False)
env = RewardWrapper(env)

state, _ = env.reset()
for i in range(len(best_actions)):
    action = best_actions[i]
    next_state, reward, done, terminated, _ = env.step(action)
    print(f"State: {state}, Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}, Terminated: {terminated}")
    state = next_state
    if done or terminated:
        break"""

env = gym.make("CliffWalking-v0", render_mode="human", is_slippery=True)

state, _ = env.reset()
done = False
step = 0
total_reward = 0
while not done and step < agent.T_MAX:
    env.render()
    action = agent.select_action(state, training=False)
    next_state, reward, done, terminated, _ = env.step(action)
    done = done or terminated
    state = next_state
    total_reward += reward
    step += 1
env.close()
print(f"Total reward after rendering: {total_reward}")

#draw_history(losses, "Loss")