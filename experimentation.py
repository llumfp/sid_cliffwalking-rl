import argparse
from execution import gym

parser = argparse.ArgumentParser(description="Execution in the Cliff Walking environment.")
parser.add_argument("--alg", type=str, choices=['q_learning','model_based', 'value_iteration'], 
                    help="Algorithm to be used in the training: 'q_learning', 'model_based', or 'value_iteration'", default='q_learning')
parser.add_argument("--eps", type=str, help="Number of episodes to train", default="1000")
parser.add_argument("--disc", type=float, help="Discount factor for training", default=0.99)
parser.add_argument("--rew", type=str, help="Reward signal to use", default="default")
parser.add_argument("--exp", type=float, help="Exploration coefficient in Q-Learning", default=0.1)
parser.add_argument("--exp_decay", type=str, choices=['none', 'linear', 'exponential'], 
                    help="Type of exploration decay in Q-Learning: 'none', 'linear', or 'exponential'", default='none')
parser.add_argument("--lr", type=float, help="Learning rate in Q-Learning", default=0.1)
parser.add_argument("--lr_decay", type=str, choices=['none', 'linear', 'exponential'],
                    help="Type of learning rate decay in Q-Learning: 'none', 'linear', or 'exponential'", default='none')

args = parser.parse_args()

print("Arguments received:")
print(f"Algorithm used: {args.alg}")
print(f"Episodes: {args.eps}")
print(f"Discount Factor: {args.disc}")
print(f"Reward Signal: {args.rew}")
print(f"Exploration Coefficient: {args.exp}")
print(f"Exploration Decay: {args.exp_decay}")
print(f"Learning Rate: {args.lr}")
print(f"Learning Rate Decay: {args.lr_decay}")

class CustomFrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        state, reward, is_done, truncated, info = self.env.step(action)
        
        if state in [47]:
            is_done = True
        
        return state, reward, is_done, is_done, info

env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
env = CustomFrozenLakeWrapper(env)


if args.alg == "q_learning":
    pass

elif args.alg == "model_based":
    pass

elif args.alg == "value_iteration":
    from algorithms.value_iteration import ValueIteration

    agent = ValueIteration(env, gamma=args.disc, num_episodes=int(args.eps))
    rewards, max_diffs = agent.train()

elif args.alg == "REINFORCE":
    pass    