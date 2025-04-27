import argparse
from execution import gym
import csv, os
import time
from algorithms.utils import draw_rewards, revert_state_to_row_col, print_policy

parser = argparse.ArgumentParser(description="Execution in the Cliff Walking environment.")
parser.add_argument("--exp_id", type=float, help="ID from current experimentation parameters", default=0)
parser.add_argument("--alg", type=str, choices=['q_learning','model_based', 'value_iteration'], 
                    help="Algorithm to be used in the training: 'q_learning', 'model_based', or 'value_iteration'", default='q_learning')
parser.add_argument("--episodes", type=str, help="Number of episodes to train", default="1000")
parser.add_argument("--gamma", type=float, help="Discount factor for training", default=0.99)
parser.add_argument("--rew", type=str, help="Reward signal to use", default="default")
parser.add_argument("--epsilon", type=float, help="Exploration coefficient in Q-Learning", default=0.1)
parser.add_argument("--epsilon_decay", type=str, choices=['none', 'linear', 'exponential'], 
                    help="Type of exploration decay in Q-Learning: 'none', 'linear', or 'exponential'", default='none')
parser.add_argument("--lr", type=float, help="Learning rate in Q-Learning", default=0.1)
parser.add_argument("--lr_decay", type=str, choices=['none', 'linear', 'exponential'],
                    help="Type of learning rate decay in Q-Learning: 'none', 'linear', or 'exponential'", default='none')

args = parser.parse_args()

print("Arguments received:")
print(f"Algorithm used: {args.alg}")
print(f"Episodes: {args.episodes}")
print(f"Discount Factor (gamma): {args.gamma}")
print(f"Reward Signal: {args.rew}")
print(f"Exploration Coefficient (epsilon): {args.epsilon}")
print(f"Exploration Decay: {args.epsilon_decay}")
print(f"Learning Rate: {args.lr}")
print(f"Learning Rate Decay: {args.lr_decay}")


# Create the output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Create a CSV file to store the results
results_file = os.path.join(output_dir, f"results_{args.alg}.csv")
if not os.path.exists(results_file):
    with open(results_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id","alg", "episodes", "gamma", "reward_signal", "epsilon", "epsilon_decay",
            "lr", "lr_decay", "reward_train", "time"
        ])

if args.alg == "q_learning":
    # Por el momento no aplicamos el decay del lr ni del epsilon
    from algorithms.q_learning import Qlearning
    T_MAX = 30

    # Environment configuration
    env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
    if args.rew == "final_100":
        from algorithms.utils import RewardWrapperFinal100
        env = RewardWrapperFinal100(env)
    elif args.rew == "custom":
        env = CustomWrapper(env)
    
    # Agent initialization
    agent = Qlearning(env,
                      gamma=args.gamma,
                      learning_rate=args.lr,
                      epsilon=args.epsilon,
                      t_max=T_MAX,
                      epsilon_decay=args.epsilon_decay,
                      lr_decay=args.lr_decay)

elif args.alg == "model_based":
    pass

elif args.alg == "value_iteration":
    from algorithms.value_iteration import ValueIteration
    from algorithms.utils import CustomWrapper
    env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
    env = CustomWrapper(env)
    agent = ValueIteration(env, gamma=args.disc, num_episodes=int(args.eps))
    rewards, max_diffs = agent.train()

elif args.alg == "REINFORCE":
    pass    

# Taining with the suitable agent
start_time = time.time()
# Training
rewards_train = agent.train(int(args.episodes))
draw_rewards(rewards_train, show=False, path=f'results/rewards_{args.alg}_{args.exp_id}.png')

policy = agent.policy()

print_policy(policy)

elapsed = time.time() - start_time
avg_reward = sum(rewards_train) / len(rewards_train)

with open(results_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        args.alg,
        args.exp_id,
        args.episodes,
        args.gamma,
        args.rew,
        args.epsilon,
        args.epsilon_decay,
        args.lr,
        args.lr_decay,
        round(avg_reward, 4),
        round(elapsed, 4)
    ])
