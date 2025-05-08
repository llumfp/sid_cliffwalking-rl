import argparse
from execution import gym
import numpy as np
import csv, os
import time
from algorithms.utils import draw_rewards, revert_state_to_row_col, print_policy

############################################################
############### 1. Parameters configuration ################
############################################################

parser = argparse.ArgumentParser(description="Execution in the Cliff Walking environment.")
parser.add_argument("--exp_id", type=float, help="ID from current experimentation parameters", default=0)
parser.add_argument("--alg", type=str, choices=['q_learning','model_based', 'value_iteration', 'REINFORCE', 'actor_critic'], 
                    help="Algorithm to be used in the training: 'q_learning', 'model_based', 'value_iteration', 'REINFORCE' or 'actor_critic'", default='q_learning')
parser.add_argument("--episodes", type=str, help="Number of episodes to train", default="1000")
parser.add_argument("--gamma", type=float, help="Discount factor for training", default=0.99)
parser.add_argument("--rew", type=str, help="Reward signal to use", default="default")
parser.add_argument("--epsilon", type=float, help="Exploration coefficient in Q-Learning", default=0.1)
parser.add_argument("--epsilon_decay", type=str, choices=['none', 'linear', 'exponential'], 
                    help="Type of exploration decay in Q-Learning: 'none', 'linear', or 'exponential'", default='none')
parser.add_argument("--lr", type=float, help="Learning rate in Q-Learning", default=0.1)
parser.add_argument("--lr_decay", type=str, #choices=['none', 'linear', 'exponential'],
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

############################################################
############## 2. Output file initialization ###############
############################################################

# Create the output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"results/{args.alg}", exist_ok=True)
os.makedirs(f"results/{args.alg}/train", exist_ok=True)
os.makedirs(f"results/{args.alg}/test", exist_ok=True)


# Create a CSV file to store the results
results_file = os.path.join(output_dir, f"results_{args.alg}.csv")
if not os.path.exists(results_file):
    with open(results_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id","alg", "episodes", "gamma", "reward_signal", "epsilon", "epsilon_decay",
            "lr", "lr_decay", "reward_train", "reward_test", "time", "optimality"
        ])

############################################################
################ 3. RL Agent initialization ################
############################################################

"""
Agent needs to have:
- train(num_episodes) method returning rewards
- test(num_test_episodes) method returning rewards

"""
T_MAX = 100



if args.alg == "q_learning":
    # Por el momento no aplicamos el decay del lr ni del epsilon
    from algorithms.q_learning import Qlearning

    # Environment configuration
    env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
    if args.rew == "final_100":
        from algorithms.utils import RewardWrapperFinal100
        env = RewardWrapperFinal100(env)
    elif args.rew == "custom":
        from algorithms.utils import CustomWrapper
        env = CustomWrapper(env)
    
    # Agent initialization
    agent = Qlearning(env,
                      gamma=args.gamma,
                      learning_rate=args.lr,
                      epsilon=args.epsilon,
                      t_max=T_MAX,
                      epsilon_decay=args.epsilon_decay,
                      lr_decay=args.lr_decay)

elif args.alg == "value_iteration":
    from algorithms.value_iteration import ValueIteration
    env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
    
    if args.rew == "custom":
        from algorithms.utils import CustomWrapper
        env = CustomWrapper(env)
        
    agent = ValueIteration(env, 
                           gamma=args.gamma, 
                           t_max=T_MAX)

elif args.alg == "REINFORCE":
    from algorithms.REINFORCE import ReinforceAgent
    env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
    
    if args.rew == "custom":
        from algorithms.utils import CustomWrapper
        env = CustomWrapper(env)
        
    agent = ReinforceAgent(env, 
                           gamma=args.gamma, 
                           learning_rate=args.lr,
                           lr_decay=float(args.lr_decay), 
                           seed=0, 
                           t_max=T_MAX)

elif args.alg == "actor_critic":
    from algorithms.actor_critic import TabularActorCritic
    env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
    
    if args.rew == "custom":
        from algorithms.utils import CustomWrapper
        env = CustomWrapper(env)
        
    agent = TabularActorCritic(env=env,
                               alpha=args.lr, 
                               beta=args.lr * 0.1, 
                               gamma=args.gamma,
                               t_max=T_MAX) 
elif args.alg == "model_based":
    from algorithms.model_based import ModelBased
    env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
    
    if args.rew == "custom":
        from algorithms.utils import CustomWrapper
        env = CustomWrapper(env)
    agent = ModelBased(env,
                      gamma=args.gamma,
                      num_trajectories=100,
                      t_max=T_MAX,
                      threshold=0.01)

############################################################
################### 4. RL Agent training ###################
############################################################

NUM_TEST_EPISODES = 20
# Taining with the suitable agent
start_time = time.time()
# Training
rewards_train = agent.train(int(args.episodes))
#draw_rewards(rewards_train, show=False, path=f'results/{args.alg}/train/rewards_{args.exp_id}.png')

from set_params import OPTIMAL_SOLUTION
OPTIMAL_SOLUTION = np.array(OPTIMAL_SOLUTION)

policy = agent.policy()
optimality = np.sum(OPTIMAL_SOLUTION != policy.reshape(4, 12))
print_policy(policy)

elapsed = time.time() - start_time
avg_reward_train = sum(rewards_train) / len(rewards_train)

############################################################
################### 4. RL Agent testing ####################
############################################################

rewards_test = agent.test(NUM_TEST_EPISODES)
avg_reward_test = sum(rewards_test) / len(rewards_test)

#draw_rewards(rewards_test, show=False, path=f"results/{args.alg}/test/rewards_{args.exp_id}.png")

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
        round(avg_reward_train, 4),
        round(avg_reward_test, 4),
        round(elapsed, 4),
        optimality
    ])
