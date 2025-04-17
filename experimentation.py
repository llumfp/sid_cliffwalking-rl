import argparse

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
