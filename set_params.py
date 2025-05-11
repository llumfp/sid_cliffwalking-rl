EPISODES = [500, 4000, 10000]
GAMMA = [0.99, 0.75, 0.5, 0.25]
EPSILON = [0.1, 0.3, 0.5, 0.8]
LR = [0.1, 0.01, 0.001]
LR_DECAY = ["none", "hyperbolic", "exponential"]
REWARD_SIGNAL = ["default", "custom", "final_100"]
EPSILON_DECAY = ["none", "hyperbolic", "exponential"]
ALGORITHMS = ["actor_critic","value_iteration", "model_based", "REINFORCE", "actor_critic", "q_learning",]
METRICS = ["reward_train", "reward_test", "optimality", "time"]
CRITERIA_BEST_RESPONSE = ["reward_test", "optimality"]


parametres_Q_learning = [
        {
            "alg": "q_learning",
            "episodes": str(episodes),
            "gamma": str(gamma),
            "epsilon": str(epsilon),
            "epsilon_decay": epsilon_decay,
            "reward_signal": reward_signal,
            "lr": str(lr),
            "lr_decay": lr_decay
        }
        for episodes in EPISODES
        for gamma in GAMMA
        for epsilon in EPSILON
        for epsilon_decay in EPSILON_DECAY
        for reward_signal in REWARD_SIGNAL
        for lr in LR
        for lr_decay in LR_DECAY
    ]


parametres_value_iteration = [
        {
            "alg": "value_iteration",
            "episodes": str(episodes),
            "gamma": str(gamma),
            "epsilon": "0",
            "epsilon_decay": "none",
            "reward_signal": reward_signal,
            "lr": "0",
            "lr_decay": "0"
        }
        for episodes in EPISODES
        for gamma in GAMMA
        for reward_signal in REWARD_SIGNAL
    ]

parametres_model_based = [
        {
            "alg": "model_based",
            "episodes": str(episodes),
            "gamma": str(gamma),
            "epsilon": "0",
            "epsilon_decay": "none",
            "reward_signal": reward_signal,
            "lr": "0",
            "lr_decay": "0"
        }
        for episodes in EPISODES
        for gamma in GAMMA
        for reward_signal in REWARD_SIGNAL
    ]


parametres_reinforce = [
        {
            "alg": "REINFORCE",
            "episodes": str(episodes),
            "gamma": str(gamma),
            "epsilon": "0",
            "epsilon_decay": "none",
            "reward_signal": reward_signal,
            "lr": str(lr),
            "lr_decay": str(lr_decay)
        }
        for episodes in EPISODES
        for gamma in GAMMA
        for reward_signal in REWARD_SIGNAL
        for lr in LR
        for lr_decay in LR_DECAY
    ]

parametres_actor_critic = [
        {
            "alg": "actor_critic",
            "episodes": str(episodes),
            "gamma": str(gamma),
            "epsilon": "0",
            "epsilon_decay": "none",
            "reward_signal": reward_signal,
            "lr": str(lr),
            "lr_decay": "0"

        }
        for episodes in EPISODES
        for gamma in GAMMA
        for reward_signal in REWARD_SIGNAL
        for lr in LR
    ]


PARAMETRES = parametres_Q_learning #parametres_value_iteration + parametres_actor_critic + parametres_reinforce  + parametres_model_based # + parametres_Q_learning

OPTIMAL_SOLUTION = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
]
