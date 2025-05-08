EPISODES = [500, 3000]
GAMMA = [0.99, 0.5]
EPSILON = [0.1, 0.3]
LR = [0.1, 0.01]
LR_DECAY = [0.99, 0.5]
REWARD_SIGNAL = ["custom", "custom"]
EPSILON_DECAY = ["none"] # "linear", "exponential"


parametres_Q_learning = [
        {
            "alg": "q_learning",
            "episodes": str(episodes),
            "gamma": str(gamma),
            "epsilon": str(epsilon),
            "epsilon_decay": "none",
            "reward_signal": reward_signal,
            "lr": str(lr),
            "lr_decay": lr_decay
        }
        for episodes in EPISODES
        for gamma in GAMMA
        for epsilon in EPSILON
        #for epsilon_decay in ["none", "linear", "exponential"]
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


PARAMETRES = parametres_value_iteration + parametres_actor_critic + parametres_reinforce + parametres_Q_learning  + parametres_model_based


OPTIMAL_SOLUTION = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
]
