from agent import Agent

import numpy as np
import random

def default_policy(agent: Agent) -> str:
    actions = ["left", "right", "none"]
    # the exploration rate
    aleatoire = 0.25

    # if no reward has been found, increase the exploration rate
    if np.sum(agent.known_rewards) == 0:
        aleatoire *= 1.5

    # random action
    if random.random() < aleatoire:
        return random.choice(actions)

    # change position according to known rewards to maximise rewards
    else:
        if agent.position > 0 and agent.known_rewards[agent.position - 1] > 0:
            return "left"
        elif agent.position < len(agent.known_rewards) - 1 and agent.known_rewards[agent.position + 1] > 0:
            return "right"
        else:
            return "none"