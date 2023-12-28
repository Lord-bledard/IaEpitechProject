from agent import Agent


import random

def default_policy(agent: Agent) -> str:
    actions = ["left", "right", "none"]
    epsilon = 0.25  # Probabilité d'exploration

    # Exploration: choisir une action au hasard
    if random.random() < epsilon:
        return random.choice(actions)

    # Exploitation: choisir une action basée sur les récompenses connues
    else:
        if agent.position > 0 and agent.known_rewards[agent.position - 1] > 0:
            return "left"
        elif agent.position < len(agent.known_rewards) - 1 and agent.known_rewards[agent.position + 1] > 0:
            return "right"
        else:
            return "none"