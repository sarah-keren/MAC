import numpy as np
import random

def _decode_state(state):
    return np.reshape(np.vstack(state), (1, -1))


def _decode_action(action, num_actions, num_agents):
    out = []
    for ind in range(num_agents):
        out.append(action % num_actions)
        action = action // num_actions
    return list(reversed(out))


def get_random_action(environment, observation):
    return np.random.choice(environment.get_available_actions_dictionary()[0])

def get_random_joint_action(environment, observation):
    return np.random.choice(environment.get_available_actions_dictionary()[0])


