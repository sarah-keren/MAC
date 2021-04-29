import sys; sys.path.append('..')
sys.path.append('../environments/multiagent-particle-envs')
import importlib

from control.controls import CentralizedControl, DecentralizedControl
from utils.random_agent import RandomAgent

import make_env
from pprint import pprint
import numpy as np

def main():
    centralized_random_test()

def centralized_random_test():
    print(f"Running centralized random test:\n")

    env = make_env.make_env('simple_spread')
    env.discrete_action_input = True
    observation = env.reset()
    env.render()

    agent = RandomAgent(env.action_space)
    controller = CentralizedControl(env, agent)
    controller.run()

if __name__ == "__main__":
    main()