import argparse
import sys

import numpy as np

sys.path.append('..')
from control.controls import CentralizedControl, DecentralizedControl
from mac_utils.random_agent import RandomAgent

def main():
    args = parse_args()
    env = set_env(args.env)
    centralized_random_test(env)
    decentralized_random_test(env)

def centralized_random_test(env):
    print("Running centralized control test:")
    agent = RandomAgent(env.action_spaces)
    observation = env.reset()
    controller = CentralizedControl(env, agent)
    controller.run(100)

def decentralized_random_test(env):
    print("Running decentralized control test:")
    agents = {agent: RandomAgent(env.action_spaces[agent]) for agent in env.agents}
    observation = env.reset()
    controller = DecentralizedControl(env, agents)
    controller.run(100)

def set_env(environment_name):
    print('Initializing environment...')
    
    if environment_name == 'taxi':
        sys.path.append('../environments/MultiTaxiEnv')
        from taxi_environment import TaxiEnv
        env = TaxiEnv(2)
        # Make sure it works with our API:
        env.agents = env.taxis_names
        env.action_spaces = {
            agent_name: env.action_space for agent_name in env.agents
        }
    
    elif environment_name == 'cleanup':
        sys.path.append('../environments/cleanup')
        from social_dilemmas.envs.cleanup import CleanupEnv
        env = CleanupEnv(num_agents=5, render=True)
        env.action_spaces = {
            agent_name: env.action_space for agent_name in env.agents
        }

    # Petting Zoo:
    elif environment_name == 'particle':
        from pettingzoo.mpe import simple_spread_v2
        env = simple_spread_v2.parallel_env()

    elif environment_name == 'piston':
        from pettingzoo.butterfly import pistonball_v4
        env = pistonball_v4.parallel_env()

    return env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--env',
        required=True,
        choices=['taxi', 'particle', 'cleanup', 'piston'],
        help='Environment to run test on.'
        )
    return parser.parse_args()

if __name__ == "__main__":
    main()