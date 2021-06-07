import argparse
import sys
import numpy as np
sys.path.append('..')
from MAC.control import controller_decentralized
from MAC.agents import agent, policy_gradient

class RandomAgent:
    """action_space can be either an action space (Discrete, Box etc)
       or a dictionary (key per agent) of action spaces
    """
    def __init__(self, action_space):
        self.space = action_space

    def get_action(self, observation):
        if type(self.space) == dict:
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()

def main():
    args = parse_args()
    env = set_env(args.env)
    decentralized_test(env)

def decentralized_test(env):
    print("Running decentralized random test:")

    agents = {agent_name: RandomAgent(env.action_spaces[agent_name])
              for agent_name in env.agents}
    
    control = controller_decentralized.Decentralized(agents, env)
    control.run(render=True, max_iteration=100)

def set_env(environment_name):
    print('Initializing environment...')

    if environment_name == 'taxi':
        sys.path.append('../environments/MultiTaxiEnv')
        from MAC.environments.MultiTaxiEnv.taxi_environment import TaxiEnv
        env = TaxiEnv(2)
        # Make sure it works with our API:
        env.agents = env.taxis_names
        env.action_spaces = {
            agent_name: env.action_space for agent_name in env.agents
        }
        env.observation_spaces = {
            agent_name: env.observation_space for agent_name in env.agents
        }
        env.possible_agents = [agent for agent in env.agents]

    elif environment_name == 'cleanup':
        sys.path.append('environments/cleanup')
        from MAC.environments.cleanup.social_dilemmas.envs.cleanup import CleanupEnv
        env = CleanupEnv(num_agents=5, render=True)
        env.action_spaces = {
            agent_name: env.action_space for agent_name in env.agents
        }
        env.observation_spaces = {
            agent_name: env.observation_space for agent_name in env.agents
        }
        env.possible_agents = [agent for agent in env.agents.keys()]

    elif environment_name == 'corners':
        sys.path.append('environments/corners')
        from corners_env import CornersEnv
        env = CornersEnv()

    # Petting Zoo:
    elif environment_name == 'particle':
        from pettingzoo.mpe import simple_spread_v2
        env = simple_spread_v2.parallel_env(max_cycles=np.inf)

    elif environment_name == 'piston':
        from pettingzoo.butterfly import pistonball_v4
        env = pistonball_v4.parallel_env(continuous=False)

    env.reset() # We have to call reset() for the env to have agents list
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--env',
        required=True,
        choices=['taxi', 'particle', 'cleanup', 'piston', 'corners'],
        help='Environment to run test on.'
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
