import argparse
import sys
import numpy as np
sys.path.append('../..')
from control import controller_decentralized
from agents import agent, policy_gradient


def main():
    args = parse_args()
    env = set_env(args.env)
    #centralized_pg_test(env)
    decentralized_test(env)

def decentralized_test(env):
    """Runs a decentralized test with the Policy Gradient Algorithm."""
    print("Running decentralized test:")
    num_actions_per_agent = env.action_spaces[env.possible_agents[0]].n

    if len(env.observation_spaces[env.possible_agents[0]].shape) == 2:
        obs_size_per_agent = env.observation_spaces[env.possible_agents[0]].shape[1]
    else:
        obs_size_per_agent = env.observation_spaces[env.possible_agents[0]].shape[0]

    agents = {agent_name: agent.Agent(policy_gradient.PolicyGradient(num_actions_per_agent,
              np.random.rand(obs_size_per_agent, num_actions_per_agent),mapping_fn=lambda x: x.flatten()))
              for agent_name in env.possible_agents}
    control = controller_decentralized.Decentralized(agents, env)
    control.train(25, 100)
    control.run(render=True, max_iteration=100)


def set_env(environment_name):
    print('Initializing environment...')

    if environment_name == 'taxi':
        sys.path.append('../environments/MultiTaxiEnv')
        from environments.MultiTaxiEnv.taxi_environment import TaxiEnv
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
        sys.path.append('../environments/cleanup')
        from environments.cleanup.social_dilemmas.envs.cleanup import CleanupEnv
        env = CleanupEnv(num_agents=5, render=True)
        env.action_spaces = {
            agent_name: env.action_space for agent_name in env.agents
        }
        env.observation_spaces = {
            agent_name: env.observation_space for agent_name in env.agents
        }
        env.possible_agents = [agent for agent in env.agents.keys()]

    # Petting Zoo:
    elif environment_name == 'particle':
        from pettingzoo.mpe import simple_spread_v2
        env = simple_spread_v2.parallel_env(max_cycles=np.inf)

    elif environment_name == 'piston':
        from pettingzoo.butterfly import pistonball_v4
        env = pistonball_v4.parallel_env(continuous=False)

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
