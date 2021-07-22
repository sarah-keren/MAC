import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../src')
from control.controller_decentralized import Decentralized
from control.controller_decentralized_RL import DecentralizedRL
from control.controller_centralized_RL import CentralizedRL
from agents.deep_policy_gradient import DeepPolicyGradient
from agents.policy_gradient import PolicyGradient
from agents.deep_q import DQN
from agents.agent import Agent
from agents.random_agent import RandomAgent
from environments.env_wrapper import EnvWrappper


def main():
    # controller = make_random_decentralized_control()
    # results = evaluate_controller(controller, 100, RL=False)
    controller = make_rl_decentralized_control('dpg')
    results = evaluate_controller(controller, 100, RL=True)
    sums = sum_all_agents(results)
    plt.style.use('seaborn')
    plt.hist(sums)
    plt.show()

def make_rl_decentralized_control(centralized_policy):
    env = set_env()
    decision_maker = create_centralized_agent(centralized_policy, env)
    env_agents = env.get_env_agents()
    centralized_agents = {agent_name: None  for agent_name in env_agents}
    controller = CentralizedRL(env, centralized_agents, decision_maker)
    return controller

def make_random_decentralized_control():
    env = set_env()
    # Make Random Decentralized
    spaces = env.get_env().action_spaces
    
    agents = {
        agent_name: RandomAgent(spaces[agent_name])
        for agent_name in spaces
    }
    controller = Decentralized(env, agents)
    return controller

def evaluate_controller(controller, num_runs, RL):
    all_results = []
    for _ in range(num_runs):
        if RL:
            controller.run(render=False, max_iteration=10, max_episode_len=25, num_episodes=10, batch_size=0)
        else:
            controller.run(False, 20)
        all_results.append(controller.total_rewards)
    return all_results

def set_env():
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
    needs_conv = False
        
    return EnvWrappper(env, needs_conv=needs_conv)
    
def sum_all_agents(results):
    sums = [] # Across all agents
    for episode_results in results:
        agents_sums = list(itertools.chain.from_iterable([
            step_results.values() for step_results in episode_results
        ]))
        sums.append(sum(agents_sums))
    return sums

def create_centralized_agent(policy_name, env):
    needs_conv = env.get_needs_conv()
    num_obs = env.get_num_obs() if needs_conv else\
        (1, env.get_num_obs()[::-1][0] * (len(env.get_env_agents())))
    num_actions = (env.get_num_actions()) ** (len(env.get_env_agents()))    
    mapping_fn = lambda x: x.flatten() if not needs_conv else None
    
    if policy_name == 'pg':
        return Agent(PolicyGradient(num_actions, num_obs, mapping_fn=mapping_fn))
    
    elif policy_name == 'dpg':
        return Agent(DeepPolicyGradient(num_obs, num_actions, is_conv=needs_conv,
                                        mapping_fn=mapping_fn)) 
        
    elif policy_name == 'dqn':
        return Agent(DQN(num_obs, num_actions, is_conv=needs_conv,
                                        mapping_fn=mapping_fn)) 
        
    print("Invalid Policy!")       
    return

if __name__ == "__main__":
    main()