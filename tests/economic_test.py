import argparse
import sys
import numpy as np

sys.path.append('..')
from MAC.control.controller_economic import EconomicControl

def main():
    args = parse_args()
    env, agent_class, tasks = set_env(args.env)
    economic_control_test(env, agent_class, tasks)

def economic_control_test(env, agent_class, tasks):
    print("Running economic control test:")
    agents = {agent: agent_class() for agent in env.agents}
    controller = EconomicControl(env, agents, tasks)
    controller.run(float('inf'))

def set_env(environment_name):
    print('Initializing environment...')
    if environment_name == 'corners':
        sys.path.append('environments/corners')
        from corners_env import CornersEnv
        env = CornersEnv()
        from corners_agents import EconomicGoalAgent
        basic_agent_class = EconomicGoalAgent

        tasks = [[0,0], [0, 4], [4, 0], [4, 4], [2,2]] # The goal position

    if environment_name == 'taxi':
        sys.path.append('../environments/MultiTaxiEnv')
        from MAC.environments.MultiTaxiEnv.taxi_environment import TaxiEnv
        from MAC.environments.MultiTaxiEnv.taxi_task_agent import TaxiTaskAgent
        
        num_taxis, num_passengers = 2, 2 # For readability
        env = TaxiEnv(num_taxis = num_taxis, num_passengers = num_passengers)
        # Make sure it works with our API:
        env.agents = env.taxis_names
        env.action_spaces = {
            agent_name: env.action_space for agent_name in env.agents
        }
        env.observation_spaces = {
            agent_name: env.observation_space for agent_name in env.agents
        }
        env.possible_agents = [agent for agent in env.agents]
        basic_agent_class = TaxiTaskAgent
        
        # Calculating tasks from the initial observation
        obs = env.reset()['taxi_1'][0] # We need the passenger information
        print(obs)
        tasks = [] # Tasks are a tuple (passenger_position, passenger_dest), each of them is (x,y)
        pass_pos =  [
            (obs[num_taxis*3+i*2], obs[num_taxis*3+i*2+1]) 
            for i in range(num_passengers)
            ]
        pass_dest = [
            (obs[num_taxis*3+num_passengers*2+i*2], obs[num_taxis*3+num_passengers*2+i*2+1])
            for i in range(num_passengers)
            ]
        tasks = list(zip(pass_pos, pass_dest))
    
    return env, basic_agent_class, tasks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--env',
        required=True,
        choices=['corners', 'taxi'],
        help='Environment to run test on.'
        )
    return parser.parse_args()

if __name__ == "__main__":
    main()