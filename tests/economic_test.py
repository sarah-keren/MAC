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

    return env, basic_agent_class, tasks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--env',
        required=True,
        choices=['corners'],
        help='Environment to run test on.'
        )
    return parser.parse_args()

if __name__ == "__main__":
    main()