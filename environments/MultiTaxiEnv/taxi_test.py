# Bad imports to fix paths: see https://stackoverflow.com/questions/30669474/beyond-top-level-package-error-in-relative-import
import sys; sys.path.append('../..')

from MultiTaxiWrapper import MultiTaxiWrapper
from control.controls import CentralizedControl, DecentralizedControl
from taxi_agents import RandomNTaxisAgent

def main():
    print(f"\n{'*'*80}\nStarting Tests:\n{'*'*80}\n")
    centralized_random_test()
    decentralized_random_test()

def centralized_random_test():
    print(f"Running centralized random test:\n")

    num_taxis = 2
    
    print('Initializing environment...')
    env = MultiTaxiWrapper(2, 1)

    agent = RandomNTaxisAgent(['taxi_1', 'taxi_2'])
    controller = CentralizedControl(env, agent)
    controller.run()

def decentralized_random_test():
    print(f"Running decentralized random test:\n")
    
    print('Initializing environment...')
    env = MultiTaxiWrapper(2, 1)

    agent1 = RandomNTaxisAgent(['taxi_1'])
    agent2 = RandomNTaxisAgent(['taxi_2'])
    controller = DecentralizedControl(env, [agent1, agent2])
    controller.run()

if __name__ == "__main__":
    main()