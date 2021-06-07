import time

import numpy as np
from numpy.core.arrayprint import _get_format_function
from corners_env import CornersEnv, Action

class GoalAgent:
    """Move toward a goal, 50% for move in x
    """
    def __init__(self, goal):
        self.goal_y, self.goal_x = goal

    def get_action(self, observation):
        y, x = observation[0, :]
        actions = []

        if x > self.goal_x:
            actions.append(Action.left)
        elif x < self.goal_x:
            actions.append(Action.right)

        if y > self.goal_y:
            actions.append(Action.up)
        elif y < self.goal_y:
            actions.append(Action.down)
        
        if not actions:
            # If we reached the goal
            actions.append(Action.noop)

        return np.random.choice(actions)

class EconomicGoalAgent(GoalAgent):
    """Like Goal agent, also also bids and the goal
    can be set later
    """
    def __init__(self, position=[0, 0]):
        super().__init__([0,0])
        self.pos = position # We need to know to position for the bidding:

    def reset(self, observation):
        self.pos = observation[0, :]

    def get_bid(self, suggested_goal):
        # Manhatten Distance
        return abs(self.pos[0] - suggested_goal[0]) + abs(self.pos[1] - suggested_goal[1])

    def set_task(self, task):
        self.goal_y, self.goal_x = task

    

def main():
    # Test the agents:
    env = CornersEnv()
    obs = env.reset()
    goal_agents = {
        'A': GoalAgent([0, 0]),
        'B': GoalAgent([0, 4]),
        'C': GoalAgent([4, 0]),
        'D': GoalAgent([4, 4]),
        'E': GoalAgent([2, 2])
    }
    done = {'temp_agent': False}

    while(not all(value == True for value in done.values())):
        actions = {agent: goal_agents[agent].get_action(obs[agent]) for agent in env.agents}
        obs, _, done, _ = env.step(actions)
        env.render()
        time.sleep(0.2)


if __name__ == "__main__":
    main()