import numpy as np
from gym import spaces
from itertools import chain
from enum import IntEnum

BOARD_SIZE = 5
CORNER_REWARD = 100
CENTER_REWARD = 10
STEP_REWARD = -1

BOARD_UP =   "╔═══════════╗"
BOARD_LINE = "║           ║"
BOARD_DOWN = "╚═══════════╝"

class Action(IntEnum):noop=0;up=1;right=2;down=3;left=4

class CornersEnv:

    def __init__(self):
        self.agents = ['A', 'B', 'C', 'D', 'E']
        
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}

        # Observation: the pos of all other agents (4 agent, 2 coordinated each)
        self.observation_spaces = {agent: spaces.MultiDiscrete([5]*8) for agent in self.agents}
 
    def reset(self):
        # Random Start:
        self.pos = {agent: np.random.choice(range(BOARD_SIZE), size=(2,)) for agent in self.agents}
        
    def render(self):
        lines = [list(BOARD_LINE) + ['\n'] for i in range(5)]
        for agent in self.agents:
            pos = self.pos[agent]
            lines[pos[0]][2 + 2*pos[1]] = agent
        board_list = list(BOARD_UP) + ['\n'] + list(chain.from_iterable(lines)) + list(BOARD_DOWN) + ['\n']
        print("".join(board_list))
        
    def step(self, actions):
        for agent in self.agents:
            if actions[agent] == Action.noop:
                continue
            elif actions[agent] == Action.up:
                new_pos = self.pos[agent] + [-1, 0]
            elif actions[agent] == Action.right:
                new_pos = self.pos[agent] + [0, 1]
            elif actions[agent] == Action.down:
                new_pos = self.pos[agent] + [1, 0]
            elif actions[agent] == Action.left:
                new_pos = self.pos[agent] + [0, -1]

            if self._is_legal_pos(new_pos):
                self.pos[agent] = new_pos

        if self._is_episode_over():
            rewards = {agent: CORNER_REWARD for agent in self.agents}
            dones = {agent: True for agent in self.agents}
        else:
            rewards = {agent: STEP_REWARD for agent in self.agents}
            dones = {agent: False for agent in self.agents}

        info = {}
        observations = {agent: np.array([pos for pos in self.pos if pos != self.pos[agent]])}
        return observations, rewards, dones, info


    def _is_legal_pos(self, pos):
        already_occupied = any([pos[0] == p[0] and pos[1] == p[1] for p in self.pos.values()])
        return not already_occupied and (pos < BOARD_SIZE).all() and (pos >= 0).all()

    def _is_episode_over(self):
        target_pos = [
            [0,0], [4, 0], [0, 4], [4, 4], [2, 2]
        ]
        return all([
            any([pos[0] == target[0] and pos[1] == target[1] for pos in self.pos])
        for target in target_pos])