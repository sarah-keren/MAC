import time

import numpy as np

class TaxiTaskAgent():
    """Like Goal agent, also also bids and the goal
    can be set later
    """
    def __init__(self, position=[0, 0]):
        self.pos = position # We need to know to position for the bidding:

    def reset(self, observation):
        self.pos = tuple(observation[0][:2])
    
    def get_bid(self, suggested_task):
        # Manhatten Distances Sum to the passenger and then to the destination
        manhatten = lambda pos1, pos2: abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        pass_loc, dest_loc = suggested_task
        return manhatten(self.pos, pass_loc) + manhatten(pass_loc, dest_loc)

    def set_task(self, task):
        self.pass_loc, self.dest_loc = task

    def get_action(self, observation):
        return 0

    def navigate(self, dest):
        pass