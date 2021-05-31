from MAC.control.controller import Controller
import random
import MAC.control.utils


# chooses a random action for of its agents
class Centralized(Controller):

    def __init__(self, agents, env, decision_maker, observation_filter = None):

        self.decision_maker = decision_maker
        self.observation_filter = observation_filter
        # initialize super class
        super().__init__(env, agents)

    # randomly select a joint action
    def get_joint_action(self):

        cur_observations = []
        for index, agent in enumerate(self.agents):
            cur_observation = agent.get_observation(self.environment.state, index)
            cur_observations.append(cur_observation)


        state = control.utils._decode_state(cur_observations)

        # if the controller has partial observability - filter the observations
        if self.observation_filter is not None:
            filtered_state = self.observation_filter.get_filtered_observation(state)

        joint_action = self.decision_maker.get_decision(self.environment, filtered_state)
        decoded_joint_action = control.utils._decode_action(joint_action, self.environment.num_actions,self.environment.num_agents)
        return decoded_joint_action

    def get_random_action(self, state):
        num_of_joint_actions = self.environment.get_num_jointactions()
        return random.randrange(num_of_joint_actions)


    def is_done(self):
        return False























































