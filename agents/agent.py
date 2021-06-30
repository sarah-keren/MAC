class Agent:

    def __init__(self, decision_maker):
        self.decision_maker = decision_maker

    def get_decision_maker(self):
        return self.decision_maker


class DecisionMaker:

    def __init__(self):
        pass

    def get_action(self, observation):
        pass

    def get_train_action(self, observation):
        pass

    def update_step(self, obs, action,new_obs, reward, done):
        pass

    def update_episode(self, batch_size=0):
        pass
