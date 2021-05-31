class Agent():

    def __init__(self, decision_maker):
        self.decision_maker = decision_maker

    def get_action(self, observation):
        return self.decision_maker.get_action(observation)

    def get_train_action(self, observation):
        return self.decision_maker.get_train_action(observation)

    def update_step(self, obs, action,new_obs, reward, done):
        self.decision_maker.update_step(obs, action,new_obs, reward, done)

    def update_episode(self, batch_size=0):
        if batch_size == 0:
            self.decision_maker.update_episode()
        else:
            self.decision_maker.update_episode(batch_size)