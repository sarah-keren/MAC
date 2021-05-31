from MAC.control.controller import Controller


# chooses a random action for of each agent
class Decentralized(Controller):

    def __init__(self, agents, env):
        # initialize super class
        super().__init__(env, agents)

    def get_joint_action(self, observation):

        joint_action = {}
        for agent_name in self.agents.keys():
            action = self.agents[agent_name].get_action(observation[agent_name])
            joint_action[agent_name] = action

        return joint_action

    def train(self, max_episode_len, num_episodes, batch_size=0):
        episode_rewards = [0.0]

        print("Starting training...")
        for i in range(num_episodes):
            obs = self.environment.reset()
            ep_reward = 0.0
            agent_rewards = {agent_name: 0.0 for agent_name in self.agents.keys()}
            for _ in range(max_episode_len):
                actions = {agent_name: self.agents[agent_name].get_train_action(obs[agent_name]) for
                               agent_name in self.agents.keys()}
                new_obs, rewards, done, info = self.environment.step(actions)
                for agent_name in self.agents.keys():
                    self.agents[agent_name].update_step(obs[agent_name], actions[agent_name],
                                                            new_obs[agent_name], rewards[agent_name], done[agent_name])
                    ep_reward += rewards[agent_name]

                if batch_size > 0:
                    for agent_name in self.agents.keys():
                        self.agents[agent_name].update_episode(batch_size)

                obs = new_obs
                terminal = False
                terminal = all(value == True for value in done.values())

                if terminal:
                    break

            episode_rewards.append(ep_reward)

            if batch_size > 0:
                for agent_name in self.agents.keys():
                    self.agents[agent_name].update_episode()


























































